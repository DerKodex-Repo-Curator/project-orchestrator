//! Neo4j side of the conversation segments: the episodic memory of a chat.
//!
//! ```text
//! (:ConversationSegment {id, conversation, dimension, start_turn, json})  one per stretch
//! (:ConversationBoundary {conversation, dimension, at_turn, from, to, json})  one per change
//! (:ConversationSegment)-[:PRECEDES]->(:ConversationSegment)               the chain, per boundary
//! ```
//!
//! Both documents carry their full `serde_json` payload (snake_case, as the pure
//! model already serialises) plus the scalar fields a reader can filter and order
//! by. [`Neo4jSegmentStore`] adapts any [`GraphStore`] to the [`SegmentStore`]
//! the wiring depends on; the segment ids are deterministic, so a re-`put` merges
//! onto the same nodes instead of duplicating them.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use neo4rs::query;

use super::client::Neo4jClient;
use super::traits::GraphStore;
use crate::chat::provider::cognitive::segment::{Boundary, Segment};
use crate::chat::provider::cognitive::segment_store::SegmentStore;

fn segment_from_json(json: &str) -> Result<Segment> {
    serde_json::from_str(json).context("unreadable conversation segment")
}

fn boundary_from_json(json: &str) -> Result<Boundary> {
    serde_json::from_str(json).context("unreadable conversation boundary")
}

/// The `MERGE` of a `ConversationSegment` keys on `(conversation, id)`, never `id` alone:
/// segment ids are derived from the conversation seed, and two seed-0 sessions produce the
/// same ids, so `id` alone would merge the segments of two different conversations.
const SEGMENT_MERGE: &str = "\
MERGE (s:ConversationSegment {conversation: $conversation, id: $id}) \
SET s.conversation = $conversation, s.dimension = $dimension, \
    s.start_turn = $start_turn, s.json = $json";

/// Orders stored segments by their `Dimension` enum order then `start_turn` — NOT the
/// lexicographic order of the serialised axis string, which would put `problem` and
/// `resolution` before `topic`.
fn order_segments(mut segments: Vec<Segment>) -> Vec<Segment> {
    segments.sort_by(|a, b| (a.dimension, a.start_turn).cmp(&(b.dimension, b.start_turn)));
    segments
}

/// Orders stored boundaries by the turn they fell on, then by axis (enum order). The exact
/// write order is not expressible in Cypher without a stored sequence number, so this
/// deterministic `(at_turn, dimension)` order is the documented fallback.
fn order_boundaries(mut boundaries: Vec<Boundary>) -> Vec<Boundary> {
    boundaries.sort_by(|a, b| (a.at_turn, a.dimension).cmp(&(b.at_turn, b.dimension)));
    boundaries
}

impl Neo4jClient {
    /// Stores (replacing) the segments of a conversation. Segments are written before
    /// boundaries (the caller does `put_segments` then `put_boundaries`), so the
    /// `PRECEDES` links the boundaries create have their endpoints.
    pub async fn put_segments_impl(&self, conversation: &str, segments: &[Segment]) -> Result<()> {
        // The caller "replaces": drop what this conversation had, then write the
        // new set. The deterministic ids make a re-`put` land on the same nodes.
        let clear =
            query("MATCH (s:ConversationSegment {conversation: $conversation}) DETACH DELETE s")
                .param("conversation", conversation.to_string());
        self.graph.run(clear).await?;
        for segment in segments {
            let json = serde_json::to_string(segment)?;
            let q = query(SEGMENT_MERGE)
                .param("id", segment.id.as_str().to_string())
                .param("conversation", conversation.to_string())
                .param("dimension", segment.dimension.as_str().to_string())
                .param("start_turn", i64::from(segment.start_turn))
                .param("json", json);
            self.graph.run(q).await?;
        }
        Ok(())
    }

    /// The stored segments of a conversation, ordered by `Dimension` enum order then first
    /// turn (not the lexicographic order of the serialised axis string).
    pub async fn get_segments_impl(&self, conversation: &str) -> Result<Vec<Segment>> {
        let q = query(
            "MATCH (s:ConversationSegment {conversation: $conversation}) \
             RETURN s.json AS json",
        )
        .param("conversation", conversation.to_string());
        let mut result = self.graph.execute(q).await?;
        let mut segments = Vec::new();
        while let Some(row) = result.next().await? {
            match segment_from_json(&row.get::<String>("json")?) {
                Ok(segment) => segments.push(segment),
                Err(error) => tracing::warn!(%error, "conversation segment skipped"),
            }
        }
        Ok(order_segments(segments))
    }

    /// Stores (replacing) the boundaries of a conversation, and links the chain
    /// each boundary cuts (`from` -> `to`). The caller writes the segments FIRST
    /// (`put_segments_impl`), so the endpoints exist; a link that still cannot be made is
    /// logged, never an error: the boundary is already stored.
    pub async fn put_boundaries_impl(
        &self,
        conversation: &str,
        boundaries: &[Boundary],
    ) -> Result<()> {
        let clear =
            query("MATCH (b:ConversationBoundary {conversation: $conversation}) DETACH DELETE b")
                .param("conversation", conversation.to_string());
        self.graph.run(clear).await?;
        for boundary in boundaries {
            let json = serde_json::to_string(boundary)?;
            let q = query(
                "MERGE (b:ConversationBoundary {conversation: $conversation, dimension: $dimension, at_turn: $at_turn}) \
                 SET b.from = $from, b.to = $to, b.json = $json",
            )
            .param("conversation", conversation.to_string())
            .param("dimension", boundary.dimension.as_str().to_string())
            .param("at_turn", i64::from(boundary.at_turn))
            .param("from", boundary.from.as_str().to_string())
            .param("to", boundary.to.as_str().to_string())
            .param("json", json);
            self.graph.run(q).await?;

            let link = query(
                "MATCH (from:ConversationSegment {conversation: $conversation, id: $from}) \
                 MATCH (to:ConversationSegment {conversation: $conversation, id: $to}) \
                 MERGE (from)-[:PRECEDES]->(to)",
            )
            .param("conversation", conversation.to_string())
            .param("from", boundary.from.as_str().to_string())
            .param("to", boundary.to.as_str().to_string());
            if let Err(error) = self.graph.run(link).await {
                tracing::warn!(conversation, %error, "segment boundary link not made");
            }
        }
        Ok(())
    }

    /// The stored boundaries of a conversation, ordered by the turn they fell on then axis
    /// (enum order) — the documented fallback for the write order, which Cypher cannot
    /// express without a stored sequence number.
    pub async fn get_boundaries_impl(&self, conversation: &str) -> Result<Vec<Boundary>> {
        let q = query(
            "MATCH (b:ConversationBoundary {conversation: $conversation}) \
             RETURN b.json AS json",
        )
        .param("conversation", conversation.to_string());
        let mut result = self.graph.execute(q).await?;
        let mut boundaries = Vec::new();
        while let Some(row) = result.next().await? {
            match boundary_from_json(&row.get::<String>("json")?) {
                Ok(boundary) => boundaries.push(boundary),
                Err(error) => tracing::warn!(%error, "conversation boundary skipped"),
            }
        }
        Ok(order_boundaries(boundaries))
    }
}

/// [`SegmentStore`] over the graph.
#[derive(Clone)]
pub struct Neo4jSegmentStore {
    graph: Arc<dyn GraphStore>,
}

impl Neo4jSegmentStore {
    /// A store over `graph`.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }
}

#[async_trait]
impl SegmentStore for Neo4jSegmentStore {
    async fn put_boundaries(&self, conversation: &str, boundaries: &[Boundary]) -> Result<()> {
        self.graph.put_boundaries(conversation, boundaries).await
    }

    async fn boundaries(&self, conversation: &str) -> Result<Vec<Boundary>> {
        self.graph.get_boundaries(conversation).await
    }

    async fn put_segments(&self, conversation: &str, segments: &[Segment]) -> Result<()> {
        self.graph.put_segments(conversation, segments).await
    }

    async fn segments(&self, conversation: &str) -> Result<Vec<Segment>> {
        self.graph.get_segments(conversation).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::cognitive::segment::{
        Dimension, Fingerprint, Observation, SegmentPolicy, Segmenter,
    };
    use crate::neo4j::mock::MockGraphStore;

    fn obs(turn: u32, prints: &[(Dimension, &[&str])]) -> Observation {
        Observation {
            turn,
            prints: prints
                .iter()
                .map(|(dimension, labels)| (*dimension, Fingerprint::new(labels.iter().copied())))
                .collect(),
        }
    }

    fn replay() -> Segmenter {
        // Two turns on one axis that switch harness: a single boundary, two segments.
        let history = vec![
            obs(0, &[(Dimension::Harness, &["deepseek"])]),
            obs(1, &[(Dimension::Harness, &["claude-code"])]),
        ];
        Segmenter::replay(SegmentPolicy::default(), 7, &history).unwrap()
    }

    // The Cypher above is exercised by the Neo4j CI job only; here the adapter is
    // checked over the mock graph, which wraps the in-memory store.
    #[tokio::test]
    async fn the_adapter_reads_and_writes_segments_and_boundaries_through_the_graph() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let store = Neo4jSegmentStore::new(graph.clone());
        let segmenter = replay();
        let segments = segmenter.segments();
        let boundaries = segmenter.boundaries().to_vec();

        assert!(store.segments("conv").await.unwrap().is_empty());
        assert!(store.boundaries("conv").await.unwrap().is_empty());

        store.put_segments("conv", &segments).await.unwrap();
        store.put_boundaries("conv", &boundaries).await.unwrap();

        assert_eq!(store.segments("conv").await.unwrap(), segments);
        assert_eq!(store.boundaries("conv").await.unwrap(), boundaries);
        // Another conversation is still empty.
        assert!(store.segments("other").await.unwrap().is_empty());
        assert!(store.boundaries("other").await.unwrap().is_empty());

        // Putting again replaces, never accumulates.
        store.put_segments("conv", &segments).await.unwrap();
        store.put_boundaries("conv", &boundaries).await.unwrap();
        assert_eq!(store.segments("conv").await.unwrap(), segments);
        assert_eq!(store.boundaries("conv").await.unwrap(), boundaries);
    }

    #[test]
    fn the_segment_merge_keys_on_conversation_and_id() {
        assert!(
            SEGMENT_MERGE
                .contains("MERGE (s:ConversationSegment {conversation: $conversation, id: $id})"),
            "the merge key must be (conversation, id), never id alone: {SEGMENT_MERGE}"
        );
    }

    #[test]
    fn order_segments_puts_topic_before_problem_not_lexicographic() {
        // Two turns on Topic and Problem, no boundary: one segment per axis. Enum order is
        // Topic (1) then Problem (2); the serialised strings sort the OTHER way
        // ("problem" < "topic").
        let history = vec![
            obs(
                0,
                &[
                    (Dimension::Topic, &["relay"]),
                    (Dimension::Problem, &["chat.debug"]),
                ],
            ),
            obs(
                1,
                &[
                    (Dimension::Topic, &["relay"]),
                    (Dimension::Problem, &["chat.debug"]),
                ],
            ),
        ];
        let segmenter = Segmenter::replay(SegmentPolicy::default(), 7, &history).unwrap();
        let mut segments = segmenter.segments();
        assert_eq!(segments.len(), 2);
        segments.reverse();
        let dims: Vec<Dimension> = order_segments(segments)
            .iter()
            .map(|s| s.dimension)
            .collect();
        assert_eq!(dims, [Dimension::Topic, Dimension::Problem]);
    }

    #[test]
    fn order_boundaries_sorts_by_turn_then_axis() {
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        // A harness switch at turn 2, then a topic drift confirmed at turn 4 (placed at 3).
        for (turn, harness, topic) in [
            (0, "a", "x"),
            (1, "a", "x"),
            (2, "b", "x"),
            (3, "b", "y"),
            (4, "b", "y"),
        ] {
            segmenter
                .observe(&obs(
                    turn,
                    &[
                        (Dimension::Harness, &[harness]),
                        (Dimension::Topic, &[topic]),
                    ],
                ))
                .unwrap();
        }
        let mut boundaries = segmenter.boundaries().to_vec();
        assert_eq!(boundaries.len(), 2);
        boundaries.reverse();
        let ordered = order_boundaries(boundaries);
        assert_eq!(
            ordered
                .iter()
                .map(|b| (b.at_turn, b.dimension))
                .collect::<Vec<_>>(),
            [(2, Dimension::Harness), (3, Dimension::Topic)]
        );
    }

    /// Two sessions without an id collapse to seed 0, hence the SAME segment ids: the store
    /// must key on (conversation, id), so writing one conversation never merges into or
    /// overwrites the other.
    #[tokio::test]
    async fn two_seed_zero_conversations_keep_their_segments_apart() {
        let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let store = Neo4jSegmentStore::new(graph);
        let history = vec![
            obs(0, &[(Dimension::Harness, &["deepseek"])]),
            obs(1, &[(Dimension::Harness, &["claude-code"])]),
        ];
        let a = Segmenter::replay(SegmentPolicy::default(), 0, &history).unwrap();
        let b = Segmenter::replay(SegmentPolicy::default(), 0, &history).unwrap();
        assert_eq!(
            a.segments()[0].id,
            b.segments()[0].id,
            "seed 0 collides by design"
        );
        store.put_segments("conv-a", &a.segments()).await.unwrap();
        store.put_segments("conv-b", &b.segments()).await.unwrap();
        assert_eq!(
            store.segments("conv-a").await.unwrap().len(),
            a.segments().len()
        );
        assert_eq!(
            store.segments("conv-b").await.unwrap().len(),
            b.segments().len()
        );
        assert!(store.segments("other").await.unwrap().is_empty());
    }
}
