//! S2 slice: building one [`Observation`] per turn from a task signature.
//!
//! The segmenter (S1) only decides, from fingerprints, when a conversation
//! changes direction. This slice extracts what a turn looks like on each
//! [`Dimension`] from the offline [`TaskSignature`] and the topic labels of the
//! message, following a pinned vocabulary (v1):
//!
//! * **Harness** — the provider id serving the turn.
//! * **Topic** — the message's topic labels (empty means the axis is absent).
//! * **Problem** — the task class key, plus `"retry"` on any attempt after the
//!   first.
//! * **Resolution** — the task class key (how the problem is being solved).
//! * **Understanding** — `"understood"` on the first attempt, `"uncertain"`
//!   after.
//!
//! Nothing here reads a message body, calls a model or touches the network: the
//! labels come straight from the signature and the topic slice. Keeping this
//! extraction separate from the segment decision is what makes the vocabulary
//! testable and the segmentation replayable.

use std::collections::BTreeMap;

use super::segment::{Dimension, Fingerprint, Observation};
use super::signature::TaskSignature;

/// Turns one turn of a conversation into an [`Observation`], applying the
/// pinned vocabulary. Only non-empty fingerprints are recorded: an axis with
/// nothing to say is absent rather than present-but-empty.
pub fn observation(
    turn: u32,
    provider_id: &str,
    signature: &TaskSignature,
    topic: &[String],
) -> Observation {
    let class_key = signature.class.key();
    let mut prints = BTreeMap::new();

    let harness = Fingerprint::new([provider_id]);
    if !harness.is_empty() {
        prints.insert(Dimension::Harness, harness);
    }

    let topic_print = Fingerprint::new(topic.iter().map(String::as_str));
    if !topic_print.is_empty() {
        prints.insert(Dimension::Topic, topic_print);
    }

    let mut problem_labels = vec![class_key.clone()];
    if signature.attempt > 1 {
        problem_labels.push("retry".to_owned());
    }
    let problem = Fingerprint::new(problem_labels);
    if !problem.is_empty() {
        prints.insert(Dimension::Problem, problem);
    }

    let resolution = Fingerprint::new([class_key.as_str()]);
    if !resolution.is_empty() {
        prints.insert(Dimension::Resolution, resolution);
    }

    let understanding = Fingerprint::new([if signature.attempt == 1 {
        "understood"
    } else {
        "uncertain"
    }]);
    if !understanding.is_empty() {
        prints.insert(Dimension::Understanding, understanding);
    }

    Observation { turn, prints }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::resolver::Role;

    fn signature(attempt: u32) -> TaskSignature {
        TaskSignature {
            role: Role::Executor,
            class: crate::chat::provider::cognitive::signature::TaskClass::Chat(
                crate::chat::provider::cognitive::signature::ChatIntent::Debug,
            ),
            complexity: 1,
            context_need_tokens: 8_000,
            needs_tools: true,
            needs_images: false,
            budget_remaining_usd: None,
            attempt,
            project_slug: None,
            persona: None,
        }
    }

    fn labels(print: &Fingerprint) -> Vec<&str> {
        print.labels().collect()
    }

    #[test]
    fn every_axis_is_populated_for_a_chat_turn() {
        let observation = observation(
            7,
            "deepseek",
            &signature(1),
            &["Relay".to_owned(), "provider".to_owned()],
        );
        assert_eq!(observation.turn, 7);
        assert_eq!(observation.prints.len(), 5);
        assert_eq!(
            labels(observation.prints.get(&Dimension::Harness).unwrap()),
            ["deepseek"]
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Topic).unwrap()),
            ["provider", "relay"]
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Problem).unwrap()),
            ["chat.debug"]
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Resolution).unwrap()),
            ["chat.debug"]
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Understanding).unwrap()),
            ["understood"]
        );
    }

    #[test]
    fn a_retry_adds_the_retry_label() {
        let observation = observation(3, "deepseek", &signature(2), &[]);
        assert_eq!(
            labels(observation.prints.get(&Dimension::Problem).unwrap()),
            ["chat.debug", "retry"]
        );
        // The resolution keeps only the class key: a retry does not change how it
        // is solved, only what problem it is.
        assert_eq!(
            labels(observation.prints.get(&Dimension::Resolution).unwrap()),
            ["chat.debug"]
        );
    }

    #[test]
    fn understanding_flips_to_uncertain_after_attempt_one() {
        assert_eq!(
            labels(
                observation(0, "a", &signature(1), &[])
                    .prints
                    .get(&Dimension::Understanding)
                    .unwrap()
            ),
            ["understood"]
        );
        assert_eq!(
            labels(
                observation(0, "a", &signature(2), &[])
                    .prints
                    .get(&Dimension::Understanding)
                    .unwrap()
            ),
            ["uncertain"]
        );
    }

    #[test]
    fn an_empty_topic_omits_the_topic_axis() {
        let observation = observation(0, "a", &signature(1), &[]);
        assert!(!observation.prints.contains_key(&Dimension::Topic));
        assert!(observation.prints.get(&Dimension::Topic).is_none());
    }

    #[test]
    fn labels_are_normalized_and_stable() {
        let observation = observation(
            0,
            " DeepSeek ",
            &signature(1),
            &["  Relay ".to_owned(), "RELAY".to_owned(), "".to_owned()],
        );
        assert_eq!(
            labels(observation.prints.get(&Dimension::Harness).unwrap()),
            ["deepseek"]
        );
        // Empty and duplicate labels are dropped; the rest are trimmed and
        // lower-cased, and the fingerprint keeps them in sorted order.
        assert_eq!(
            labels(observation.prints.get(&Dimension::Topic).unwrap()),
            ["relay"]
        );
    }
}
