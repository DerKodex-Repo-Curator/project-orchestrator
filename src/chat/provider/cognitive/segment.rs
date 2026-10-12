//! Conversation segments (v0.0.18, S1): where a conversation changes direction.
//!
//! A conversation is not one block. It is a chain of stretches, each with its own
//! direction, and the direction can change along several independent axes at once:
//! the harness that serves it, the subject, the problem being worked, the way it is
//! being solved, how well it is understood. This module is the PURE MODEL of that:
//! it takes one [`Observation`] per turn and says where a [`Segment`] ends and the
//! next one starts, on each [`Dimension`] separately.
//!
//! Nothing here reads a message, calls a model or touches the network. What a turn
//! looks like on each axis is a [`Fingerprint`] (a small set of labels) that a later
//! slice (S2) extracts from the stored events; this slice only decides, from those
//! fingerprints, when a direction has really changed. Keeping the two apart is what
//! makes the decision testable, and replayable.
//!
//! * **Hysteresis, not a one-turn latch.** A single off-topic turn is not a change of
//!   direction. A boundary needs [`Threshold::confirmations`] consecutive observations
//!   that all sit at least [`Threshold::distance_permille`] away from the current
//!   segment, and it is then placed at the FIRST of them (the segment really changed
//!   there, not when it was confirmed). One observation back within the threshold
//!   cancels the run. The harness axis confirms at once: a switch is explicit.
//! * **No evidence is not a change.** An empty fingerprint means the turn said nothing
//!   on that axis. It neither opens a segment, nor deviates, nor confirms.
//! * **Replayable.** Integer arithmetic only (distances are permille), ordered maps,
//!   and segment ids derived by a fixed hash of `(seed, dimension, start turn)`: the
//!   same seed and the same observations give the same segments and boundaries, byte
//!   for byte, on any machine. The caller derives the seed from the conversation or
//!   decision id (constraint: every automatic decision is attributable and replayable).
//! * **Atomic.** An observation that is refused (turn not strictly increasing) leaves
//!   the state exactly as it was.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Largest distance: nothing in common.
pub const PERMILLE_MAX: u16 = 1_000;
/// Most labels a segment keeps in its fingerprint. A segment that keeps absorbing never
/// grows without bound; labels beyond the cap are not added (the first ones win, in the
/// fingerprint's sorted order, so the result does not depend on arrival order).
pub const MAX_FINGERPRINT_LABELS: usize = 64;

/// An axis along which a conversation can change direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dimension {
    /// The provider/engine serving the conversation.
    Harness,
    /// What the conversation is about.
    Topic,
    /// The problem being worked.
    Problem,
    /// The way that problem is being solved.
    Resolution,
    /// How well the situation is understood.
    Understanding,
}

impl Dimension {
    /// Every dimension, in the fixed order observations are processed in.
    pub const ALL: [Self; 5] = [
        Self::Harness,
        Self::Topic,
        Self::Problem,
        Self::Resolution,
        Self::Understanding,
    ];

    /// Stable name (feeds the segment id: never rename).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Harness => "harness",
            Self::Topic => "topic",
            Self::Problem => "problem",
            Self::Resolution => "resolution",
            Self::Understanding => "understanding",
        }
    }
}

/// What a turn looks like on one axis: a set of normalised labels.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Fingerprint(BTreeSet<String>);

impl Fingerprint {
    /// Builds a fingerprint. Labels are trimmed and lower-cased; empty ones are dropped.
    pub fn new<I, S>(labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self(
            labels
                .into_iter()
                .map(|label| label.as_ref().trim().to_lowercase())
                .filter(|label| !label.is_empty())
                .collect(),
        )
    }

    /// Whether the turn said nothing on this axis.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The labels, in sorted order.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }

    /// Jaccard distance to `other`, in permille: 0 = same set, 1000 = nothing in common.
    /// Two empty sets are at distance 0 (no evidence is no difference). The similarity is
    /// rounded DOWN, so the distance is rounded up: a borderline pair counts as farther.
    pub fn distance_permille(&self, other: &Self) -> u16 {
        let common = self.0.intersection(&other.0).count();
        let total = self.0.len() + other.0.len() - common;
        if total == 0 {
            return 0;
        }
        let similarity = common * usize::from(PERMILLE_MAX) / total;
        let similarity = u16::try_from(similarity).unwrap_or(PERMILLE_MAX);
        PERMILLE_MAX - similarity.min(PERMILLE_MAX)
    }

    /// Adds `other`'s labels, up to [`MAX_FINGERPRINT_LABELS`].
    fn merge(&mut self, other: &Self) {
        for label in &other.0 {
            if self.0.len() >= MAX_FINGERPRINT_LABELS {
                break;
            }
            self.0.insert(label.clone());
        }
    }
}

/// One turn, seen on every axis it has something to say about.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Observation {
    /// Index of the turn in the conversation; strictly increasing across observations.
    pub turn: u32,
    /// What the turn looks like per axis. A missing or empty entry is "no evidence".
    pub prints: BTreeMap<Dimension, Fingerprint>,
}

/// How far, and for how long, a direction must move before it counts as changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Threshold {
    /// Distance, in permille, from which an observation deviates (1..=1000).
    pub distance_permille: u16,
    /// Consecutive deviating observations that confirm a boundary (at least 1).
    pub confirmations: u8,
}

impl Threshold {
    /// A checked threshold.
    pub fn new(distance_permille: u16, confirmations: u8) -> Result<Self, SegmentError> {
        if distance_permille == 0 || distance_permille > PERMILLE_MAX {
            return Err(SegmentError::DistanceOutOfRange(distance_permille));
        }
        if confirmations == 0 {
            return Err(SegmentError::NoConfirmation);
        }
        Ok(Self {
            distance_permille,
            confirmations,
        })
    }
}

/// The thresholds of every axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentPolicy {
    thresholds: BTreeMap<Dimension, Threshold>,
}

impl Default for SegmentPolicy {
    /// The harness changes at once (a switch is explicit); the other axes need two
    /// consecutive turns that sit clearly away from the segment.
    fn default() -> Self {
        let mut thresholds = BTreeMap::new();
        for (dimension, distance, confirmations) in [
            (Dimension::Harness, PERMILLE_MAX, 1),
            (Dimension::Topic, 700, 2),
            (Dimension::Problem, 600, 2),
            (Dimension::Resolution, 600, 2),
            (Dimension::Understanding, 600, 2),
        ] {
            thresholds.insert(
                dimension,
                Threshold {
                    distance_permille: distance,
                    confirmations,
                },
            );
        }
        Self { thresholds }
    }
}

impl SegmentPolicy {
    /// This policy with `dimension` set to `threshold`.
    pub fn with(mut self, dimension: Dimension, threshold: Threshold) -> Self {
        self.thresholds.insert(dimension, threshold);
        self
    }

    /// The threshold of `dimension`.
    pub fn threshold(&self, dimension: Dimension) -> Threshold {
        // The default fills every axis and `with` only replaces: an axis is never absent.
        // The fallback is the strictest threshold, so a hole could only ever under-split.
        self.thresholds
            .get(&dimension)
            .copied()
            .unwrap_or(Threshold {
                distance_permille: PERMILLE_MAX,
                confirmations: u8::MAX,
            })
    }
}

/// Why a boundary was placed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryReason {
    /// Distance of the first deviating observation from the segment it left, in permille.
    pub distance_permille: u16,
    /// The threshold that was crossed.
    pub threshold_permille: u16,
    /// Consecutive deviating observations that confirmed it.
    pub confirmations: u8,
}

/// Stable identifier of a segment: a fixed hash, not a random id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SegmentId(String);

impl SegmentId {
    fn derive(seed: u64, dimension: Dimension, start_turn: u32) -> Self {
        // FNV-1a, 64 bits: fixed by definition, unlike `DefaultHasher`, whose algorithm
        // the standard library does not promise to keep.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            // A separator, so ("ab","c") and ("a","bc") do not collide.
            hash ^= 0xff;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        };
        feed(&seed.to_be_bytes());
        feed(dimension.as_str().as_bytes());
        feed(&start_turn.to_be_bytes());
        Self(format!("seg-{hash:016x}"))
    }

    /// The identifier as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A stretch of the conversation that keeps one direction on one axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    /// Derived from the seed, the axis and the first turn.
    pub id: SegmentId,
    /// The axis.
    pub dimension: Dimension,
    /// First turn of the segment.
    pub start_turn: u32,
    /// Last turn observed in it.
    pub last_turn: u32,
    /// How many observations it absorbed.
    pub observations: u32,
    /// Whether a later segment replaced it.
    pub closed: bool,
    /// What the segment looks like: the labels of the turns it absorbed (capped).
    pub fingerprint: Fingerprint,
    /// Why it started; `None` for the first segment of an axis.
    pub opened_by: Option<BoundaryReason>,
}

impl Segment {
    fn open(
        seed: u64,
        dimension: Dimension,
        start_turn: u32,
        fingerprint: Fingerprint,
        opened_by: Option<BoundaryReason>,
    ) -> Self {
        Self {
            id: SegmentId::derive(seed, dimension, start_turn),
            dimension,
            start_turn,
            last_turn: start_turn,
            observations: 1,
            closed: false,
            fingerprint,
            opened_by,
        }
    }

    fn absorb(&mut self, turn: u32, print: &Fingerprint) {
        self.last_turn = turn;
        self.observations = self.observations.saturating_add(1);
        self.fingerprint.merge(print);
    }
}

/// A change of direction on one axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Boundary {
    /// The axis.
    pub dimension: Dimension,
    /// First turn of the new segment: where the direction really changed.
    pub at_turn: u32,
    /// The segment that ended.
    pub from: SegmentId,
    /// The segment that started.
    pub to: SegmentId,
    /// Why.
    pub reason: BoundaryReason,
}

/// A refused input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SegmentError {
    /// The turn does not come after the last one observed.
    #[error("turn {turn} does not follow turn {last}: observations must be strictly increasing")]
    OutOfOrder {
        /// The refused turn.
        turn: u32,
        /// The last turn observed.
        last: u32,
    },
    /// A threshold distance outside 1..=1000.
    #[error("a distance threshold is between 1 and 1000 permille, got {0}")]
    DistanceOutOfRange(u16),
    /// A threshold that no observation can confirm.
    #[error("a boundary needs at least one confirmation")]
    NoConfirmation,
}

/// An observation that deviates and is waiting to be confirmed.
#[derive(Debug, Clone)]
struct Pending {
    turn: u32,
    print: Fingerprint,
    distance: u16,
}

#[derive(Debug, Default)]
struct Track {
    current: Option<Segment>,
    pending: Vec<Pending>,
}

/// Cuts a conversation into segments, one observation at a time.
#[derive(Debug)]
pub struct Segmenter {
    policy: SegmentPolicy,
    seed: u64,
    tracks: BTreeMap<Dimension, Track>,
    closed: Vec<Segment>,
    boundaries: Vec<Boundary>,
    last_turn: Option<u32>,
}

impl Segmenter {
    /// An empty segmenter. `seed` is derived by the caller from the conversation or
    /// decision id; it only feeds the segment ids.
    pub fn new(policy: SegmentPolicy, seed: u64) -> Self {
        Self {
            policy,
            seed,
            tracks: BTreeMap::new(),
            closed: Vec::new(),
            boundaries: Vec::new(),
            last_turn: None,
        }
    }

    /// Replays a recorded conversation from scratch.
    pub fn replay(
        policy: SegmentPolicy,
        seed: u64,
        observations: &[Observation],
    ) -> Result<Self, SegmentError> {
        let mut segmenter = Self::new(policy, seed);
        for observation in observations {
            segmenter.observe(observation)?;
        }
        Ok(segmenter)
    }

    /// Takes one turn in. Returns the boundaries it confirmed (at most one per axis).
    /// A refused turn changes nothing.
    pub fn observe(&mut self, observation: &Observation) -> Result<Vec<Boundary>, SegmentError> {
        if let Some(last) = self.last_turn {
            if observation.turn <= last {
                return Err(SegmentError::OutOfOrder {
                    turn: observation.turn,
                    last,
                });
            }
        }
        self.last_turn = Some(observation.turn);

        let mut confirmed = Vec::new();
        for dimension in Dimension::ALL {
            let Some(print) = observation.prints.get(&dimension) else {
                continue;
            };
            if print.is_empty() {
                continue;
            }
            if let Some(boundary) = self.observe_axis(dimension, observation.turn, print) {
                self.boundaries.push(boundary.clone());
                confirmed.push(boundary);
            }
        }
        Ok(confirmed)
    }

    fn observe_axis(
        &mut self,
        dimension: Dimension,
        turn: u32,
        print: &Fingerprint,
    ) -> Option<Boundary> {
        let threshold = self.policy.threshold(dimension);
        let seed = self.seed;
        let track = self.tracks.entry(dimension).or_default();

        let Some(mut current) = track.current.take() else {
            track.current = Some(Segment::open(seed, dimension, turn, print.clone(), None));
            return None;
        };

        let distance = current.fingerprint.distance_permille(print);
        if distance < threshold.distance_permille {
            // Back inside the segment: any run of deviations was a detour, not a change.
            track.pending.clear();
            current.absorb(turn, print);
            track.current = Some(current);
            return None;
        }

        track.pending.push(Pending {
            turn,
            print: print.clone(),
            distance,
        });
        if track.pending.len() < usize::from(threshold.confirmations) {
            track.current = Some(current);
            return None;
        }

        // Confirmed: the new segment starts at the FIRST deviating turn.
        let pending = std::mem::take(&mut track.pending);
        let first = pending.first()?;
        let last = pending.last()?;
        let reason = BoundaryReason {
            distance_permille: first.distance,
            threshold_permille: threshold.distance_permille,
            confirmations: threshold.confirmations,
        };
        let mut fingerprint = Fingerprint::default();
        for entry in &pending {
            fingerprint.merge(&entry.print);
        }
        let mut next = Segment::open(
            seed,
            dimension,
            first.turn,
            fingerprint,
            Some(reason.clone()),
        );
        next.last_turn = last.turn;
        next.observations = u32::try_from(pending.len()).unwrap_or(u32::MAX);

        current.closed = true;
        let boundary = Boundary {
            dimension,
            at_turn: first.turn,
            from: current.id.clone(),
            to: next.id.clone(),
            reason,
        };
        self.closed.push(current);
        track.current = Some(next);
        Some(boundary)
    }

    /// The open segment of an axis, if the axis has been observed.
    pub fn current(&self, dimension: Dimension) -> Option<&Segment> {
        self.tracks.get(&dimension)?.current.as_ref()
    }

    /// Every segment, closed and open, ordered by axis then first turn.
    pub fn segments(&self) -> Vec<Segment> {
        let mut all: Vec<Segment> = self.closed.clone();
        all.extend(self.tracks.values().filter_map(|t| t.current.clone()));
        all.sort_by(|a, b| (a.dimension, a.start_turn).cmp(&(b.dimension, b.start_turn)));
        all
    }

    /// The segment `turn` belongs to on `dimension`: the latest one that started at or
    /// before it. A turn that deviates but is not yet confirmed still belongs to the
    /// current segment; it is moved when (and if) the boundary is confirmed.
    pub fn segment_at(&self, dimension: Dimension, turn: u32) -> Option<Segment> {
        self.segments()
            .into_iter()
            .rfind(|s| s.dimension == dimension && s.start_turn <= turn)
    }

    /// Every boundary confirmed so far, in the order they were confirmed.
    pub fn boundaries(&self) -> &[Boundary] {
        &self.boundaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(labels: &[&str]) -> Fingerprint {
        Fingerprint::new(labels.iter().copied())
    }

    fn obs(turn: u32, prints: &[(Dimension, &[&str])]) -> Observation {
        Observation {
            turn,
            prints: prints.iter().map(|(d, l)| (*d, fp(l))).collect(),
        }
    }

    fn topic(turn: u32, labels: &[&str]) -> Observation {
        obs(turn, &[(Dimension::Topic, labels)])
    }

    fn harness(turn: u32, id: &str) -> Observation {
        obs(turn, &[(Dimension::Harness, &[id])])
    }

    fn run(observations: &[Observation]) -> Segmenter {
        Segmenter::replay(SegmentPolicy::default(), 7, observations).unwrap()
    }

    #[test]
    fn fingerprint_normalizes_and_ignores_empty_labels() {
        let print = Fingerprint::new(["  Relay ", "relay", "", "   ", "Routing"]);
        assert_eq!(print.labels().collect::<Vec<_>>(), ["relay", "routing"]);
        assert!(Fingerprint::new(["", " "]).is_empty());
    }

    #[test]
    fn distance_is_jaccard_in_permille() {
        assert_eq!(fp(&["a", "b"]).distance_permille(&fp(&["a", "b"])), 0);
        assert_eq!(fp(&["a"]).distance_permille(&fp(&["b"])), 1_000);
        // 1 label in common out of 3 distinct: similarity 333 (rounded down) -> 667.
        assert_eq!(fp(&["a", "b"]).distance_permille(&fp(&["b", "c"])), 667);
        // Two empty sets say nothing: not a difference.
        assert_eq!(fp(&[]).distance_permille(&fp(&[])), 0);
    }

    #[test]
    fn an_empty_observation_is_no_evidence() {
        let segmenter = run(&[topic(0, &[]), topic(1, &[]), obs(2, &[])]);
        assert!(segmenter.segments().is_empty());
        assert!(segmenter.boundaries().is_empty());
    }

    #[test]
    fn a_harness_change_is_a_boundary_at_once() {
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        assert!(segmenter
            .observe(&harness(0, "deepseek"))
            .unwrap()
            .is_empty());
        assert!(segmenter
            .observe(&harness(1, "deepseek"))
            .unwrap()
            .is_empty());
        let boundaries = segmenter.observe(&harness(2, "claude-code")).unwrap();
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].dimension, Dimension::Harness);
        assert_eq!(boundaries[0].at_turn, 2);
        assert_eq!(boundaries[0].reason.confirmations, 1);
        let before = segmenter.segment_at(Dimension::Harness, 1).unwrap();
        assert!(before.closed);
        assert_eq!(before.last_turn, 1);
        assert_eq!(before.id, boundaries[0].from);
    }

    #[test]
    fn a_single_off_topic_turn_is_not_a_boundary() {
        let segmenter = run(&[
            topic(0, &["relay", "provider"]),
            topic(1, &["relay", "provider"]),
            topic(2, &["weather"]),
            topic(3, &["relay", "provider"]),
        ]);
        assert!(segmenter.boundaries().is_empty());
        let only = segmenter.current(Dimension::Topic).unwrap();
        assert_eq!(
            (only.start_turn, only.last_turn, only.observations),
            (0, 3, 3)
        );
    }

    #[test]
    fn consecutive_deviations_open_a_segment_at_the_first_deviating_turn() {
        let segmenter = run(&[
            topic(0, &["relay", "provider"]),
            topic(1, &["relay", "provider"]),
            topic(2, &["weather", "antibes"]),
            topic(3, &["weather", "antibes"]),
        ]);
        let boundary = &segmenter.boundaries()[0];
        // Confirmed on turn 3, but the direction changed on turn 2.
        assert_eq!(boundary.at_turn, 2);
        assert_eq!(boundary.reason.distance_permille, 1_000);
        assert_eq!(boundary.reason.threshold_permille, 700);
        let old = segmenter.segment_at(Dimension::Topic, 1).unwrap();
        assert_eq!((old.start_turn, old.last_turn, old.closed), (0, 1, true));
        let new = segmenter.segment_at(Dimension::Topic, 3).unwrap();
        assert_eq!((new.start_turn, new.last_turn, new.closed), (2, 3, false));
        assert_eq!(new.opened_by.as_ref(), Some(&boundary.reason));
    }

    #[test]
    fn a_return_inside_the_segment_cancels_the_pending_run() {
        let segmenter = run(&[
            topic(0, &["relay"]),
            topic(1, &["weather"]), // deviates (1 of 2 needed)
            topic(2, &["relay"]),   // back: the run is cancelled
            topic(3, &["weather"]), // deviates again, but alone
        ]);
        assert!(segmenter.boundaries().is_empty());
    }

    #[test]
    fn axes_are_independent() {
        let segmenter = run(&[
            obs(
                0,
                &[
                    (Dimension::Harness, &["deepseek"]),
                    (Dimension::Topic, &["relay"]),
                ],
            ),
            obs(
                1,
                &[
                    (Dimension::Harness, &["claude-code"]),
                    (Dimension::Topic, &["relay"]),
                ],
            ),
        ]);
        let moved: Vec<_> = segmenter.boundaries().iter().map(|b| b.dimension).collect();
        assert_eq!(moved, [Dimension::Harness]);
        assert_eq!(segmenter.current(Dimension::Topic).unwrap().observations, 2);
    }

    #[test]
    fn a_refused_turn_changes_nothing() {
        let mut segmenter = Segmenter::new(SegmentPolicy::default(), 7);
        segmenter.observe(&topic(5, &["relay"])).unwrap();
        let before = segmenter.segments();
        for turn in [5, 3, 0] {
            assert_eq!(
                segmenter.observe(&topic(turn, &["weather"])),
                Err(SegmentError::OutOfOrder { turn, last: 5 })
            );
        }
        assert_eq!(segmenter.segments(), before);
        assert!(segmenter.boundaries().is_empty());
        // And the next valid turn is still accepted.
        segmenter.observe(&topic(6, &["relay"])).unwrap();
    }

    #[test]
    fn replay_is_deterministic_and_ids_follow_the_seed() {
        let history = [
            obs(
                0,
                &[
                    (Dimension::Harness, &["deepseek"]),
                    (Dimension::Topic, &["relay"]),
                ],
            ),
            obs(
                1,
                &[
                    (Dimension::Harness, &["deepseek"]),
                    (Dimension::Topic, &["relay"]),
                ],
            ),
            obs(
                2,
                &[
                    (Dimension::Harness, &["claude-code"]),
                    (Dimension::Topic, &["tests"]),
                ],
            ),
            obs(
                3,
                &[
                    (Dimension::Harness, &["claude-code"]),
                    (Dimension::Topic, &["tests"]),
                ],
            ),
        ];
        let a = Segmenter::replay(SegmentPolicy::default(), 42, &history).unwrap();
        let b = Segmenter::replay(SegmentPolicy::default(), 42, &history).unwrap();
        assert_eq!(a.segments(), b.segments());
        assert_eq!(a.boundaries(), b.boundaries());
        let other = Segmenter::replay(SegmentPolicy::default(), 43, &history).unwrap();
        assert_ne!(a.segments()[0].id, other.segments()[0].id);
        // Same segments, only the ids differ.
        assert_eq!(a.segments().len(), other.segments().len());
        // The hash itself is pinned in `segment_id_is_pinned`.
        assert!(a.segments()[0].id.as_str().starts_with("seg-"));
        assert_eq!(a.segments()[0].id.as_str().len(), "seg-".len() + 16);
    }

    #[test]
    fn segment_id_is_pinned() {
        // Computed once from the definition (FNV-1a 64 over seed, axis name, first turn,
        // each followed by 0xff). If this changes, stored segment ids stop matching.
        assert_eq!(
            SegmentId::derive(42, Dimension::Harness, 3).as_str(),
            "seg-30a42f3e03fc7385"
        );
        assert_ne!(
            SegmentId::derive(0, Dimension::Harness, 0),
            SegmentId::derive(0, Dimension::Topic, 0)
        );
        assert_ne!(
            SegmentId::derive(0, Dimension::Harness, 0),
            SegmentId::derive(0, Dimension::Harness, 1)
        );
    }

    #[test]
    fn a_fingerprint_stops_growing_at_the_cap() {
        // Every turn shares one label with the segment (so it is never "totally disjoint")
        // and brings ten new ones: only a segment that keeps absorbing can grow past the cap.
        let policy = SegmentPolicy::default()
            .with(Dimension::Understanding, Threshold::new(1_000, 1).unwrap());
        let labels: Vec<String> = (0..200).map(|i| format!("l{i:03}")).collect();
        let mut segmenter = Segmenter::new(policy, 7);
        for (turn, chunk) in labels.chunks(10).enumerate() {
            let mut observation = Observation {
                turn: u32::try_from(turn).unwrap(),
                ..Observation::default()
            };
            let mut print = vec!["core".to_owned()];
            print.extend(chunk.iter().cloned());
            observation
                .prints
                .insert(Dimension::Understanding, Fingerprint::new(print));
            segmenter.observe(&observation).unwrap();
        }
        let segments = segmenter.segments();
        assert_eq!(segments.len(), 1, "one segment absorbed all twenty turns");
        assert_eq!(segments[0].observations, 20);
        assert_eq!(
            segments[0].fingerprint.labels().count(),
            MAX_FINGERPRINT_LABELS
        );
    }

    #[test]
    fn thresholds_are_validated() {
        assert_eq!(
            Threshold::new(0, 1),
            Err(SegmentError::DistanceOutOfRange(0))
        );
        assert_eq!(
            Threshold::new(1_001, 1),
            Err(SegmentError::DistanceOutOfRange(1_001))
        );
        assert_eq!(Threshold::new(500, 0), Err(SegmentError::NoConfirmation));
        assert!(Threshold::new(500, 1).is_ok());
    }

    #[test]
    fn a_custom_policy_changes_where_a_boundary_falls() {
        let strict =
            SegmentPolicy::default().with(Dimension::Topic, Threshold::new(700, 3).unwrap());
        let history = [
            topic(0, &["relay"]),
            topic(1, &["weather"]),
            topic(2, &["weather"]),
        ];
        // Two deviations do not confirm a three-confirmation threshold...
        let segmenter = Segmenter::replay(strict, 7, &history).unwrap();
        assert!(segmenter.boundaries().is_empty());
        // ...they do confirm the default one.
        assert_eq!(run(&history).boundaries().len(), 1);
    }

    #[test]
    fn boundaries_serialize_with_snake_case_axes() {
        let segmenter = run(&[harness(0, "a"), harness(1, "b")]);
        let json = serde_json::to_value(&segmenter.boundaries()[0]).unwrap();
        assert_eq!(json["dimension"], "harness");
        assert_eq!(json["at_turn"], 1);
        let back: Boundary = serde_json::from_value(json).unwrap();
        assert_eq!(&back, &segmenter.boundaries()[0]);
    }
}
