//! Per-instance circuit breaker: turns a stream of call outcomes into a
//! live "is this instance healthy right now" signal.
//!
//! This module is intentionally standalone: it has no dependency on
//! [`super::catalog`], [`super::resolver`], [`super::cognitive`] or
//! [`super::errors`]. It knows nothing about providers, HTTP status codes or
//! `nexus_claude::agent::ProviderError` — the caller decides what counts as
//! a [`Outcome::Failure`] (see `ProviderError::retryable()` in nexus, or
//! `OpenFailure.retryable` in [`super::errors`]) and feeds it in.
//!
//! Wiring this breaker's output into `ModelFacts.healthy`
//! (`cognitive::candidates`) or into `StoreCatalog::with_unhealthy`
//! (`catalog`) is a deliberate follow-up, not done here: at the time this
//! module was written, both `catalog.rs` and `cognitive/candidates.rs` had
//! several worktrees actively modifying them (routing-decision work,
//! review-fix passes); adding this new, self-contained file avoids adding a
//! merge surface to that in-flight work. See decision on the plan
//! "Résilience & routage multi-provider inspirés OmniRoute".
//!
//! # Design
//!
//! Three states, like a standard (Hystrix-style) breaker:
//!
//! - **Closed**: calls are attempted; consecutive failures are counted.
//!   `failure_threshold` consecutive failures opens the circuit. Any
//!   success resets the counter to zero.
//! - **Open**: calls are refused ([`CircuitBreaker::can_attempt`] returns
//!   `false`) until `cooldown` has elapsed since opening. Once elapsed, the
//!   *next* [`CircuitBreaker::can_attempt`] call transitions to `HalfOpen`
//!   and returns `true` exactly once — this is the single probe attempt.
//! - **HalfOpen**: exactly one call is in flight (the probe). A
//!   [`Outcome::Success`] closes the circuit (back to `Closed` with the
//!   counter at zero). A [`Outcome::Failure`] reopens it with a fresh
//!   cooldown window.
//!
//! ## The gap this fixes (found auditing OmniRoute's `adaptiveCircuit.ts`)
//!
//! OmniRoute's breaker closes on *any* success it is told about, even one
//! reported while the circuit is `Open` and no probe was ever granted (a
//! call started before the circuit opened, or a caller that never checks
//! `can_attempt`, reporting its outcome late). That lets a circuit close
//! without ever having passed through a real probe.
//!
//! This implementation refuses that: [`CircuitBreaker::record`] only acts
//! on a `Success` when the state is `HalfOpen` (reported in response to the
//! one attempt `can_attempt` granted) or `Closed` (where it is a no-op
//! beyond resetting the counter, since the circuit was never open). A
//! `Success` reported while `Open` is dropped — see
//! `a_late_success_reported_while_open_does_not_close_the_circuit` below.

use std::time::{Duration, Instant};

/// The three states of the breaker, exposed for observability/logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Calls are attempted; `failures` consecutive failures have been seen.
    Closed,
    /// Calls are refused until the cooldown elapses.
    Open,
    /// One probe call is in flight; its outcome decides the next state.
    HalfOpen,
}

/// The outcome of one call, as judged by the caller.
///
/// This breaker does not interpret HTTP status codes or provider error
/// types itself — see the module doc. The caller maps whatever error
/// taxonomy it has (e.g. `ProviderError::retryable()` in nexus) to one of
/// these two variants before calling [`CircuitBreaker::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The call succeeded (or failed in a way that should not count against
    /// the instance, e.g. a client error unrelated to provider health).
    Success,
    /// The call failed in a way that should count against the instance's
    /// health (timeout, 5xx, connection refused, rate limited, ...).
    Failure,
}

/// A circuit breaker for one provider instance.
///
/// Not thread-safe by itself (no internal lock): wrap it in a `Mutex` (or
/// one per instance behind a `DashMap`) at the call site, matching how
/// `catalog::StoreCatalog` already keys per-instance state.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    cooldown: Duration,
    state: CircuitState,
    consecutive_failures: u32,
    /// Set when `state == Open`: the instant at or after which the next
    /// `can_attempt` call is allowed to grant a probe.
    next_probe_at: Option<Instant>,
}

impl CircuitBreaker {
    /// Creates a new breaker, starting `Closed` with a zero failure count.
    ///
    /// `failure_threshold` must be at least 1 (a threshold of 0 would open
    /// on the very first recorded failure with no chance to accumulate,
    /// which is a degenerate but not unreasonable choice — it is not
    /// rejected here, just documented).
    #[must_use]
    pub fn new(failure_threshold: u32, cooldown: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown,
            state: CircuitState::Closed,
            consecutive_failures: 0,
            next_probe_at: None,
        }
    }

    /// The current state, for logging/metrics. Does not mutate.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Whether a call may be attempted right now.
    ///
    /// This is the one method that can transition `Open` -> `HalfOpen`: it
    /// is mutating (`&mut self`) specifically so that granting the single
    /// probe attempt is an atomic, observable step from the breaker's own
    /// point of view — not two racy reads (“is cooldown over” then “mark
    /// probe in flight”) split across the call site.
    pub fn can_attempt(&mut self, now: Instant) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => false, // probe already in flight
            CircuitState::Open => {
                let due = self.next_probe_at.is_some_and(|at| now >= at);
                if due {
                    self.state = CircuitState::HalfOpen;
                    self.next_probe_at = None;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Records the outcome of a call made because `can_attempt` returned
    /// `true`. Calling this after `can_attempt` returned `false` is a
    /// caller bug (the call should not have been made); this method
    /// tolerates it by being a no-op on points that would otherwise close
    /// an unprobed `Open` circuit — see the module doc "gap" section.
    pub fn record(&mut self, outcome: Outcome, now: Instant) {
        match (self.state, outcome) {
            (CircuitState::Closed, Outcome::Success) => {
                self.consecutive_failures = 0;
            }
            (CircuitState::Closed, Outcome::Failure) => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.failure_threshold.max(1) {
                    self.open(now);
                }
            }
            (CircuitState::HalfOpen, Outcome::Success) => {
                self.state = CircuitState::Closed;
                self.consecutive_failures = 0;
            }
            (CircuitState::HalfOpen, Outcome::Failure) => {
                self.open(now);
            }
            (CircuitState::Open, _) => {
                // Deliberately ignored: no outcome reported while the
                // circuit is Open (i.e. not the granted probe) may change
                // its state. A late Success must not close it (the
                // OmniRoute gap); a late Failure is already reflected in
                // the circuit being open and must not reset the cooldown
                // clock, which would let a slow caller keep it open
                // indefinitely.
            }
        }
    }

    fn open(&mut self, now: Instant) {
        self.state = CircuitState::Open;
        self.next_probe_at = Some(now + self.cooldown);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn breaker() -> CircuitBreaker {
        CircuitBreaker::new(3, Duration::from_secs(60))
    }

    #[test]
    fn starts_closed_with_no_failures() {
        let b = breaker();
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn fewer_than_threshold_failures_stay_closed() {
        let mut b = breaker();
        let t0 = Instant::now();
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn threshold_consecutive_failures_opens_the_circuit() {
        let mut b = breaker();
        let t0 = Instant::now();
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        assert_eq!(b.state(), CircuitState::Open);
    }

    #[test]
    fn a_success_resets_the_consecutive_failure_count() {
        let mut b = breaker();
        let t0 = Instant::now();
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Success, t0);
        // Two more failures should not be enough to open (count was reset).
        b.record(Outcome::Failure, t0);
        b.record(Outcome::Failure, t0);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn can_attempt_refuses_calls_while_open_and_within_cooldown() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        assert_eq!(b.state(), CircuitState::Open);
        assert!(!b.can_attempt(t0 + Duration::from_secs(30)));
        // Refusing a probe must not itself change the state.
        assert_eq!(b.state(), CircuitState::Open);
    }

    #[test]
    fn can_attempt_grants_exactly_one_probe_after_cooldown() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let after_cooldown = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(after_cooldown));
        assert_eq!(b.state(), CircuitState::HalfOpen);
        // A second concurrent caller must not get a second probe.
        assert!(!b.can_attempt(after_cooldown));
    }

    #[test]
    fn a_successful_probe_closes_the_circuit() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let probe_at = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(probe_at));
        b.record(Outcome::Success, probe_at);
        assert_eq!(b.state(), CircuitState::Closed);
    }

    #[test]
    fn a_failed_probe_reopens_with_a_fresh_cooldown() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        let probe_at = t0 + Duration::from_secs(61);
        assert!(b.can_attempt(probe_at));
        b.record(Outcome::Failure, probe_at);
        assert_eq!(b.state(), CircuitState::Open);
        // The fresh cooldown is measured from the failed probe, not from
        // the original opening: right after the failed probe, still closed
        // to new attempts.
        assert!(!b.can_attempt(probe_at + Duration::from_secs(1)));
        assert!(b.can_attempt(probe_at + Duration::from_secs(61)));
    }

    /// The OmniRoute gap this breaker deliberately avoids: see module doc.
    #[test]
    fn a_late_success_reported_while_open_does_not_close_the_circuit() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        assert_eq!(b.state(), CircuitState::Open);
        // No call to can_attempt here: this success is reported by some
        // straggler call that was in flight before the circuit opened, or
        // by a caller that skipped the can_attempt check entirely.
        b.record(Outcome::Success, t0 + Duration::from_secs(1));
        assert_eq!(
            b.state(),
            CircuitState::Open,
            "a success not granted via can_attempt must not close an Open circuit"
        );
    }

    /// Symmetric case: a late failure reported while Open must not reset
    /// the cooldown clock (which would let a slow, already-accounted-for
    /// failure keep extending the outage).
    #[test]
    fn a_late_failure_reported_while_open_does_not_extend_the_cooldown() {
        let mut b = breaker();
        let t0 = Instant::now();
        for _ in 0..3 {
            b.record(Outcome::Failure, t0);
        }
        // A straggler failure arrives mid-cooldown.
        b.record(Outcome::Failure, t0 + Duration::from_secs(30));
        // The original cooldown (relative to t0) must still be the one in
        // effect: it elapses at t0 + 60s, not later.
        assert!(b.can_attempt(t0 + Duration::from_secs(61)));
    }

    #[test]
    fn a_threshold_of_zero_is_treated_as_one() {
        let mut b = CircuitBreaker::new(0, Duration::from_secs(60));
        let t0 = Instant::now();
        b.record(Outcome::Failure, t0);
        assert_eq!(b.state(), CircuitState::Open);
    }
}
