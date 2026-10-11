//! Sub-agents of a native session (P18): the `Agent` tool of the nexus harness, run as a
//! child session of project-orchestrator (decision A18: a provider without sub-agents of its
//! own gets the host's child session).
//!
//! The native harness offers its `Agent` tool when the session's hooks say
//! `runs_subagents()`, and hands every call to `SessionHooks::run_subagent`. [`SubagentHooks`]
//! is that host: it wraps the session's other hooks (forwarded untouched) and, on a call,
//! [`ChatManager::run_subagent`] opens the child exactly as `chat(send_message)` opens one
//! for an agent caller (A17, `envelope.rs`):
//!
//! * the envelope of the parent: depth 1 (a session that was itself spawned cannot spawn),
//!   at most [`MAX_LIVE_CHILDREN`](super::envelope::MAX_LIVE_CHILDREN) children alive, a
//!   permission mode never wider than the parent's, the parent's project and directories;
//! * `spawned_by = conversation { parent_session_id, tool_use_id }`: the child is in the
//!   parent's session tree, and its token is restricted (H6: its parent is a third party);
//! * the parent's provider and model, the parent's claims — and, when the parent has none
//!   (reopened by the restart recovery, which keeps no claims), the parent's stored OWNER:
//!   the child belongs to the same person (P18b round 3, fix A). No claims mints no token,
//!   so the child's MCP access stays as restricted as before;
//! * the parent's workspace (inherited by the envelope), with its directories clamped to
//!   what the parent could already see (P18b round 2, fix 1);
//! * the prompt as an inert message (no reference block may ride on a model's text).
//!
//! The call waits for the child's first `result` and answers its text; the child is then
//! closed (it stays resumable and visible). A dropped call (the parent's turn interrupted or
//! the call cancelled) closes the child too — at ANY point of the call, the opening
//! included: [`ChatManager::run_subagent`] opens, awaits and closes the child in a task of
//! its own and only AWAITS that task, so a drop cannot cut the opening halfway. That matters
//! because `create_session` runs as long as the provider's open timeout and has already sent
//! the child's first message when it returns: a future dropped inside it used to leave a
//! child running that nobody knew the id of. The caller's drop closes a one-shot channel the
//! task watches; the task then closes the child as soon as the opening gives its id.
//!
//! The child's answer is read from its live event stream, and from its stored thread
//! whenever the stream cannot be trusted: before the first wait (a turn that ended before
//! the subscription), once the stream is closed, and — from the first `Lagged` on — every
//! second until the thread shows the `result` or the close, the live stream still watched
//! alongside (P18b round 3, fix C: the `result` may have been among the dropped events and
//! reach the thread arbitrarily later; a fixed number of re-reads left the parent waiting
//! until its turn timeout). The tick spaces the re-reads: a child that lags again and again
//! costs one tail read per second. What is kept from the stream is bounded
//! ([`keep_for_answer`]): a chatty child cannot grow it.
//!
//! The close of the child is backed by a guard armed from the moment the child exists
//! until its explicit `close_session` has RETURNED (round 3, fix E: a panic in the close
//! itself is covered; a second close is a harmless "not found or inactive"); the guard
//! owns the parent's slot, so a panic gives it back only after the detached close.
//!
//! Depth is held three times: the backend gives [`SubagentHooks`] only to a native session
//! whose record has no `spawned_by` (`ChatManager::build_agent_spec_with_access`); a spawned
//! native session is opened with `SessionSpec::lineage` at depth 1, so the harness itself
//! refuses it the tool; and the envelope refuses a parent that has a parent. A child keeps its
//! `spawned_by` when it moves to another provider (relay).
//!
//! The cap of live children holds under parallel calls (the harness runs the calls of one step
//! concurrently): a slot of the parent is taken BEFORE the envelope counts its children, in
//! [`ChatManager::run_subagent`], and held by [`ChatManager::open_wait_close`]'s OWN task —
//! not by `run_subagent`'s frame — until the sub-agent actually ends, right alongside its
//! `close_session` ([`MAX_LIVE_CHILDREN`](super::envelope::MAX_LIVE_CHILDREN) slots per
//! parent; P18b round 2, fix 2: a dropped `run_subagent` future used to free the slot as soon
//! as it was dropped, which could be well before the task it spawned had even learned the
//! child's id, let alone closed it). The ceiling is the parent's CURRENT mode: a native
//! session's mode changes are kept live and persisted (`set_session_permission_mode`).
//!
//! What a child may not do: ask the user (`AskUserQuestion` is denied to a spawned native
//! session; a question seen anyway fails the call at once); wait for a permission (round 3,
//! fix B: a request still unanswered [`UNANSWERED_ASK_GRACE`] after it was shown fails the
//! call, and the child is closed like on every other end — nobody watches a sub-agent's
//! session, so the request would otherwise hold the parent until its 30-minute turn
//! timeout); return more than [`MAX_ANSWER_CHARS`] characters (the rest is cut, and said).
//! Failing fast was chosen over opening the child with a deny-instead-of-ask policy: the
//! harness has no such mode (`PolicyMode` is plan / ask / auto-edits / trust), and a fail
//! needs no new policy plumbing and no new prompt — the parent's model reads why and does the
//! step itself. The grace exists for a request the backend answers by itself (a lasting
//! rule of the child's owner covers it: its `permission_decision` follows within moments).
//! The call otherwise waits as long as the child works, bounded by the parent's turn
//! timeout. Budget and cost of the children are not counted in the parent's (task B36).

use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use nexus_claude::agent::{
    CompactionInfo, HookVerdict, SessionHooks, SubagentRequest, ToolCallInfo, ToolResultInfo,
    TurnContext, TurnDirective,
};
use uuid::Uuid;

use super::manager::ChatManager;
use super::types::{ChatEvent, ChatRequest, SpawnedBy};

/// The hooks of a native session that may run sub-agents: its other hooks, plus the host of
/// the `Agent` tool.
pub(crate) struct SubagentHooks {
    inner: Option<Arc<dyn SessionHooks>>,
    manager: Weak<ChatManager>,
    parent_session_id: String,
    user_claims: Option<crate::auth::jwt::Claims>,
}

impl SubagentHooks {
    pub(crate) fn new(
        inner: Option<Arc<dyn SessionHooks>>,
        manager: Weak<ChatManager>,
        parent_session_id: impl Into<String>,
        user_claims: Option<crate::auth::jwt::Claims>,
    ) -> Self {
        Self {
            inner,
            manager,
            parent_session_id: parent_session_id.into(),
            user_claims,
        }
    }
}

#[async_trait]
impl SessionHooks for SubagentHooks {
    async fn before_tool(&self, call: &ToolCallInfo) -> HookVerdict {
        match &self.inner {
            Some(inner) => inner.before_tool(call).await,
            None => HookVerdict::Continue,
        }
    }

    async fn after_tool(&self, result: &ToolResultInfo) -> Option<String> {
        match &self.inner {
            Some(inner) => inner.after_tool(result).await,
            None => None,
        }
    }

    async fn before_compaction(&self, info: &CompactionInfo) -> Option<String> {
        match &self.inner {
            Some(inner) => inner.before_compaction(info).await,
            None => None,
        }
    }

    async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
        match &self.inner {
            Some(inner) => inner.before_turn(ctx).await,
            None => TurnDirective::default(),
        }
    }

    fn runs_subagents(&self) -> bool {
        true
    }

    async fn run_subagent(&self, request: &SubagentRequest) -> Result<String, String> {
        let Some(manager) = self.manager.upgrade() else {
            return Err("the server is shutting down: no sub-agent can start".to_owned());
        };
        manager
            .run_subagent(&self.parent_session_id, self.user_claims.clone(), request)
            .await
    }
}

/// One running sub-agent of a parent (module documentation): taken before the envelope
/// counts the parent's live children, given back when the sub-agent ends or its call is dropped.
pub(crate) struct SubagentSlot {
    manager: Weak<ChatManager>,
    parent: String,
}

impl SubagentSlot {
    /// A slot of `parent`, or `None` when its [`MAX_LIVE_CHILDREN`] slots are taken.
    ///
    /// [`MAX_LIVE_CHILDREN`]: super::envelope::MAX_LIVE_CHILDREN
    pub(crate) fn take(manager: &Arc<ChatManager>, parent: &str) -> Option<Self> {
        let mut slots = manager
            .subagent_slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let taken = slots.entry(parent.to_string()).or_insert(0);
        if *taken >= super::envelope::MAX_LIVE_CHILDREN {
            return None;
        }
        *taken += 1;
        Some(Self {
            manager: Arc::downgrade(manager),
            parent: parent.to_string(),
        })
    }
}

impl Drop for SubagentSlot {
    fn drop(&mut self) {
        let Some(manager) = self.manager.upgrade() else {
            return;
        };
        let mut slots = manager
            .subagent_slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(taken) = slots.get_mut(&self.parent) {
            *taken = taken.saturating_sub(1);
            if *taken == 0 {
                slots.remove(&self.parent);
            }
        }
    }
}

/// A backstop for exactly one window: between a resource existing and the END of the
/// explicit code that releases it. [`Self::new`] arms it with what to run if it is ever
/// dropped armed; [`Self::disarm`] is called right AFTER that explicit release returned, so
/// a panic inside the release itself is still covered (the release must then be safe to
/// run twice). Used by [`ChatManager::open_wait_close`] (P18b round 2, fix 3; round 3,
/// fix E) for the window between `create_session` succeeding and the child's explicit
/// `close_session` having returned: a panic there (none today) would otherwise leave the
/// child open with nothing left to close it until the next restart's recovery scan.
struct ClosePanicGuard {
    armed: bool,
    on_drop: Option<Box<dyn FnOnce() + Send>>,
}

impl ClosePanicGuard {
    fn new(on_drop: impl FnOnce() + Send + 'static) -> Self {
        Self {
            armed: true,
            on_drop: Some(Box::new(on_drop)),
        }
    }

    /// The ordinary path took over: this guard's backstop must not run any more.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ClosePanicGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Some(f) = self.on_drop.take() {
            f();
        }
    }
}

/// Longest answer of a sub-agent given back to its parent, in characters: a sub-agent that
/// copies a whole file must not fill the parent's context.
pub(crate) const MAX_ANSWER_CHARS: usize = 64 * 1024;

/// `text` cut to [`MAX_ANSWER_CHARS`], saying how much was left out.
fn bounded(text: String) -> String {
    let total = text.chars().count();
    if total <= MAX_ANSWER_CHARS {
        return text;
    }
    let kept: String = text.chars().take(MAX_ANSWER_CHARS).collect();
    format!(
        "{kept}\n[… {} characters of the sub-agent's answer left out]",
        total - MAX_ANSWER_CHARS
    )
}

/// A question the child asked: the call fails on it.
fn is_question(event: &ChatEvent) -> bool {
    matches!(event, ChatEvent::AskUserQuestion { .. })
}

/// Something the child said in its own voice (not inside one of its own tool calls).
fn is_said(event: &ChatEvent) -> bool {
    matches!(
        event,
        ChatEvent::AssistantText {
            parent_tool_use_id: None,
            ..
        }
    )
}

/// The end of the child's turn.
fn is_result(event: &ChatEvent) -> bool {
    matches!(event, ChatEvent::Result { .. })
}

/// Adds `event` to what [`answer_of`] will be given, keeping only what it actually reads:
/// the first question, the LAST thing said, and the first `result` — at most three events,
/// whatever the child emits. Without this bound a chatty child (thousands of deltas and
/// tool results) grew the parent's vector for the whole call.
fn keep_for_answer(seen: &mut Vec<ChatEvent>, event: ChatEvent) {
    if is_said(&event) {
        // `answer_of` falls back to the LAST thing said: replace it in place.
        match seen.iter_mut().find(|e| is_said(e)) {
            Some(held) => *held = event,
            None => seen.push(event),
        }
    } else if (is_question(&event) && !seen.iter().any(is_question))
        || (is_result(&event) && !seen.iter().any(is_result))
    {
        seen.push(event);
    }
}

/// The answer of a sub-agent from its `result`: the result text, else what it said. A
/// sub-agent that asked the user a question fails the call: nobody sees a child's question.
fn answer_of(events: &[ChatEvent]) -> Option<Result<String, String>> {
    if events.iter().any(is_question) {
        return Some(Err(
            "the sub-agent asked the user a question; a sub-agent cannot ask: give it what it needs in the prompt".to_owned(),
        ));
    }
    let (is_error, result_text, subtype) = events.iter().find_map(|e| match e {
        ChatEvent::Result {
            is_error,
            result_text,
            subtype,
            ..
        } => Some((*is_error, result_text.clone(), subtype.clone())),
        _ => None,
    })?;
    let said: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            ChatEvent::AssistantText {
                content,
                parent_tool_use_id: None,
            } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    let text = result_text
        .filter(|t| !t.trim().is_empty())
        .or_else(|| said.last().map(|s| s.to_string()))
        .unwrap_or_default();
    Some(if is_error {
        Err(format!(
            "the sub-agent failed ({subtype}){}",
            if text.is_empty() {
                String::new()
            } else {
                format!(": {text}")
            }
        ))
    } else {
        Ok(bounded(text))
    })
}

impl ChatManager {
    /// Lets this manager hand itself to what outlives a call: the sub-agents of native
    /// sessions (P18) open their child sessions through it. Called once the manager is in its
    /// `Arc`; without it, native sessions are given no `Agent` tool.
    pub fn enable_subagents(self: &Arc<Self>) {
        let _ = self.self_ref.set(Arc::downgrade(self));
    }

    /// The manager itself, once [`Self::enable_subagents`] was called.
    pub(crate) fn weak_self(&self) -> Option<Weak<ChatManager>> {
        self.self_ref.get().cloned()
    }

    /// Runs one sub-agent of `parent_session_id` as its child session and returns what it
    /// answered (module documentation). `Err` is what the model reads when it could not run.
    pub(crate) async fn run_subagent(
        self: Arc<Self>,
        parent_session_id: &str,
        user_claims: Option<crate::auth::jwt::Claims>,
        request: &SubagentRequest,
    ) -> Result<String, String> {
        use super::envelope;
        let refused = |why: String| format!("the sub-agent could not start: {why}");
        let parent_uuid = Uuid::parse_str(parent_session_id).map_err(|e| refused(e.to_string()))?;
        let parent = self
            .graph
            .get_chat_session(parent_uuid)
            .await
            .map_err(|e| refused(e.to_string()))?
            .ok_or_else(|| refused("the parent session is unknown".to_owned()))?;
        // Taken before the envelope counts the parent's children (A17); moved into
        // `open_wait_close`'s own task below and held there for the child's whole real
        // lifetime, NOT in this frame — a drop of this future (the parent's turn
        // interrupted) must not free the slot before the spawned task has actually
        // closed the child (P18b round 2, fix 2).
        let slot = SubagentSlot::take(&self, parent_session_id).ok_or_else(|| {
            refused(format!(
                "envelope_too_many_children (at most {} sub-agents of one session run at once)",
                envelope::MAX_LIVE_CHILDREN
            ))
        })?;
        let env =
            envelope::resolve_envelope(self.graph.as_ref(), self.as_ref(), parent_session_id, None)
                .await
                .map_err(|e| refused(e.code().to_owned()))?;
        let mut child = ChatRequest {
            message: crate::refs::compose::inert(&request.prompt),
            cwd: parent.cwd.clone(),
            model: Some(parent.model.clone()),
            provider: parent.provider_id.clone(),
            add_dirs: parent.add_dirs.clone(),
            user_claims,
            task_context: request.description.clone(),
            // Never wider than the parent (a read-only parent has no `Agent`; defence in depth).
            access: Some(parent.access),
            ..Default::default()
        };
        let default_mode = self.default_permission_mode().await;
        env.apply(&mut child, &default_mode)
            .map_err(|e| refused(e.code().to_owned()))?;
        child.spawned_by = Some(
            SpawnedBy::Conversation {
                parent_session_id: env.parent_session_id,
                tool_use_id: Some(request.tool_call_id.clone()),
            }
            .to_json_string(),
        );
        // The opening must not be droppable (module documentation): it is done in a task of
        // its own, and this future only awaits that task. `alive` is dropped with this
        // future — whether it ends normally (the task is over by then) or is cut — and that
        // is how the task learns nobody awaits the child any more.
        let (alive, cancelled) = tokio::sync::oneshot::channel::<std::convert::Infallible>();
        let opening = {
            let manager = Arc::clone(&self);
            let parent = parent_session_id.to_string();
            let tool_use_id = request.tool_call_id.clone();
            tokio::spawn(async move {
                manager
                    .open_wait_close(child, cancelled, &parent, &tool_use_id, slot)
                    .await
            })
        };
        let _alive = alive;
        opening
            .await
            .unwrap_or_else(|e| Err(refused(format!("the sub-agent's opening ended: {e}"))))
    }

    /// Opens the child of one sub-agent call, waits for its answer and closes it. Runs in a
    /// task of its own ([`Self::run_subagent`]) so no drop can cut the opening halfway;
    /// `cancelled` resolves when nobody awaits the call any more, and the child is then
    /// closed as soon as the opening has given its id. `slot` is the parent's reservation
    /// (module documentation): held by this task, then by its close guard, so it is given
    /// back only once the child this task owns is actually closed — on the ordinary path,
    /// on a dropped caller (P18b round 2, fix 2) and on a panic (round 3, fix E).
    async fn open_wait_close(
        self: Arc<Self>,
        child: ChatRequest,
        cancelled: tokio::sync::oneshot::Receiver<std::convert::Infallible>,
        parent_session_id: &str,
        tool_use_id: &str,
        slot: SubagentSlot,
    ) -> Result<String, String> {
        let opened = self
            .create_session(&child)
            .await
            .map_err(|e| format!("the sub-agent could not start: {e:#}"))?;
        let child_id = opened.session_id;
        tracing::info!(parent = %parent_session_id, child = %child_id, tool_use_id = %tool_use_id, "sub-agent started");
        // The child exists from here: whatever ends this task, it is closed first (it stays
        // resumable and visible). Nothing between here and the end of the explicit close
        // below panics today, but nothing rules one out after this module grows — `guard`
        // is the backstop for exactly that window, the close call itself INCLUDED: it is
        // disarmed only once `close_session` has returned (P18b round 3, fix E; a second
        // close of the same child is a harmless "not found or inactive"). The guard owns the
        // slot: a panic gives it back only after the detached close, never before.
        let manager = Arc::clone(&self);
        let backstop_id = child_id.clone();
        let mut guard = ClosePanicGuard::new(move || {
            tokio::spawn(async move {
                if let Err(e) = manager.close_session(&backstop_id).await {
                    tracing::debug!(child = %backstop_id, error = %e, "sub-agent close");
                }
                drop(slot);
            });
        });
        let answer = tokio::select! {
            biased;
            _ = cancelled => Err("the sub-agent's call was cancelled".to_owned()),
            answer = self.wait_for_answer(&child_id) => answer,
        };
        #[cfg(test)]
        panic_injection::panic_if_injected(parent_session_id);
        if let Err(e) = self.close_session(&child_id).await {
            tracing::debug!(child = %child_id, error = %e, "sub-agent close");
        }
        // The child is closed: the backstop is not needed any more, and dropping the guard
        // gives the slot back — after the close, as on every other path.
        guard.disarm();
        drop(guard);
        answer
    }

    /// The first `result` of the session `sid`, live or already persisted.
    async fn wait_for_answer(&self, sid: &str) -> Result<String, String> {
        let rx = self
            .subscribe(sid)
            .await
            .map_err(|e| format!("the sub-agent cannot be followed: {e:#}"))?;
        self.answer_from(sid, rx).await
    }

    /// [`Self::wait_for_answer`] on a subscription already taken — the seam the test of a
    /// lagged subscription drives.
    pub(super) async fn answer_from(
        &self,
        sid: &str,
        mut rx: tokio::sync::broadcast::Receiver<ChatEvent>,
    ) -> Result<String, String> {
        let mut watch = Watch::default();
        // A turn that ended before the subscription is in the store.
        if let Some(end) = self.read_back(sid, &mut watch).await {
            return end;
        }
        // Once a `Lagged` was seen, the stream alone cannot be trusted any more: the thread
        // is re-read on every tick until it shows the end (P18b round 3, fix C) — no fixed
        // number of attempts, since the child's events may land on its thread long after
        // the drop (a verbose child is exactly the one that lags). The tick spaces those
        // reads one second apart however often the stream lags, so a chatty child does not
        // make the parent hammer the store; it also wakes the loop while a permission
        // request waits (fix B).
        let mut tick = tokio::time::interval(READ_BACK_EVERY);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut lagged = false;
        loop {
            if let Some(end) = watch.unanswered_ask(tokio::time::Instant::now()) {
                return end;
            }
            tokio::select! {
                received = rx.recv() => match received {
                    Ok(event) => {
                        if let Some(end) = watch.observe(event, tokio::time::Instant::now()) {
                            return end;
                        }
                    }
                    // Events were dropped, and the child's `result` may have been among
                    // them: waiting on the stream alone could last until the parent's
                    // 30-minute turn timeout. The thread is the truth (P18b, fix 4).
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(dropped)) => {
                        if !lagged {
                            tracing::warn!(child = %sid, dropped, "sub-agent events dropped: its thread is read back until it ends");
                        }
                        lagged = true;
                    }
                    // The stream is over: what it did not carry, the thread may still hold.
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        return self.read_back(sid, &mut watch).await.unwrap_or_else(|| {
                            Err("the sub-agent's session ended before it answered".into())
                        });
                    }
                },
                _ = tick.tick(), if lagged || watch.is_asking() => {
                    if lagged {
                        if let Some(end) = self.read_back(sid, &mut watch).await {
                            return end;
                        }
                    }
                }
            }
        }
    }

    /// The end of `sid` as its STORED thread tells it — its answer, or its close — and
    /// what it still asks for there, into `watch` (a decision the stream dropped is seen
    /// here). `None` while the thread shows no end.
    async fn read_back(&self, sid: &str, watch: &mut Watch) -> Option<Result<String, String>> {
        let persisted = self.persisted_events(sid).await;
        watch.reconcile(&persisted, tokio::time::Instant::now());
        if let Some(answer) = answer_of(&persisted) {
            return Some(answer);
        }
        persisted
            .iter()
            .any(|e| matches!(e, ChatEvent::SessionClosed { .. }))
            .then(|| Err(CLOSED_BEFORE_ANSWER.to_owned()))
    }

    /// The answer of `sid` as its STORED thread tells it, when its `result` is already
    /// there.
    #[cfg(test)]
    async fn persisted_answer(&self, sid: &str) -> Option<Result<String, String>> {
        answer_of(&self.persisted_events(sid).await)
    }

    /// The most recent events of `sid`'s STORED thread. Reads the TAIL of the thread, not
    /// the head (P18b round 2, fix 5): a long child's `result` can land after more than
    /// [`PERSISTED_READ_WINDOW`] events, and `get_chat_events` only reads forward from
    /// `after_seq` — starting at 0 would never reach it.
    async fn persisted_events(&self, sid: &str) -> Vec<ChatEvent> {
        let Ok(uuid) = Uuid::parse_str(sid) else {
            return Vec::new();
        };
        let latest_seq = self
            .graph
            .get_latest_chat_event_seq(uuid)
            .await
            .unwrap_or(0);
        let after_seq = (latest_seq - PERSISTED_READ_WINDOW).max(0);
        self.graph
            .get_chat_events(uuid, after_seq, PERSISTED_READ_WINDOW)
            .await
            .unwrap_or_default()
            .iter()
            .filter_map(|r| serde_json::from_str(&r.data).ok())
            .collect()
    }
}

/// What the parent's wait on one child holds: what [`answer_of`] will read, and the
/// permission requests of the child still waiting for an answer (P18b round 3, fix B).
#[derive(Default)]
struct Watch {
    seen: Vec<ChatEvent>,
    /// Request id → (tool, when the parent first saw it waiting).
    asks: std::collections::HashMap<String, (String, tokio::time::Instant)>,
}

impl Watch {
    /// Takes one live event; the end of the call when this event ends it.
    fn observe(
        &mut self,
        event: ChatEvent,
        now: tokio::time::Instant,
    ) -> Option<Result<String, String>> {
        match &event {
            ChatEvent::PermissionRequest { id, tool, .. } => {
                self.asks.entry(id.clone()).or_insert((tool.clone(), now));
            }
            ChatEvent::PermissionDecision { id, .. } => {
                self.asks.remove(id);
            }
            _ => {}
        }
        let done = is_result(&event);
        let asked = is_question(&event);
        let closed = matches!(event, ChatEvent::SessionClosed { .. });
        keep_for_answer(&mut self.seen, event);
        if done || asked {
            // A question fails the call at once: nobody would ever answer it.
            return Some(answer_of(&self.seen).unwrap_or_else(|| Ok(String::new())));
        }
        if closed {
            // A child can also close without this event reaching the parent (dropped, a
            // crash, a restart): the read-back after a `Lagged` sees it on the thread.
            return Some(Err(CLOSED_BEFORE_ANSWER.into()));
        }
        None
    }

    /// Brings the waiting requests in line with the stored thread: what it shows decided
    /// is not waiting any more, what it shows asked and undecided is.
    fn reconcile(&mut self, persisted: &[ChatEvent], now: tokio::time::Instant) {
        for event in persisted {
            if let ChatEvent::PermissionRequest { id, tool, .. } = event {
                self.asks.entry(id.clone()).or_insert((tool.clone(), now));
            }
        }
        for event in persisted {
            if let ChatEvent::PermissionDecision { id, .. } = event {
                self.asks.remove(id);
            }
        }
    }

    fn is_asking(&self) -> bool {
        !self.asks.is_empty()
    }

    /// The failure of the call when a permission request of the child has waited past
    /// [`UNANSWERED_ASK_GRACE`]: nobody watches a sub-agent's session, so nobody would
    /// answer it before the parent's turn timeout.
    fn unanswered_ask(&self, now: tokio::time::Instant) -> Option<Result<String, String>> {
        let (tool, _) = self
            .asks
            .values()
            .find(|(_, since)| now.duration_since(*since) >= UNANSWERED_ASK_GRACE)?;
        Some(Err(format!(
            "the sub-agent needed a permission for `{tool}`, and a sub-agent cannot ask for one \
             (nobody answers in its session): do that step yourself, or give it a task that \
             needs no permission"
        )))
    }
}

/// What the call answers when the child closed before its `result`.
const CLOSED_BEFORE_ANSWER: &str = "the sub-agent's session was closed before it answered";

/// How long a permission request of the child may wait before the call fails (P18b round
/// 3, fix B). A request one of the child's own grants covers (a lasting rule of its owner)
/// is answered by the backend itself within moments of being shown
/// (`AgentSessionHandle::emit`: its `permission_decision` follows); anything still waiting
/// after this has nobody to answer it.
const UNANSWERED_ASK_GRACE: Duration = Duration::from_secs(3);

/// How often the parent re-reads the thread of a child whose stream lagged, and checks its
/// waiting permission requests (P18b round 3, fix C).
const READ_BACK_EVERY: Duration = Duration::from_secs(1);

/// How many of a child's most recent persisted events [`ChatManager::persisted_answer`]
/// reads back (P18b round 2, fix 5): enough for `answer_of` to find a `result` that is
/// not buried under a very chatty child's thread, without reading the whole thing.
const PERSISTED_READ_WINDOW: i64 = 5_000;

/// A panic injected into [`ChatManager::open_wait_close`] between the child's answer and
/// its explicit close, keyed by parent session id (unique per test): the proof that the
/// close guard is wired into the real task, not just that the struct works (P18b round 3,
/// fix E). That point is the one the guard did NOT cover before: it was disarmed there.
#[cfg(test)]
pub(crate) mod panic_injection {
    static BEFORE_CLOSE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    /// The next sub-agent of `parent` panics right before its child's `close_session`.
    pub(crate) fn before_close(parent: &str) {
        BEFORE_CLOSE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(parent.to_string());
    }

    pub(super) fn panic_if_injected(parent: &str) {
        let hit = {
            let mut armed = BEFORE_CLOSE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match armed.iter().position(|p| p == parent) {
                Some(i) => {
                    armed.remove(i);
                    true
                }
                None => false,
            }
        };
        if hit {
            panic!("injected panic before the sub-agent's close (test)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::GraphStore;

    fn bare_manager() -> (Arc<crate::neo4j::mock::MockGraphStore>, Arc<ChatManager>) {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let manager = Arc::new(ChatManager::new_without_memory(
            dyn_graph,
            crate::test_helpers::mock_app_state().meili,
            Default::default(),
        ));
        (graph, manager)
    }

    fn said(t: &str) -> ChatEvent {
        ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: None,
        }
    }

    fn asked(id: &str) -> ChatEvent {
        ChatEvent::PermissionRequest {
            id: id.into(),
            tool: "Bash".into(),
            input: serde_json::json!({"command": "rm -rf build"}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
            tool_use_id: None,
            always: None,
        }
    }

    /// P18b round 3, fix C: the stream lags, and the child's `result` reaches its thread
    /// only 6 s later — past the ~4–5 s the round-2 poll gave it. The parent still gets the
    /// answer, about when it is stored, not at its turn timeout. Red before the fix: after
    /// its 5 reads the poll went back to `recv()` for good, and the stream (still open,
    /// quiet) never said anything again.
    #[tokio::test(start_paused = true)]
    async fn a_result_stored_long_after_a_lag_still_reaches_the_parent() {
        let (graph, manager) = bare_manager();
        let uuid = Uuid::new_v4();
        let sid = uuid.to_string();
        let (tx, rx) = tokio::sync::broadcast::channel::<ChatEvent>(2);
        tx.send(said("working")).unwrap();
        let started = tokio::time::Instant::now();
        let waiting = {
            let manager = manager.clone();
            let sid = sid.clone();
            tokio::spawn(async move { manager.answer_from(&sid, rx).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Overrun: eight events into two slots, without yielding to the reader.
        for i in 0..8 {
            tx.send(said(&format!("chatter {i}"))).unwrap();
        }
        tokio::time::sleep(Duration::from_secs(6)).await;
        assert!(!waiting.is_finished(), "nothing stored yet: it still waits");
        graph
            .store_chat_events(
                uuid,
                vec![crate::neo4j::models::ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: 1,
                    event_type: "result".into(),
                    data: serde_json::json!({
                        "type": "result", "session_id": sid, "duration_ms": 1,
                        "subtype": "success", "is_error": false, "result_text": "LATE-ANSWER",
                    })
                    .to_string(),
                    created_at: chrono::Utc::now(),
                }],
            )
            .await
            .unwrap();
        // The stream stays open and quiet: only the read-back can end the wait.
        let out = tokio::time::timeout(Duration::from_secs(60), waiting)
            .await
            .expect("the parent must not wait for its turn timeout")
            .unwrap();
        assert_eq!(out, Ok("LATE-ANSWER".to_string()));
        assert!(
            started.elapsed() < Duration::from_secs(9),
            "answered about when stored: {:?}",
            started.elapsed()
        );
        drop(tx);
    }

    /// P18b round 3, fix B: a permission request of the child nobody answers fails the call
    /// within [`UNANSWERED_ASK_GRACE`], not at the parent's turn timeout; one the backend
    /// answers itself (a grant covers it: its `permission_decision` follows) does not.
    #[tokio::test(start_paused = true)]
    async fn an_unanswered_permission_fails_the_call_and_a_covered_one_does_not() {
        let (_graph, manager) = bare_manager();
        let sid = Uuid::new_v4().to_string();

        let (tx, rx) = tokio::sync::broadcast::channel::<ChatEvent>(16);
        let started = tokio::time::Instant::now();
        let waiting = {
            let manager = manager.clone();
            let sid = sid.clone();
            tokio::spawn(async move { manager.answer_from(&sid, rx).await })
        };
        tx.send(asked("p1")).unwrap();
        let out = tokio::time::timeout(Duration::from_secs(60), waiting)
            .await
            .expect("an unanswered permission must not hold the parent")
            .unwrap();
        assert!(
            out.as_ref()
                .is_err_and(|e| e.contains("permission") && e.contains("Bash")),
            "{out:?}"
        );
        assert!(started.elapsed() < UNANSWERED_ASK_GRACE + READ_BACK_EVERY * 2);

        // Covered: decided right away, then the child goes on and answers.
        let (tx, rx) = tokio::sync::broadcast::channel::<ChatEvent>(16);
        let waiting = {
            let manager = manager.clone();
            let sid = sid.clone();
            tokio::spawn(async move { manager.answer_from(&sid, rx).await })
        };
        tx.send(asked("p2")).unwrap();
        tx.send(ChatEvent::PermissionDecision {
            id: "p2".into(),
            allow: true,
            scope: None,
            rule: None,
        })
        .unwrap();
        tokio::time::sleep(UNANSWERED_ASK_GRACE * 3).await;
        assert!(!waiting.is_finished(), "a decided request does not fail it");
        tx.send(ChatEvent::Result {
            session_id: sid.clone(),
            duration_ms: 1,
            cost_usd: None,
            subtype: "success".into(),
            is_error: false,
            num_turns: None,
            result_text: Some("DONE".into()),
            cost: None,
            usage: None,
            model: None,
            stop_reason: None,
        })
        .unwrap();
        assert_eq!(waiting.await.unwrap(), Ok("DONE".to_string()));
    }

    /// P18b round 2, fix 3: a guard still armed when it drops runs its backstop exactly
    /// once. Forcing a real panic through the whole `create_session` → `close_session`
    /// async chain is not clean to set up, so this drives the guard's own `Drop` directly —
    /// the seam `open_wait_close` relies on.
    #[test]
    fn an_armed_guard_runs_its_backstop_on_drop() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        {
            let _guard = ClosePanicGuard::new(move || {
                tx.send(()).unwrap();
            });
            // Dropped here, still armed: the backstop must run.
        }
        rx.try_recv().expect("the backstop ran on drop");
    }

    /// The ordinary path (`disarm` before the guard drops, exactly what `open_wait_close`
    /// does right before its own explicit `close_session`) must not ALSO run the backstop:
    /// the child would be closed twice.
    #[test]
    fn a_disarmed_guard_does_not_run_its_backstop() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        {
            let mut guard = ClosePanicGuard::new(move || {
                tx.send(()).unwrap();
            });
            guard.disarm();
        }
        assert!(
            rx.try_recv().is_err(),
            "a disarmed guard must not run its backstop"
        );
    }

    #[test]
    fn the_answer_is_the_result_text_else_the_last_thing_said() {
        let result = |is_error: bool, text: Option<&str>| ChatEvent::Result {
            session_id: "s".into(),
            duration_ms: 1,
            cost_usd: None,
            subtype: if is_error {
                "error_during_execution"
            } else {
                "success"
            }
            .into(),
            is_error,
            num_turns: None,
            result_text: text.map(str::to_owned),
            cost: None,
            usage: None,
            model: None,
            stop_reason: None,
        };
        let said = |t: &str| ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: None,
        };
        assert_eq!(answer_of(&[said("x")]), None, "no result yet");
        assert_eq!(
            answer_of(&[said("first"), said("last"), result(false, Some("FINAL"))]),
            Some(Ok("FINAL".into()))
        );
        assert_eq!(
            answer_of(&[said("first"), said("last"), result(false, None)]),
            Some(Ok("last".into()))
        );
        let failed = answer_of(&[result(true, Some("boom"))])
            .unwrap()
            .unwrap_err();
        assert!(
            failed.contains("failed") && failed.contains("boom"),
            "{failed}"
        );
    }

    fn ok_result(text: &str) -> ChatEvent {
        ChatEvent::Result {
            session_id: "s".into(),
            duration_ms: 1,
            cost_usd: None,
            subtype: "success".into(),
            is_error: false,
            num_turns: None,
            result_text: Some(text.to_owned()),
            cost: None,
            usage: None,
            model: None,
            stop_reason: None,
        }
    }

    /// N3: a sub-agent that copies a whole file does not fill its parent's context.
    #[test]
    fn a_long_answer_is_cut_and_says_so() {
        let long = "x".repeat(1024 * 1024);
        let answer = answer_of(&[ok_result(&long)]).unwrap().unwrap();
        assert!(
            answer.chars().count() < MAX_ANSWER_CHARS + 100,
            "{}",
            answer.len()
        );
        assert!(answer.ends_with(&format!(
            "[… {} characters of the sub-agent's answer left out]",
            1024 * 1024 - MAX_ANSWER_CHARS
        )));
        let short = answer_of(&[ok_result("short")]).unwrap().unwrap();
        assert_eq!(short, "short");
    }

    /// N2: a question a child asks reaches nobody: the call fails, saying why.
    #[test]
    fn a_child_that_asks_the_user_fails_the_call() {
        let asked = ChatEvent::AskUserQuestion {
            id: "q".into(),
            tool_call_id: "cq".into(),
            questions: serde_json::json!([{"question": "Which?"}]),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
            synthetic: Some(true),
        };
        let out = answer_of(&[asked, ok_result("asked")]).unwrap();
        assert!(out.is_err_and(|e| e.contains("cannot ask")));
    }

    /// P18b, fix 4: what the parent holds of a child's stream is bounded — at most the
    /// question, the last thing said and the `result` — and `answer_of` still reads the
    /// same answer out of it.
    #[test]
    fn what_is_kept_of_a_chatty_child_is_bounded() {
        let said = |t: &str| ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: None,
        };
        let inside = ChatEvent::AssistantText {
            content: "inside a tool call".into(),
            parent_tool_use_id: Some("toolu_1".into()),
        };
        let mut seen = Vec::new();
        for i in 0..5_000 {
            keep_for_answer(&mut seen, said(&format!("line {i}")));
            keep_for_answer(&mut seen, inside.clone());
            keep_for_answer(
                &mut seen,
                ChatEvent::ToolUse {
                    id: "t".into(),
                    tool: "Read".into(),
                    input: serde_json::json!({}),
                    parent_tool_use_id: None,
                    category: None,
                    canonical: None,
                },
            );
        }
        assert_eq!(seen.len(), 1, "only the last thing said is held: {seen:?}");
        keep_for_answer(&mut seen, ok_result("FINAL"));
        keep_for_answer(&mut seen, ok_result("A SECOND RESULT"));
        assert_eq!(seen.len(), 2);
        assert_eq!(answer_of(&seen), Some(Ok("FINAL".into())));

        // Without a result text, the fallback is the LAST thing said, not the first.
        let mut seen = Vec::new();
        keep_for_answer(&mut seen, said("first"));
        keep_for_answer(&mut seen, said("last"));
        keep_for_answer(
            &mut seen,
            ChatEvent::Result {
                session_id: "s".into(),
                duration_ms: 1,
                cost_usd: None,
                subtype: "success".into(),
                is_error: false,
                num_turns: None,
                result_text: None,
                cost: None,
                usage: None,
                model: None,
                stop_reason: None,
            },
        );
        assert_eq!(answer_of(&seen), Some(Ok("last".into())));

        // A question is kept once, whatever follows.
        let question = ChatEvent::AskUserQuestion {
            id: "q".into(),
            tool_call_id: "cq".into(),
            questions: serde_json::json!([{"question": "Which?"}]),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
            synthetic: Some(true),
        };
        let mut seen = Vec::new();
        keep_for_answer(&mut seen, question.clone());
        keep_for_answer(&mut seen, question);
        keep_for_answer(&mut seen, ok_result("whatever"));
        assert_eq!(seen.len(), 2);
        assert!(answer_of(&seen)
            .unwrap()
            .is_err_and(|e| e.contains("cannot ask")));
    }

    /// P18b round 2, fix 5: a child's `result` landing well past
    /// [`PERSISTED_READ_WINDOW`] events is still found — `persisted_answer` reads the
    /// TAIL of the thread. Red before the fix: `get_chat_events(uuid, 0, 5_000)` reads
    /// forward from the head and never reaches a `result` stored at seq 6001.
    #[tokio::test]
    async fn persisted_answer_reads_the_tail_not_the_head() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let manager = ChatManager::new_without_memory(
            dyn_graph,
            crate::test_helpers::mock_app_state().meili,
            Default::default(),
        );
        let sid = Uuid::new_v4();
        let said = |t: &str| ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: None,
        };
        let mut events: Vec<crate::neo4j::models::ChatEventRecord> = (1..=6_000i64)
            .map(|i| crate::neo4j::models::ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: sid,
                seq: i,
                event_type: "assistant_text".into(),
                data: serde_json::to_string(&said(&format!("line {i}"))).unwrap(),
                created_at: chrono::Utc::now(),
            })
            .collect();
        events.push(crate::neo4j::models::ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: sid,
            seq: 6_001,
            event_type: "result".into(),
            data: serde_json::json!({
                "type": "result", "session_id": sid.to_string(), "duration_ms": 1,
                "subtype": "success", "is_error": false, "result_text": "TAIL-ANSWER",
            })
            .to_string(),
            created_at: chrono::Utc::now(),
        });
        graph.store_chat_events(sid, events).await.unwrap();
        let answer = manager.persisted_answer(&sid.to_string()).await;
        assert_eq!(answer, Some(Ok("TAIL-ANSWER".to_string())));
    }
}
