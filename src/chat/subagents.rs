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
//! * the parent's provider and model, the parent's claims;
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
//! the subscription) and after a `Lagged` (the `result` may have been among the dropped
//! events; waiting on would last until the parent's turn timeout). What is kept from the
//! stream is bounded ([`keep_for_answer`]): a chatty child cannot grow it.
//!
//! Depth is held three times: the backend gives [`SubagentHooks`] only to a native session
//! whose record has no `spawned_by` (`ChatManager::build_agent_spec_with_access`); a spawned
//! native session is opened with `SessionSpec::lineage` at depth 1, so the harness itself
//! refuses it the tool; and the envelope refuses a parent that has a parent. A child keeps its
//! `spawned_by` when it moves to another provider (relay).
//!
//! The cap of live children holds under parallel calls (the harness runs the calls of one step
//! concurrently): a slot of the parent is taken BEFORE the envelope counts its children and is
//! held until the sub-agent ends ([`MAX_LIVE_CHILDREN`](super::envelope::MAX_LIVE_CHILDREN)
//! slots per parent). The ceiling is the parent's CURRENT mode: a native session's mode changes
//! are kept live and persisted (`set_session_permission_mode`).
//!
//! What a child may not do: ask the user (`AskUserQuestion` is denied to a spawned native
//! session; a question seen anyway fails the call); return more than [`MAX_ANSWER_CHARS`]
//! characters (the rest is cut, and said). The call waits as long as the child works, bounded
//! by the parent's turn timeout; a child in a mode that asks for permissions shows its request
//! in its own session, where nobody may be watching: it then waits until that timeout. Budget
//! and cost of the children are not counted in the parent's (task B36).

use std::sync::{Arc, Weak};

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
        // Held for the whole call: parallel calls cannot all pass the count below (A17).
        let _slot = SubagentSlot::take(&self, parent_session_id).ok_or_else(|| {
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
                    .open_wait_close(child, cancelled, &parent, &tool_use_id)
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
    /// closed as soon as the opening has given its id.
    async fn open_wait_close(
        self: Arc<Self>,
        child: ChatRequest,
        cancelled: tokio::sync::oneshot::Receiver<std::convert::Infallible>,
        parent_session_id: &str,
        tool_use_id: &str,
    ) -> Result<String, String> {
        let opened = self
            .create_session(&child)
            .await
            .map_err(|e| format!("the sub-agent could not start: {e:#}"))?;
        let child_id = opened.session_id;
        tracing::info!(parent = %parent_session_id, child = %child_id, tool_use_id = %tool_use_id, "sub-agent started");
        // The child exists from here: whatever ends this task, it is closed first (it stays
        // resumable and visible).
        let answer = tokio::select! {
            biased;
            _ = cancelled => Err("the sub-agent's call was cancelled".to_owned()),
            answer = self.wait_for_answer(&child_id) => answer,
        };
        if let Err(e) = self.close_session(&child_id).await {
            tracing::debug!(child = %child_id, error = %e, "sub-agent close");
        }
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
        // A turn that ended before the subscription is in the store.
        if let Some(answer) = self.persisted_answer(sid).await {
            return answer;
        }
        let mut seen: Vec<ChatEvent> = Vec::new();
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let done = is_result(&event);
                    let closed = matches!(event, ChatEvent::SessionClosed { .. });
                    keep_for_answer(&mut seen, event);
                    if done {
                        return answer_of(&seen).unwrap_or_else(|| Ok(String::new()));
                    }
                    if closed {
                        return Err("the sub-agent's session was closed before it answered".into());
                    }
                }
                // Events were dropped, and the child's `result` may have been among them:
                // waiting on would last until the parent's 30-minute turn timeout. The
                // thread is the truth — read it back (P18b, fix 4).
                Err(tokio::sync::broadcast::error::RecvError::Lagged(dropped)) => {
                    tracing::warn!(child = %sid, dropped, "sub-agent events dropped: its thread is read back");
                    if let Some(answer) = self.persisted_answer(sid).await {
                        return answer;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return Err("the sub-agent's session ended before it answered".into())
                }
            }
        }
    }

    /// The answer of `sid` as its STORED thread tells it, when its `result` is already there.
    async fn persisted_answer(&self, sid: &str) -> Option<Result<String, String>> {
        let uuid = Uuid::parse_str(sid).ok()?;
        let persisted: Vec<ChatEvent> = self
            .graph
            .get_chat_events(uuid, 0, 5_000)
            .await
            .unwrap_or_default()
            .iter()
            .filter_map(|r| serde_json::from_str(&r.data).ok())
            .collect();
        answer_of(&persisted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
