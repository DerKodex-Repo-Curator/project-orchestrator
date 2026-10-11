//! Read-only MIRROR children of a Claude Code session's sub-agents (P18c, decision 1ec3e540).
//!
//! A sub-agent Claude Code launches with its own `Task` / `Agent` tool runs INSIDE the CLI
//! (its `.claude/agents` definitions, its shared context, its model choice): it has no
//! project-orchestrator session of its own. Its activity reaches the parent's stream with
//! `parent_tool_use_id` = the id of that call, and the parent shows it nested under the
//! call's card. A native sub-agent, by contrast, is a real child session
//! (`chat::subagents`). This module gives the Claude Code sub-agent the same LINK without
//! touching what the CLI does: a child conversation that only MIRRORS it.
//!
//! # Strictly passive
//!
//! A [`MirrorTap`] SEES the parent's events after they were emitted, as the parent's other
//! observers do; it never alters, delays nor intercepts the call or its events in the
//! parent (the parent's thread and its nested display are exactly what they were). The tap
//! only clones the few events it needs into an unbounded channel, in order; a task of its
//! own does every write, so the parent's stream never waits on the mirror.
//!
//! Where the parent's events are seen:
//! - agent engine (the default for Claude Code): `ManagerTurnServices::observe`, which the
//!   runtime calls for every event a session emits (`AgentSessionHandle::emit_one`); only a
//!   session whose provider kind is `claude_code` gets a tap (`finish_agent_open`);
//! - historical engine: the `emit_chat` helper of `ChatManager::stream_response`, through
//!   which every event of a turn is broadcast, plus an explicit end of turn before the turn's
//!   `streaming_status: false` (`ActiveSession::mirror_tap`).
//!
//! A native session never gets a tap: its sub-agents are real children already.
//!
//! # The mirror
//!
//! On the FIRST event carrying `parent_tool_use_id = X` where X is a top-level call of the
//! `Task` or `Agent` tool, ONE child session node is written:
//! `spawned_by = conversation { parent_session_id, tool_use_id: X }` (the join the parent's
//! Agent card already makes), the parent's provider, model, cwd, project, workspace, owner,
//! access and engine, the title from the call's `description` (else its prompt), and the
//! marker [`ChatSessionNode::mirror_of`](crate::neo4j::models::ChatSessionNode::mirror_of)
//! `= "claude_code"` — on the wire `"mirror_of": "claude_code"` in every session response,
//! omitted for every other session. One CRUD `created` is emitted with the trimmed payload
//! every session gets (`{project_slug, spawned_by: {type, parent_session_id}}`) plus
//! `spawned_by.tool_use_id` and `mirror_of` ([`created_payload`]: the client links the mirror
//! to the Agent card at once), then `updated {is_streaming: true}`.
//!
//! Its thread: the call's prompt as its first `user_message`, then every event carrying
//! `parent_tool_use_id = X`, in order, WITHOUT that parent id (the child shows them at its top
//! level). Permission requests and questions are not copied: they are answered in the parent,
//! and a copy would sit unanswerable in the mirror. Deltas are sent live, never stored (as
//! everywhere).
//!
//! Closed — its `result` written (the call's `tool_result` text, `is_error` kept), then
//! `updated {is_streaming: false}` — when the call's `tool_result` (or its cancel) reaches
//! the parent, else when the parent's turn ends, else when the parent session closes (its
//! tap is dropped). Events of X arriving after the close are dropped.
//!
//! Only FIRST-LEVEL calls of a session are mirrored: a sub-agent's own `Task` call (an
//! event whose parent is a call made inside a mirror) goes into its first-level mirror's
//! thread, keeping its parent id so it shows nested there.
//!
//! # Idempotency
//!
//! Never two mirrors for one (parent, call): the manager's [`MirrorRegistry`] knows every
//! mirror this process wrote (a resume replaces a tap, never the registry), and the graph is
//! asked before writing one (the parent's children whose `spawned_by.tool_use_id` is X and
//! that carry the marker): a mirror found is reused — its thread continued when open, its
//! late events dropped when it already has its `result`. Nothing is created on a reconnect
//! or a replay: those read stored events, they do not emit them again.
//!
//! # Read-only
//!
//! A mirror has no process: a message sent to it is refused before anything starts a turn
//! ([`MirrorReadOnly`]: HTTP 409 `{error, code: "mirror_read_only", retryable: false}` on
//! REST — a send, a resume through `POST /api/chat/sessions`, a permission answer; on the
//! WebSocket the flat frame `{type: "error", code: "mirror_read_only", message}` for a
//! message, queued or not, and an answer to a permission or a question), and the restart
//! recovery never resumes one (`manager::recovery`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

use super::types::{ChatEvent, SpawnedBy};
use crate::neo4j::models::{ChatEventRecord, ChatSessionNode};
use crate::neo4j::GraphStore;

/// The value of `mirror_of` on a mirror of a Claude Code sub-agent.
pub const MIRROR_OF_CLAUDE_CODE: &str = "claude_code";

/// The stable code of the refusal of a message sent to a mirror (HTTP 409).
pub const MIRROR_READ_ONLY: &str = "mirror_read_only";

/// A message was sent to a mirror ([module documentation](self)).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "this conversation mirrors a Claude Code sub-agent and is read-only: write in its parent conversation"
)]
pub struct MirrorReadOnly;

impl MirrorReadOnly {
    /// The typed refusal the API returns: 409, code [`MIRROR_READ_ONLY`].
    pub fn failure() -> super::provider::errors::OpenFailure {
        super::provider::errors::OpenFailure {
            status: 409,
            code: MIRROR_READ_ONLY,
            message: MirrorReadOnly.to_string(),
            provider_id: None,
            action: None,
            retryable: false,
            retry_after_ms: None,
            fallbacks: None,
            consent_scope: None,
            candidates: None,
        }
    }
}

/// Whether `tool` is Claude Code's sub-agent tool (`Task`, renamed `Agent`; nexus'
/// `claude_code::policy_map` puts both in the `agent` category).
pub fn is_subagent_tool(tool: &str) -> bool {
    matches!(tool, "Task" | "Agent")
}

/// The parent call id an event carries, when it has the field.
fn parent_of(event: &ChatEvent) -> Option<&str> {
    match event {
        ChatEvent::AssistantText {
            parent_tool_use_id, ..
        }
        | ChatEvent::Thinking {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolUse {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolResult {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolUseInputResolved {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolCancelled {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolTiming {
            parent_tool_use_id, ..
        }
        | ChatEvent::PermissionRequest {
            parent_tool_use_id, ..
        }
        | ChatEvent::AskUserQuestion {
            parent_tool_use_id, ..
        }
        | ChatEvent::StreamDelta {
            parent_tool_use_id, ..
        }
        | ChatEvent::Error {
            parent_tool_use_id, ..
        } => parent_tool_use_id.as_deref(),
        _ => None,
    }
}

/// The parent field of an event the mirror copies (`None`: not copied).
fn copied_parent_mut(event: &mut ChatEvent) -> Option<&mut Option<String>> {
    match event {
        ChatEvent::AssistantText {
            parent_tool_use_id, ..
        }
        | ChatEvent::Thinking {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolUse {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolResult {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolUseInputResolved {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolCancelled {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolTiming {
            parent_tool_use_id, ..
        }
        | ChatEvent::StreamDelta {
            parent_tool_use_id, ..
        }
        | ChatEvent::Error {
            parent_tool_use_id, ..
        } => Some(parent_tool_use_id),
        // Answered in the parent: never copied (module documentation).
        _ => None,
    }
}

/// The text of a `tool_result`: a string, or the text blocks of a structured content.
fn result_text(result: &serde_json::Value) -> String {
    match result {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                b.get("text")
                    .and_then(|t| t.as_str())
                    .or_else(|| b.as_str())
                    .map(str::to_string)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// The CRUD `created` payload of a mirror: every session's trimmed payload
/// (`{project_slug, spawned_by: {type, parent_session_id}}`), plus what lets a client link
/// the mirror at once to the Agent card of its parent: `spawned_by.tool_use_id` (the call)
/// and `mirror_of`. Neither names a path, a model nor an origin id.
pub(crate) fn created_payload(
    spawned_by: &str,
    project_slug: Option<&str>,
    call: &str,
) -> serde_json::Value {
    let mut payload = super::manager::session_created_payload_of(Some(spawned_by), project_slug);
    if let Some(spawned_by) = payload
        .get_mut("spawned_by")
        .and_then(|v| v.as_object_mut())
    {
        spawned_by.insert("tool_use_id".into(), call.into());
    }
    payload["mirror_of"] = MIRROR_OF_CLAUDE_CODE.into();
    payload
}

/// What the manager knows of the mirrors of this process, shared by every tap.
#[derive(Default)]
pub struct MirrorRegistry {
    /// (parent session, call id) -> mirror id: never two mirrors for one call.
    by_call: std::sync::Mutex<HashMap<(String, String), Uuid>>,
    /// The mirrors that already have their `result`.
    closed: std::sync::Mutex<HashSet<Uuid>>,
    /// The open mirrors and their live broadcast (what `ChatManager::subscribe` hands).
    open: std::sync::Mutex<HashMap<String, broadcast::Sender<ChatEvent>>>,
}

impl MirrorRegistry {
    /// A live receiver of the open mirror `session_id`.
    pub fn subscribe(&self, session_id: &str) -> Option<broadcast::Receiver<ChatEvent>> {
        self.open
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(session_id)
            .map(broadcast::Sender::subscribe)
    }

    /// Whether `session_id` is a mirror still open here.
    pub fn is_open(&self, session_id: &str) -> bool {
        self.open
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains_key(session_id)
    }

    /// The open mirrors.
    pub fn open_ids(&self) -> Vec<Uuid> {
        self.open
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keys()
            .filter_map(|id| Uuid::parse_str(id).ok())
            .collect()
    }
}

/// What a tap writes with.
#[derive(Clone)]
pub(crate) struct MirrorDeps {
    pub graph: Arc<dyn GraphStore>,
    pub event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
    pub nats: Option<Arc<crate::events::NatsEmitter>>,
    pub registry: Arc<MirrorRegistry>,
}

enum Msg {
    Event(Box<ChatEvent>),
    TurnEnded,
}

/// The passive observer of ONE Claude Code session's events ([module documentation](self)).
/// Dropping the last handle closes the mirrors still open.
pub struct MirrorTap {
    tx: mpsc::UnboundedSender<Msg>,
}

impl MirrorTap {
    /// Starts the tap of `parent_session_id` (its worker runs on the current runtime).
    pub(crate) fn spawn(parent_session_id: &str, deps: MirrorDeps) -> Arc<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let worker = Worker {
            parent: parent_session_id.to_string(),
            deps,
            calls: HashMap::new(),
            nested: HashMap::new(),
            mirrors: HashMap::new(),
            closed: HashSet::new(),
            parent_node: None,
        };
        tokio::spawn(worker.run(rx));
        Arc::new(Self { tx })
    }

    /// Sees one event of the parent, after it was emitted. Never blocks, never fails: what
    /// the mirror does not need is not even cloned.
    pub fn observe(&self, event: &ChatEvent) {
        let wanted = match event {
            _ if parent_of(event).is_some() => true,
            ChatEvent::ToolUse { tool, .. } => is_subagent_tool(tool),
            ChatEvent::ToolUseInputResolved { .. }
            | ChatEvent::ToolResult { .. }
            | ChatEvent::ToolCancelled { .. } => true,
            ChatEvent::Result { .. }
            | ChatEvent::StreamingStatus {
                is_streaming: false,
            } => {
                let _ = self.tx.send(Msg::TurnEnded);
                return;
            }
            _ => false,
        };
        if wanted {
            let _ = self.tx.send(Msg::Event(Box::new(event.clone())));
        }
    }

    /// The parent's turn ended (the historical engine says it explicitly).
    pub fn turn_ended(&self) {
        let _ = self.tx.send(Msg::TurnEnded);
    }
}

/// One open mirror.
struct Mirror {
    id: Uuid,
    next_seq: i64,
    tx: broadcast::Sender<ChatEvent>,
    opened: Instant,
}

/// The tap's state, owned by its task: every write happens here, in order.
struct Worker {
    parent: String,
    deps: MirrorDeps,
    /// Top-level sub-agent calls of the parent: id -> input as last known.
    calls: HashMap<String, serde_json::Value>,
    /// Calls made INSIDE a mirror (any depth): id -> the first-level call.
    nested: HashMap<String, String>,
    /// Open mirrors by first-level call id.
    mirrors: HashMap<String, Mirror>,
    /// First-level calls whose mirror is closed (or could not be written).
    closed: HashSet<String>,
    parent_node: Option<ChatSessionNode>,
}

impl Worker {
    async fn run(mut self, mut rx: mpsc::UnboundedReceiver<Msg>) {
        while let Some(msg) = rx.recv().await {
            match msg {
                Msg::Event(event) => self.handle(*event).await,
                Msg::TurnEnded => {
                    self.close_all("The parent turn ended before the sub-agent answered.")
                        .await
                }
            }
        }
        // The parent session is gone (every handle of the tap dropped).
        self.close_all("The parent session closed before the sub-agent answered.")
            .await;
    }

    async fn handle(&mut self, mut event: ChatEvent) {
        match parent_of(&event).map(str::to_string) {
            None => match &event {
                ChatEvent::ToolUse {
                    id, tool, input, ..
                } if is_subagent_tool(tool) => {
                    self.calls.insert(id.clone(), input.clone());
                }
                ChatEvent::ToolUseInputResolved { id, input, .. } => {
                    if let Some(known) = self.calls.get_mut(id) {
                        *known = input.clone();
                    }
                }
                ChatEvent::ToolResult {
                    id,
                    result,
                    is_error,
                    ..
                } => {
                    let (id, text, is_error) = (id.clone(), result_text(result), *is_error);
                    self.calls.remove(&id);
                    self.close(&id, text, is_error, false).await;
                }
                ChatEvent::ToolCancelled { id, .. } => {
                    let id = id.clone();
                    self.calls.remove(&id);
                    self.close(&id, "The sub-agent was cancelled.".to_string(), true, true)
                        .await;
                }
                _ => {}
            },
            Some(parent) => {
                let first = if self.calls.contains_key(&parent) {
                    parent.clone()
                } else if let Some(first) = self.nested.get(&parent) {
                    first.clone()
                } else {
                    // Its call was not seen as a sub-agent call by this tap.
                    return;
                };
                if self.closed.contains(&first) {
                    return;
                }
                if !self.mirrors.contains_key(&first) && !self.open_mirror(&first).await {
                    return;
                }
                if let ChatEvent::ToolUse { id, .. } = &event {
                    self.nested.insert(id.clone(), first.clone());
                }
                let Some(slot) = copied_parent_mut(&mut event) else {
                    return;
                };
                // The first level shows at the mirror's top level; deeper calls stay nested.
                if slot.as_deref() == Some(first.as_str()) {
                    *slot = None;
                }
                self.append(&first, event).await;
            }
        }
    }

    /// The parent's node (read once).
    async fn parent_node(&mut self) -> Option<ChatSessionNode> {
        if self.parent_node.is_none() {
            let uuid = Uuid::parse_str(&self.parent).ok()?;
            self.parent_node = self.deps.graph.get_chat_session(uuid).await.ok().flatten();
        }
        self.parent_node.clone()
    }

    /// The mirror of `call`, found or written. `false`: there is none to write into (it is
    /// closed already, or could not be written).
    async fn open_mirror(&mut self, call: &str) -> bool {
        let key = (self.parent.clone(), call.to_string());
        let known = self
            .deps
            .registry
            .by_call
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
            .copied();
        let existing = match known {
            Some(id) => Some(id),
            None => self.stored_mirror(call).await,
        };
        let id = match existing {
            Some(id) => {
                if self.is_closed(id).await {
                    self.closed.insert(call.to_string());
                    return false;
                }
                id
            }
            None => match self.write_mirror(call).await {
                Some(id) => id,
                None => {
                    self.closed.insert(call.to_string());
                    return false;
                }
            },
        };
        self.deps
            .registry
            .by_call
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key, id);
        let next_seq = self
            .deps
            .graph
            .get_latest_chat_event_seq(id)
            .await
            .unwrap_or(0)
            + 1;
        let (tx, _) = broadcast::channel(256);
        self.deps
            .registry
            .open
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id.to_string(), tx.clone());
        self.mirrors.insert(
            call.to_string(),
            Mirror {
                id,
                next_seq,
                tx,
                opened: Instant::now(),
            },
        );
        self.streaming(id, true);
        // A mirror written now opens on the call's prompt.
        if existing.is_none() {
            if let Some(prompt) = self.prompt(call) {
                self.append(call, ChatEvent::UserMessage { content: prompt })
                    .await;
            }
        }
        true
    }

    /// A mirror of `call` the graph already holds (written by an earlier tap or process).
    async fn stored_mirror(&self, call: &str) -> Option<Uuid> {
        let parent = Uuid::parse_str(&self.parent).ok()?;
        let children = self.deps.graph.get_session_children(parent).await.ok()?;
        children.into_iter().find_map(|child| {
            let linked = matches!(
                child.spawned_by.as_deref().and_then(SpawnedBy::from_json_str),
                Some(SpawnedBy::Conversation { parent_session_id, tool_use_id: Some(t) })
                    if parent_session_id == parent && t == call
            );
            (linked && child.mirror_of.is_some()).then_some(child.id)
        })
    }

    /// Whether the mirror `id` already has its `result`.
    async fn is_closed(&self, id: Uuid) -> bool {
        if self
            .deps
            .registry
            .closed
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains(&id)
        {
            return true;
        }
        let Ok(latest) = self.deps.graph.get_latest_chat_event_seq(id).await else {
            return false;
        };
        self.deps
            .graph
            .get_chat_events(id, (latest - 1).max(0), 1)
            .await
            .map(|tail| tail.iter().any(|e| e.event_type == "result"))
            .unwrap_or(false)
    }

    /// The call's `prompt`.
    fn prompt(&self, call: &str) -> Option<String> {
        self.calls
            .get(call)
            .and_then(|input| input.get("prompt"))
            .and_then(|p| p.as_str())
            .filter(|p| !p.trim().is_empty())
            .map(str::to_string)
    }

    /// Writes the mirror node of `call`, and announces it.
    async fn write_mirror(&mut self, call: &str) -> Option<Uuid> {
        let parent = self.parent_node().await?;
        let input = self.calls.get(call).cloned().unwrap_or_default();
        let prompt = self.prompt(call);
        let description = input
            .get("description")
            .and_then(|d| d.as_str())
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .map(str::to_string);
        let from_prompt = prompt
            .as_deref()
            .and_then(super::session_record::title_and_preview);
        let title = description
            .or_else(|| from_prompt.as_ref().map(|(t, _)| t.clone()))
            .unwrap_or_else(|| "Claude Code sub-agent".to_string());
        let spawned_by = SpawnedBy::Conversation {
            parent_session_id: parent.id,
            tool_use_id: Some(call.to_string()),
        }
        .to_json_string();
        let now = chrono::Utc::now();
        let node = ChatSessionNode {
            id: Uuid::new_v4(),
            cli_session_id: None,
            project_slug: parent.project_slug.clone(),
            workspace_slug: parent.workspace_slug.clone(),
            cwd: parent.cwd.clone(),
            title: Some(title),
            model: parent.model.clone(),
            created_at: now,
            updated_at: now,
            message_count: i64::from(prompt.is_some()),
            total_cost_usd: None,
            conversation_id: None,
            preview: from_prompt.map(|(_, p)| p),
            permission_mode: parent.permission_mode.clone(),
            add_dirs: parent.add_dirs.clone(),
            spawned_by: Some(spawned_by.clone()),
            provider_id: parent.provider_id.clone(),
            routed_by: None,
            routing_mode: None,
            routing_pool: None,
            capabilities: None,
            engine: parent.engine.clone(),
            resume_token: None,
            execution_place: parent.execution_place,
            access: parent.access,
            owner: parent.owner.clone(),
            mirror_of: Some(MIRROR_OF_CLAUDE_CODE.to_string()),
        };
        if let Err(e) = self.deps.graph.create_chat_session(&node).await {
            tracing::warn!(parent = %self.parent, call, error = %e, "the mirror of a Claude Code sub-agent could not be written");
            return None;
        }
        if let Err(e) = self
            .deps
            .graph
            .create_spawned_by_relation(
                &node.id.to_string(),
                &self.parent,
                "conversation",
                None,
                None,
            )
            .await
        {
            tracing::warn!(mirror = %node.id, error = %e, "SPAWNED_BY of a mirror not written");
        }
        if let Some(emitter) = &self.deps.event_emitter {
            emitter.emit_created(
                crate::events::EntityType::ChatSession,
                &node.id.to_string(),
                created_payload(&spawned_by, node.project_slug.as_deref(), call),
                None,
            );
        }
        tracing::info!(parent = %self.parent, call, mirror = %node.id, "mirror of a Claude Code sub-agent opened");
        Some(node.id)
    }

    /// Stores (but a delta) and broadcasts one event of the mirror of `call`.
    async fn append(&mut self, call: &str, event: ChatEvent) {
        let Some(mirror) = self.mirrors.get_mut(call) else {
            return;
        };
        if !matches!(event, ChatEvent::StreamDelta { .. }) {
            let record = ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: mirror.id,
                seq: mirror.next_seq,
                event_type: event.event_type().to_string(),
                data: serde_json::to_string(&event).unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            mirror.next_seq += 1;
            if let Err(e) = self
                .deps
                .graph
                .store_chat_events(mirror.id, vec![record])
                .await
            {
                tracing::warn!(mirror = %mirror.id, error = %e, "an event of a mirror was not stored");
            }
        }
        let _ = mirror.tx.send(event.clone());
        if let Some(nats) = &self.deps.nats {
            nats.publish_chat_event(&mirror.id.to_string(), event);
        }
    }

    fn streaming(&self, id: Uuid, is_streaming: bool) {
        if let Some(emitter) = &self.deps.event_emitter {
            emitter.emit_updated(
                crate::events::EntityType::ChatSession,
                &id.to_string(),
                serde_json::json!({ "is_streaming": is_streaming }),
                None,
            );
        }
    }

    /// Ends the mirror of `call` on its result. Nothing when it has none open.
    async fn close(&mut self, call: &str, text: String, is_error: bool, interrupted: bool) {
        self.closed.insert(call.to_string());
        if !self.mirrors.contains_key(call) {
            return;
        }
        let (id, elapsed) = {
            let mirror = &self.mirrors[call];
            (mirror.id, mirror.opened.elapsed())
        };
        let result = ChatEvent::Result {
            session_id: id.to_string(),
            duration_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            cost_usd: None,
            subtype: if is_error {
                "error_during_execution"
            } else {
                "success"
            }
            .to_string(),
            is_error,
            num_turns: None,
            result_text: Some(text),
            cost: None,
            usage: None,
            model: None,
            stop_reason: Some(
                if interrupted {
                    "interrupted"
                } else if is_error {
                    "error"
                } else {
                    "completed"
                }
                .to_string(),
            ),
        };
        self.append(call, result).await;
        if let Some(mirror) = self.mirrors.remove(call) {
            let _ = mirror.tx.send(ChatEvent::StreamingStatus {
                is_streaming: false,
            });
        }
        self.deps
            .registry
            .closed
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id);
        self.deps
            .registry
            .open
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&id.to_string());
        self.streaming(id, false);
        tracing::info!(parent = %self.parent, call, mirror = %id, "mirror of a Claude Code sub-agent closed");
    }

    /// Ends every mirror still open: the sub-agent was cut.
    async fn close_all(&mut self, why: &str) {
        let open: Vec<String> = self.mirrors.keys().cloned().collect();
        for call in open {
            self.close(&call, why.to_string(), true, true).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_text_of_a_result_is_its_string_or_its_text_blocks() {
        assert_eq!(result_text(&serde_json::json!("done")), "done");
        assert_eq!(
            result_text(&serde_json::json!([
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"}
            ])),
            "a\nb"
        );
        assert_eq!(result_text(&serde_json::Value::Null), "");
    }

    #[test]
    fn only_claude_codes_sub_agent_tools_are_mirrored() {
        assert!(is_subagent_tool("Task"));
        assert!(is_subagent_tool("Agent"));
        assert!(!is_subagent_tool("Bash"));
        assert!(!is_subagent_tool("Read"));
    }

    use super::test_support::{closed_mirrors, mirrors_of, thread, RecordingEmitter};
    use crate::events::CrudAction;
    use crate::neo4j::mock::MockGraphStore;

    const CALL: &str = "toolu_task";

    struct World {
        graph: Arc<MockGraphStore>,
        emitter: Arc<RecordingEmitter>,
        registry: Arc<MirrorRegistry>,
        parent: ChatSessionNode,
    }

    async fn world() -> World {
        let graph = Arc::new(MockGraphStore::new());
        let parent = ChatSessionNode {
            project_slug: Some("proj".into()),
            provider_id: Some("claude-code".into()),
            engine: Some(ChatSessionNode::ENGINE_AGENT.into()),
            owner: Some("owner-1".into()),
            ..crate::test_helpers::test_chat_session(Some("proj"))
        };
        graph.create_chat_session(&parent).await.unwrap();
        World {
            graph,
            emitter: Arc::default(),
            registry: Arc::default(),
            parent,
        }
    }

    impl World {
        fn tap(&self) -> Arc<MirrorTap> {
            MirrorTap::spawn(
                &self.parent.id.to_string(),
                MirrorDeps {
                    graph: self.graph.clone(),
                    event_emitter: Some(self.emitter.clone()),
                    nats: None,
                    registry: Arc::clone(&self.registry),
                },
            )
        }
    }

    fn call(id: &str, tool: &str, input: serde_json::Value, parent: Option<&str>) -> ChatEvent {
        ChatEvent::ToolUse {
            id: id.into(),
            tool: tool.into(),
            input,
            parent_tool_use_id: parent.map(str::to_string),
            category: None,
            canonical: None,
        }
    }

    fn said(text: &str, parent: Option<&str>) -> ChatEvent {
        ChatEvent::AssistantText {
            content: text.into(),
            parent_tool_use_id: parent.map(str::to_string),
        }
    }

    fn answered(id: &str, text: &str, parent: Option<&str>) -> ChatEvent {
        ChatEvent::ToolResult {
            id: id.into(),
            result: serde_json::json!(text),
            is_error: false,
            parent_tool_use_id: parent.map(str::to_string),
        }
    }

    /// The events of a parent turn running one Task call, as both engines emit them: the
    /// call announced with an empty input, then resolved; the sub-agent's activity under
    /// it; the call's result; the parent goes on.
    fn task_turn() -> Vec<ChatEvent> {
        vec![
            call(CALL, "Task", serde_json::json!({}), None),
            ChatEvent::ToolUseInputResolved {
                id: CALL.into(),
                input: serde_json::json!({"description": "find the spec", "prompt": "Look for the spec of X"}),
                parent_tool_use_id: None,
            },
            said("looking", Some(CALL)),
            ChatEvent::StreamDelta {
                text: "live only".into(),
                parent_tool_use_id: Some(CALL.into()),
            },
            call(
                "toolu_read",
                "Read",
                serde_json::json!({"file_path": "a.md"}),
                Some(CALL),
            ),
            ChatEvent::PermissionRequest {
                id: "req-1".into(),
                tool: "Read".into(),
                input: serde_json::json!({}),
                parent_tool_use_id: Some(CALL.into()),
                category: None,
                canonical: None,
                tool_use_id: Some("toolu_read".into()),
                always: None,
            },
            answered("toolu_read", "the spec", Some(CALL)),
            said("found it", Some(CALL)),
            answered(CALL, "The spec is in a.md", None),
            said("the parent goes on", None),
        ]
    }

    /// P18c: one Task call → ONE mirror, linked by `spawned_by.tool_use_id`, marked
    /// `mirror_of: claude_code`, titled from the call's description, holding the prompt then
    /// the sub-agent's events at its top level (no permission request, no delta stored),
    /// closed with the call's result; one `created` (with the call id), then streaming
    /// true / false on the CRUD bus.
    #[tokio::test]
    async fn a_task_call_gets_one_mirror_holding_its_events_closed_by_its_result() {
        let w = world().await;
        let tap = w.tap();
        for event in task_turn() {
            tap.observe(&event);
        }
        let mirrors = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await;
        assert_eq!(mirrors.len(), 1, "exactly one mirror");
        let mirror = &mirrors[0];
        assert_eq!(mirror.mirror_of.as_deref(), Some(MIRROR_OF_CLAUDE_CODE));
        assert_eq!(mirror.title.as_deref(), Some("find the spec"));
        assert_eq!(
            SpawnedBy::from_json_str(mirror.spawned_by.as_deref().unwrap()),
            Some(SpawnedBy::Conversation {
                parent_session_id: w.parent.id,
                tool_use_id: Some(CALL.into()),
            })
        );
        assert_eq!(mirror.provider_id, w.parent.provider_id);
        assert_eq!(mirror.model, w.parent.model);
        assert_eq!(mirror.cwd, w.parent.cwd);
        assert_eq!(mirror.project_slug, w.parent.project_slug);
        assert_eq!(mirror.owner, w.parent.owner);

        let events = thread(w.graph.as_ref(), mirror.id).await;
        let kinds: Vec<&str> = events.iter().map(ChatEvent::event_type).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "assistant_text",
                "tool_use",
                "tool_result",
                "assistant_text",
                "result"
            ],
            "{events:?}"
        );
        assert!(
            matches!(&events[0], ChatEvent::UserMessage { content } if content == "Look for the spec of X")
        );
        assert!(
            events.iter().all(|e| parent_of(e).is_none()),
            "the sub-agent shows at the mirror's top level: {events:?}"
        );
        match events.last().unwrap() {
            ChatEvent::Result {
                result_text,
                is_error,
                ..
            } => {
                assert_eq!(result_text.as_deref(), Some("The spec is in a.md"));
                assert!(!is_error);
            }
            other => panic!("{other:?}"),
        }

        let crud = w.emitter.of(&mirror.id.to_string());
        assert_eq!(crud.len(), 3, "{crud:?}");
        assert_eq!(crud[0].0, CrudAction::Created);
        assert_eq!(
            crud[0].1,
            serde_json::json!({
                "project_slug": "proj",
                "spawned_by": {
                    "type": "conversation",
                    "parent_session_id": w.parent.id.to_string(),
                    "tool_use_id": CALL,
                },
                "mirror_of": "claude_code",
            })
        );
        assert_eq!(
            crud[1],
            (
                CrudAction::Updated,
                serde_json::json!({"is_streaming": true})
            )
        );
        assert_eq!(
            crud[2],
            (
                CrudAction::Updated,
                serde_json::json!({"is_streaming": false})
            )
        );
        assert!(!w.registry.is_open(&mirror.id.to_string()), "closed");
        // Nothing of the parent was written by the tap: it only observes.
        assert!(thread(w.graph.as_ref(), w.parent.id).await.is_empty());
    }

    /// A sub-agent's own Task call is not mirrored on its own: its events go into the
    /// first-level mirror, nested under that call there.
    #[tokio::test]
    async fn a_nested_task_call_stays_in_its_first_level_mirror() {
        let w = world().await;
        let tap = w.tap();
        let events = [
            call(
                CALL,
                "Task",
                serde_json::json!({"description": "outer", "prompt": "p"}),
                None,
            ),
            call(
                "toolu_inner",
                "Task",
                serde_json::json!({"description": "inner", "prompt": "q"}),
                Some(CALL),
            ),
            said("deep", Some("toolu_inner")),
            answered("toolu_inner", "inner done", Some(CALL)),
            answered(CALL, "outer done", None),
        ];
        for event in &events {
            tap.observe(event);
        }
        let mirrors = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await;
        assert_eq!(mirrors.len(), 1, "only the first level is mirrored");
        let events = thread(w.graph.as_ref(), mirrors[0].id).await;
        assert!(
            events.iter().any(|e| matches!(e,
                ChatEvent::AssistantText { content, parent_tool_use_id: Some(p) }
                    if content == "deep" && p == "toolu_inner")),
            "the deeper event keeps its parent, under the nested call: {events:?}"
        );
    }

    /// Never two mirrors for one call: the same events seen again — by a new tap of the
    /// same process (a resume), then by a new process (a restart: an empty registry) — write
    /// no second mirror, no second `created`, and nothing after the mirror's result.
    #[tokio::test]
    async fn a_replay_never_writes_a_second_mirror() {
        let w = world().await;
        let first = w.tap();
        for event in task_turn() {
            first.observe(&event);
        }
        let mirror = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await[0].clone();
        let stored = thread(w.graph.as_ref(), mirror.id).await.len();

        let again = w.tap();
        for event in task_turn() {
            again.observe(&event);
        }
        let restarted = World {
            graph: w.graph.clone(),
            emitter: w.emitter.clone(),
            registry: Arc::default(),
            parent: w.parent.clone(),
        };
        let fresh = restarted.tap();
        for event in task_turn() {
            fresh.observe(&event);
        }
        drop((again, fresh));
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert_eq!(mirrors_of(w.graph.as_ref(), w.parent.id).await.len(), 1);
        assert_eq!(thread(w.graph.as_ref(), mirror.id).await.len(), stored);
        let created = w
            .emitter
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.action == CrudAction::Created)
            .count();
        assert_eq!(created, 1);
    }

    /// The parent's turn ends (or the session closes) before the call's result: the mirror
    /// is closed anyway, as cut, and stops streaming.
    #[tokio::test]
    async fn the_end_of_the_parent_turn_closes_an_open_mirror() {
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            CALL,
            "Agent",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        tap.observe(&said("working", Some(CALL)));
        tap.observe(&ChatEvent::StreamingStatus {
            is_streaming: false,
        });
        let mirror = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await[0].clone();
        match thread(w.graph.as_ref(), mirror.id).await.last().unwrap() {
            ChatEvent::Result {
                is_error,
                stop_reason,
                ..
            } => {
                assert!(is_error);
                assert_eq!(stop_reason.as_deref(), Some("interrupted"));
            }
            other => panic!("{other:?}"),
        }
        let crud = w.emitter.of(&mirror.id.to_string());
        assert_eq!(
            crud.last().unwrap(),
            &(
                CrudAction::Updated,
                serde_json::json!({"is_streaming": false})
            )
        );

        // The session closes (the tap is dropped) while a second call runs.
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            CALL,
            "Task",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        tap.observe(&said("working", Some(CALL)));
        drop(tap);
        closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await;
    }

    /// Only a sub-agent call is mirrored: events under another tool, or under a call this
    /// tap never saw, write nothing.
    #[tokio::test]
    async fn activity_under_another_tool_is_never_mirrored() {
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            "toolu_bash",
            "Bash",
            serde_json::json!({"command": "ls"}),
            None,
        ));
        tap.observe(&said("not a sub-agent", Some("toolu_bash")));
        tap.observe(&said("unknown call", Some("toolu_unknown")));
        tap.observe(&answered("toolu_bash", "ok", None));
        drop(tap);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(mirrors_of(w.graph.as_ref(), w.parent.id).await.is_empty());
        assert!(w.emitter.events.lock().unwrap().is_empty());
    }

    /// The open mirror is followed live and counts as streaming until its result.
    #[tokio::test]
    async fn an_open_mirror_is_live_until_its_result() {
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            CALL,
            "Task",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        tap.observe(&said("working", Some(CALL)));
        let mut id = None;
        for _ in 0..200 {
            if let Some(m) = mirrors_of(w.graph.as_ref(), w.parent.id).await.first() {
                id = Some(m.id.to_string());
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let id = id.expect("a mirror");
        for _ in 0..200 {
            if w.registry.is_open(&id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let mut rx = w.registry.subscribe(&id).expect("followed live");
        assert!(w.registry.open_ids().iter().any(|o| o.to_string() == id));
        tap.observe(&said("more", Some(CALL)));
        let seen = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(seen, ChatEvent::AssistantText { ref content, parent_tool_use_id: None } if content == "more")
        );
        tap.observe(&answered(CALL, "done", None));
        closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await;
        assert!(!w.registry.is_open(&id));
    }
}

/// What the tests of the mirrors share (`mirror::tests`, the e2e tests of both engines).
#[cfg(test)]
pub(crate) mod test_support {
    use std::time::Duration;

    use super::*;
    use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};

    /// Keeps every CRUD event emitted.
    #[derive(Default)]
    pub(crate) struct RecordingEmitter {
        pub events: std::sync::Mutex<Vec<CrudEvent>>,
    }

    impl EventEmitter for RecordingEmitter {
        fn emit(&self, event: CrudEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    impl RecordingEmitter {
        /// The CRUD events of the chat session `id`: (action, payload).
        pub(crate) fn of(&self, id: &str) -> Vec<(CrudAction, serde_json::Value)> {
            self.events
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.entity_type == EntityType::ChatSession && e.entity_id == id)
                .map(|e| (e.action.clone(), e.payload.clone()))
                .collect()
        }
    }

    /// The mirrors the graph holds for `parent`.
    pub(crate) async fn mirrors_of(graph: &dyn GraphStore, parent: Uuid) -> Vec<ChatSessionNode> {
        graph
            .get_session_children(parent)
            .await
            .unwrap()
            .into_iter()
            .filter(|c| c.mirror_of.is_some())
            .collect()
    }

    /// The stored thread of `id`, in order.
    pub(crate) async fn thread(graph: &dyn GraphStore, id: Uuid) -> Vec<ChatEvent> {
        let mut records = graph.get_chat_events(id, 0, 1000).await.unwrap();
        records.sort_by_key(|r| r.seq);
        records
            .iter()
            .map(|r| serde_json::from_str(&r.data).unwrap())
            .collect()
    }

    /// Waits (10 s at most) until `parent` has `n` mirrors whose threads end on a `result`.
    pub(crate) async fn closed_mirrors(
        graph: &dyn GraphStore,
        parent: Uuid,
        n: usize,
    ) -> Vec<ChatSessionNode> {
        for _ in 0..400 {
            let mirrors = mirrors_of(graph, parent).await;
            let mut closed = 0;
            for m in &mirrors {
                if matches!(
                    thread(graph, m.id).await.last(),
                    Some(ChatEvent::Result { .. })
                ) {
                    closed += 1;
                }
            }
            if mirrors.len() >= n && closed >= n {
                return mirrors;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("{n} closed mirror(s) of {parent} expected");
    }
}
