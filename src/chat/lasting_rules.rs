//! Lasting approvals ("always", P11b): where they are kept and how they are read back.
//!
//! What an `always` approval covers is decided by the backend's strict comparator
//! (`session_grants::lasting_grant_for`): the IDENTICAL call, for a narrower set of calls
//! than a session grant (no file write, a single simple command without redirection,
//! operator or glob). A lasting rule is NEVER handed to a provider: not to the `allow`
//! list of nexus, not to the Claude Code CLI's settings (`localSettings`,
//! `updatedPermissions`). The provider is answered `once`; the backend answers itself the
//! later requests a rule covers, in any session of the same person on the same project.
//!
//! A rule belongs to a PERSON (the session's `owner`, the `sub` of their token) and a
//! PROJECT (the session's project slug). A session without an owner (the server's own
//! runs, the anonymous user) or without a project keeps no lasting rule: `always` is not
//! offered there.
//!
//! Storage (one file per person, `<root>/<sha256(owner)>.json`), hardened against another
//! local user or a process of the model:
//!
//! - the directory is created `0700`; a directory that is a symbolic link, belongs to
//!   another user or is open to the group or others is not used (nothing read, nothing
//!   written);
//! - a rule file is opened without following a symbolic link (`O_NOFOLLOW`) and read
//!   only when it is a regular file of this user that neither the group nor others may
//!   write; otherwise it is IGNORED (no rule holds);
//! - a write goes to a fresh temporary file of the directory (`O_EXCL`, `0600`), synced,
//!   renamed over the file (atomic), then the directory is synced;
//! - every read-modify-write holds an exclusive lock on `<file>.lock` (`flock`): two
//!   servers sharing the directory never lose each other's rules;
//! - every rule read back goes through the comparator again: a rule edited by hand into
//!   something it would never grant (`mcp__nexus__Bash(*)`, a glob, a wrapper, a write)
//!   holds nothing.
//!
//! Lasting rules live on the machine of the instance that owns the session; another
//! instance (NATS) has its own directory.

use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::session_grants::{AskedCall, Asker, SessionGrant};

/// The directory of the lasting rules under the application directory (where the vault,
/// the identity key and the native transcripts already live).
pub fn default_root() -> PathBuf {
    let app = match dirs::config_dir() {
        #[cfg(target_os = "windows")]
        Some(c) => c.join("ProjectOrchestrator"),
        #[cfg(not(target_os = "windows"))]
        Some(c) => c.join("project-orchestrator"),
        None => dirs::home_dir()
            .map(|h| h.join(".project-orchestrator"))
            .unwrap_or_else(|| PathBuf::from(".project-orchestrator")),
    };
    app.join("permission-rules")
}

/// The largest rule file read back (a bigger one is ignored).
const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// The most rules one person keeps (an add past it is refused).
const MAX_RULES: usize = 500;

/// One lasting rule: the call the person approved "always", where, and when.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LastingRule {
    pub id: Uuid,
    /// The project slug it holds for.
    pub project: String,
    /// Who asked: `claude_code` or `native` (the comparator's [`Asker`]).
    pub asker: String,
    /// The tool's name as the provider gave it (`Bash`, `mcp__nexus__Bash`).
    pub tool: String,
    /// Its stable alias, when the adapter gave one.
    #[serde(default)]
    pub canonical: Option<String>,
    /// The call's input, as approved.
    pub input: Value,
    /// What it covers, as the user is shown it (`Bash: cat README.md`).
    pub rule: String,
    pub created_at: DateTime<Utc>,
    /// The session where it was approved.
    #[serde(default)]
    pub session_id: Option<String>,
}

impl LastingRule {
    /// The call this rule was approved for.
    pub fn call(&self) -> Option<AskedCall> {
        let asker = match self.asker.as_str() {
            "claude_code" => Asker::ClaudeCode,
            "native" => Asker::Native,
            _ => return None,
        };
        Some(AskedCall {
            asker,
            tool: self.tool.clone(),
            canonical: self.canonical.clone(),
            input: self.input.clone(),
        })
    }

    /// The grant this rule stands for, decided again by the comparator
    /// (`read_only_mcp`: the third-party MCP tools declared read-only where it is
    /// checked). `None`: a rule the comparator would not grant today (edited by hand,
    /// or a declaration since withdrawn) holds nothing.
    pub fn grant(&self, read_only_mcp: &[String]) -> Option<SessionGrant> {
        let call = self.call()?;
        let grant = super::session_grants::lasting_grant_with(&call, read_only_mcp)?;
        (grant.describe() == self.rule).then_some(grant)
    }
}

/// The wire name of an [`Asker`] that may keep lasting rules.
pub fn asker_name(asker: Asker) -> Option<&'static str> {
    match asker {
        Asker::ClaudeCode => Some("claude_code"),
        Asker::Native => Some("native"),
        Asker::Codex | Asker::Other => None,
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RuleFile {
    /// The person the file belongs to (checked against the name it was looked up by).
    owner: String,
    rules: Vec<LastingRule>,
}

/// Why a lasting rule could not be kept or removed.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("lasting approvals are not kept on this platform")]
    Unsupported,
    #[error("the directory of the lasting rules is not safe: {0}")]
    UnsafeDir(String),
    #[error("the rule file is not safe: {0}")]
    UnsafeFile(String),
    #[error("too many lasting rules ({MAX_RULES})")]
    Full,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// The lasting rules of every person, in one directory.
#[derive(Debug, Clone)]
pub struct LastingRuleStore {
    root: PathBuf,
}

impl LastingRuleStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn file_name(owner: &str) -> String {
        use sha2::Digest;
        hex::encode(sha2::Sha256::digest(owner.as_bytes()))
    }

    fn file_for(&self, owner: &str) -> PathBuf {
        self.root.join(format!("{}.json", Self::file_name(owner)))
    }

    fn lock_for(&self, owner: &str) -> PathBuf {
        self.root.join(format!("{}.lock", Self::file_name(owner)))
    }

    /// Every rule of `owner` that can be read for sure; nothing when the directory or the
    /// file is not safe (logged), or there is none. Not revalidated: see
    /// [`LastingRule::grant`].
    pub fn rules(&self, owner: &str) -> Vec<LastingRule> {
        match self.read(owner) {
            Ok(file) => file.rules,
            Err(e) => {
                tracing::warn!(error = %e, "lasting permission rules ignored");
                Vec::new()
            }
        }
    }

    fn read(&self, owner: &str) -> Result<RuleFile, StoreError> {
        if !safe::dir_is_safe(&self.root, false)? {
            return Ok(RuleFile::default());
        }
        let Some(text) = safe::read_file(&self.file_for(owner))? else {
            return Ok(RuleFile::default());
        };
        let file: RuleFile = serde_json::from_str(&text)?;
        if file.owner != owner {
            return Err(StoreError::UnsafeFile(
                "the file names another person".to_string(),
            ));
        }
        Ok(file)
    }

    /// Changes `owner`'s rules under the inter-process lock, then writes them atomically.
    /// A file that cannot be read for sure is never overwritten.
    fn update<T>(
        &self,
        owner: &str,
        change: impl FnOnce(&mut Vec<LastingRule>) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        safe::dir_is_safe(&self.root, true)?;
        let _lock = safe::lock(&self.lock_for(owner))?;
        let mut file = self.read(owner)?;
        file.owner = owner.to_string();
        let out = change(&mut file.rules)?;
        safe::write_atomic(
            &self.root,
            &self.file_for(owner),
            &serde_json::to_vec_pretty(&file)?,
        )?;
        Ok(out)
    }

    /// Keeps a rule for `owner`. The same call on the same project is kept once (the
    /// existing rule is returned).
    pub fn add(&self, owner: &str, rule: LastingRule) -> Result<LastingRule, StoreError> {
        self.update(owner, |rules| {
            if let Some(same) = rules.iter().find(|r| {
                r.project == rule.project
                    && r.asker == rule.asker
                    && r.tool == rule.tool
                    && r.rule == rule.rule
            }) {
                return Ok(same.clone());
            }
            if rules.len() >= MAX_RULES {
                return Err(StoreError::Full);
            }
            rules.push(rule.clone());
            Ok(rule)
        })
    }

    /// Removes `owner`'s rule `id`; whether there was one.
    pub fn revoke(&self, owner: &str, id: Uuid) -> Result<bool, StoreError> {
        self.update(owner, |rules| {
            let before = rules.len();
            rules.retain(|r| r.id != id);
            Ok(rules.len() != before)
        })
    }
}

/// Where a session's lasting rules are kept and looked up: its owner, its project.
#[derive(Debug, Clone)]
pub struct LastingScope {
    pub store: LastingRuleStore,
    pub owner: String,
    pub project: String,
}

impl LastingScope {
    /// The scope of a session, `None` when it keeps no lasting rule: no store configured,
    /// no owner, no project.
    pub fn of_session(
        store: Option<&LastingRuleStore>,
        owner: Option<&str>,
        project: Option<&str>,
    ) -> Option<Self> {
        let owner = owner.filter(|o| !o.trim().is_empty())?;
        let project = project.filter(|p| !p.trim().is_empty())?;
        Some(Self {
            store: store?.clone(),
            owner: owner.to_string(),
            project: project.to_string(),
        })
    }

    /// The rule of this person and project that covers `grant` (the grant the comparator
    /// gives the asked call), every stored rule being decided again by the comparator.
    pub fn covering(
        &self,
        asker: Asker,
        grant: &SessionGrant,
        read_only_mcp: &[String],
    ) -> Option<LastingRule> {
        let asker = asker_name(asker)?;
        self.store.rules(&self.owner).into_iter().find(|rule| {
            rule.project == self.project
                && rule.asker == asker
                && rule.grant(read_only_mcp).as_ref() == Some(grant)
        })
    }

    /// Keeps the rule of `call` (whose lasting grant is `grant`) for this person and
    /// project.
    pub fn keep(
        &self,
        call: &AskedCall,
        grant: &SessionGrant,
        session_id: &str,
    ) -> Result<LastingRule, StoreError> {
        let asker = asker_name(call.asker).ok_or(StoreError::Unsupported)?;
        self.store.add(
            &self.owner,
            LastingRule {
                id: Uuid::new_v4(),
                project: self.project.clone(),
                asker: asker.to_string(),
                tool: call.tool.clone(),
                canonical: call.canonical.clone(),
                input: call.input.clone(),
                rule: grant.describe(),
                created_at: Utc::now(),
                session_id: Some(session_id.to_string()),
            },
        )
    }
}

/// Writes the lasting rule of an approved call (off the async workers: the write takes a
/// file lock); whether it was kept. Called once the provider took the answer. A failure is
/// logged: the caller then keeps the grant for the session only, and says so.
pub async fn keep(
    place: Option<LastingScope>,
    call: &AskedCall,
    grant: &SessionGrant,
    session_id: &str,
) -> bool {
    let Some(place) = place else {
        return false;
    };
    let (call, grant, session_id) = (call.clone(), grant.clone(), session_id.to_string());
    let kept = tokio::task::spawn_blocking(move || place.keep(&call, &grant, &session_id)).await;
    match kept {
        Ok(Ok(rule)) => {
            tracing::info!(rule = %rule.rule, project = %rule.project, "lasting permission rule kept");
            true
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "lasting permission rule NOT kept: granted for the session only");
            false
        }
        Err(e) => {
            tracing::warn!(error = %e, "lasting permission rule NOT kept: granted for the session only");
            false
        }
    }
}

#[cfg(unix)]
mod safe {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::io::Read;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};

    fn euid() -> u32 {
        // SAFETY: geteuid has no precondition and cannot fail.
        unsafe { libc::geteuid() }
    }

    /// Whether `dir` is a directory of this user that nobody else may enter (not a
    /// symbolic link). Missing: created `0700` when `create`, else `false`. Present but
    /// unsafe: an error.
    pub(super) fn dir_is_safe(dir: &Path, create: bool) -> Result<bool, StoreError> {
        match std::fs::symlink_metadata(dir) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if !create {
                    return Ok(false);
                }
                if let Some(parent) = dir.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                match std::fs::DirBuilder::new().mode(0o700).create(dir) {
                    Ok(()) => {}
                    // Created meanwhile by another writer: checked below like any.
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(e) => return Err(e.into()),
                }
                dir_is_safe(dir, false)
            }
            Err(e) => Err(e.into()),
            Ok(meta) => {
                if meta.file_type().is_symlink() || !meta.is_dir() {
                    return Err(StoreError::UnsafeDir("not a plain directory".into()));
                }
                if meta.uid() != euid() {
                    return Err(StoreError::UnsafeDir("owned by another user".into()));
                }
                if meta.mode() & 0o077 != 0 {
                    return Err(StoreError::UnsafeDir(format!(
                        "mode {:o}, expected 700",
                        meta.mode() & 0o777
                    )));
                }
                Ok(true)
            }
        }
    }

    /// Checks an opened file: a regular file of this user that neither the group nor
    /// others may write, with one name.
    fn file_is_safe(file: &File) -> Result<(), StoreError> {
        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(StoreError::UnsafeFile("not a regular file".into()));
        }
        if meta.uid() != euid() {
            return Err(StoreError::UnsafeFile("owned by another user".into()));
        }
        if meta.mode() & 0o022 != 0 {
            return Err(StoreError::UnsafeFile(format!(
                "mode {:o}: writable by others",
                meta.mode() & 0o777
            )));
        }
        if meta.nlink() > 1 {
            return Err(StoreError::UnsafeFile("more than one name".into()));
        }
        Ok(())
    }

    /// The text of `path`, never through a symbolic link, once it is checked; `None` when
    /// missing.
    pub(super) fn read_file(path: &Path) -> Result<Option<String>, StoreError> {
        let file = match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
        {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        file_is_safe(&file)?;
        let mut text = String::new();
        file.take(MAX_FILE_BYTES + 1).read_to_string(&mut text)?;
        if text.len() as u64 > MAX_FILE_BYTES {
            return Err(StoreError::UnsafeFile("too large".into()));
        }
        Ok(Some(text))
    }

    /// An exclusive lock on `path` (created `0600`, never through a symbolic link), held
    /// until the returned file is dropped. Advisory (`flock`): every writer takes it, in
    /// this process or another.
    pub(super) fn lock(path: &Path) -> Result<File, StoreError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?;
        file_is_safe(&file)?;
        file.lock()?;
        Ok(file)
    }

    /// Writes `bytes` to `path` atomically: a fresh temporary file of `dir` (`O_EXCL`,
    /// `0600`), synced, renamed over `path`, then `dir` synced.
    pub(super) fn write_atomic(dir: &Path, path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
        let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
        tmp.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))?;
        tmp.write_all(bytes)?;
        tmp.as_file().sync_all()?;
        tmp.persist(path).map_err(|e| StoreError::Io(e.error))?;
        File::open(dir)?.sync_all()?;
        Ok(())
    }
}

#[cfg(not(unix))]
mod safe {
    use super::*;

    pub(super) fn dir_is_safe(_dir: &Path, _create: bool) -> Result<bool, StoreError> {
        Err(StoreError::Unsupported)
    }
    pub(super) fn read_file(_path: &Path) -> Result<Option<String>, StoreError> {
        Err(StoreError::Unsupported)
    }
    pub(super) fn lock(_path: &Path) -> Result<std::fs::File, StoreError> {
        Err(StoreError::Unsupported)
    }
    pub(super) fn write_atomic(_dir: &Path, _path: &Path, _bytes: &[u8]) -> Result<(), StoreError> {
        Err(StoreError::Unsupported)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::PermissionsExt;

    fn store() -> (tempfile::TempDir, LastingRuleStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = LastingRuleStore::new(dir.path().join("rules"));
        (dir, store)
    }

    fn scope(store: &LastingRuleStore, owner: &str, project: &str) -> LastingScope {
        LastingScope::of_session(Some(store), Some(owner), Some(project)).unwrap()
    }

    fn bash(command: &str) -> AskedCall {
        AskedCall {
            asker: Asker::ClaudeCode,
            tool: "Bash".into(),
            canonical: None,
            input: json!({ "command": command }),
        }
    }

    fn keep(scope: &LastingScope, call: &AskedCall) -> LastingRule {
        let grant = super::super::session_grants::lasting_grant_for(call).expect("lasting");
        scope.keep(call, &grant, "s1").unwrap()
    }

    fn covered(scope: &LastingScope, call: &AskedCall) -> bool {
        super::super::session_grants::lasting_grant_for(call)
            .is_some_and(|grant| scope.covering(call.asker, &grant, &[]).is_some())
    }

    #[test]
    fn a_rule_holds_for_its_person_and_project_only() {
        let (_d, store) = store();
        let alice = scope(&store, "alice", "proj");
        keep(&alice, &bash("cat README.md"));
        assert!(covered(&alice, &bash("cat README.md")));
        assert!(!covered(
            &scope(&store, "bob", "proj"),
            &bash("cat README.md")
        ));
        assert!(!covered(
            &scope(&store, "alice", "other"),
            &bash("cat README.md")
        ));
        // No owner or no project: no scope at all.
        assert!(LastingScope::of_session(Some(&store), None, Some("proj")).is_none());
        assert!(LastingScope::of_session(Some(&store), Some("alice"), None).is_none());
        assert!(LastingScope::of_session(None, Some("alice"), Some("proj")).is_none());
    }

    #[test]
    fn the_directory_is_private_and_the_file_is_written_atomically() {
        let (_d, store) = store();
        let alice = scope(&store, "alice", "proj");
        keep(&alice, &bash("cat README.md"));
        let dir_mode = std::fs::metadata(store.root())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(dir_mode & 0o777, 0o700);
        let file = store.file_for("alice");
        let file_mode = std::fs::metadata(&file).unwrap().permissions().mode();
        assert_eq!(file_mode & 0o077, 0, "{file_mode:o}");
        // Nothing left behind but the rule file and its lock.
        let mut names: Vec<String> = std::fs::read_dir(store.root())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names
            .iter()
            .all(|n| n.ends_with(".json") || n.ends_with(".lock")));
    }

    #[test]
    fn a_rule_file_others_may_write_is_ignored() {
        let (_d, store) = store();
        let alice = scope(&store, "alice", "proj");
        keep(&alice, &bash("cat README.md"));
        let file = store.file_for("alice");
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o666)).unwrap();
        assert!(
            !covered(&alice, &bash("cat README.md")),
            "group/other writable"
        );
        // ...and never overwritten either.
        let grant = super::super::session_grants::lasting_grant_for(&bash("ls")).unwrap();
        assert!(alice.keep(&bash("ls"), &grant, "s1").is_err());
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(covered(&alice, &bash("cat README.md")));
    }

    #[test]
    fn a_directory_others_may_enter_or_a_symlinked_one_is_not_used() {
        let (d, store) = store();
        let alice = scope(&store, "alice", "proj");
        keep(&alice, &bash("cat README.md"));
        std::fs::set_permissions(store.root(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(!covered(&alice, &bash("cat README.md")));
        std::fs::set_permissions(store.root(), std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(covered(&alice, &bash("cat README.md")));

        // A rule file that is a symbolic link (to a file the model could write) is not read.
        let elsewhere = d.path().join("planted.json");
        std::fs::write(&elsewhere, std::fs::read(store.file_for("alice")).unwrap()).unwrap();
        std::fs::set_permissions(&elsewhere, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::remove_file(store.file_for("alice")).unwrap();
        std::os::unix::fs::symlink(&elsewhere, store.file_for("alice")).unwrap();
        assert!(!covered(&alice, &bash("cat README.md")), "symlink followed");
    }

    #[test]
    fn a_rule_edited_by_hand_into_something_wider_holds_nothing() {
        let (_d, store) = store();
        let alice = scope(&store, "alice", "proj");
        let kept = keep(&alice, &bash("cat README.md"));
        let tampered = |edit: &dyn Fn(&mut LastingRule)| {
            let mut rule = kept.clone();
            edit(&mut rule);
            let file = RuleFile {
                owner: "alice".into(),
                rules: vec![rule],
            };
            super::safe::write_atomic(
                store.root(),
                &store.file_for("alice"),
                &serde_json::to_vec(&file).unwrap(),
            )
            .unwrap();
        };
        // A provider-style pattern: never a pattern, never covers anything.
        tampered(&|r| {
            r.asker = "native".into();
            r.tool = "mcp__nexus__Bash(*)".into();
            r.canonical = Some("Bash".into());
            r.input = json!({"command": "*"});
            r.rule = "mcp__nexus__Bash(*): *".into();
        });
        for asked in ["*", "rm -rf /", "cat README.md"] {
            let call = AskedCall {
                asker: Asker::Native,
                tool: "mcp__nexus__Bash".into(),
                canonical: Some("Bash".into()),
                input: json!({ "command": asked }),
            };
            assert!(!covered(&alice, &call), "{asked:?}");
        }
        // A command the comparator refuses (redirection), its description left as it was.
        tampered(&|r| r.input = json!({"command": "cat x >> ~/.zshrc"}));
        assert!(!covered(&alice, &bash("cat x >> ~/.zshrc")));
        // ...or with the description matching: still refused by the comparator.
        tampered(&|r| {
            r.input = json!({"command": "ls > out"});
            r.rule = "Bash: ls > out".into();
        });
        assert!(!covered(&alice, &bash("ls > out")));
        // A write tool.
        tampered(&|r| {
            r.tool = "Write".into();
            r.input = json!({"file_path": "/x", "content": "y"});
            r.rule = "Write {\"content\":\"y\",\"file_path\":\"/x\"}".into();
        });
        let write = AskedCall {
            asker: Asker::ClaudeCode,
            tool: "Write".into(),
            canonical: None,
            input: json!({"file_path": "/x", "content": "y"}),
        };
        assert!(super::super::session_grants::lasting_grant_for(&write).is_none());
        // A file that names another person.
        let file = RuleFile {
            owner: "bob".into(),
            rules: vec![kept.clone()],
        };
        super::safe::write_atomic(
            store.root(),
            &store.file_for("alice"),
            &serde_json::to_vec(&file).unwrap(),
        )
        .unwrap();
        assert!(!covered(&alice, &bash("cat README.md")));
    }

    #[test]
    fn a_revoked_rule_no_longer_holds() {
        let (_d, store) = store();
        let alice = scope(&store, "alice", "proj");
        let rule = keep(&alice, &bash("cat README.md"));
        // The same call again is kept once.
        assert_eq!(keep(&alice, &bash("cat README.md")).id, rule.id);
        assert_eq!(store.rules("alice").len(), 1);
        assert!(!store.revoke("bob", rule.id).unwrap());
        assert!(store.revoke("alice", rule.id).unwrap());
        assert!(!covered(&alice, &bash("cat README.md")));
        assert!(!store.revoke("alice", rule.id).unwrap());
    }

    /// Two servers (here: threads with their own store, as two processes would be)
    /// writing the same person's file at once never lose a rule.
    #[test]
    fn two_servers_writing_the_same_file_lose_no_rule() {
        let (_d, store) = store();
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let store = LastingRuleStore::new(store.root().to_path_buf());
                std::thread::spawn(move || {
                    let alice = scope(&store, "alice", "proj");
                    for i in 0..15 {
                        keep(&alice, &bash(&format!("cat file-{t}-{i}.md")));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(store.rules("alice").len(), 8 * 15);
    }
}
