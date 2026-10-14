//! Topic extraction for conversation segments.
//!
//! Deterministic, bilingual (FR/EN) subject labelling of a raw message, used by
//! the segment model to build the `Topic` dimension. Pure keyword matching: a
//! label is emitted iff one of its curated keywords appears in the message.
//! No network, no LLM, no NLP, no hashing of arbitrary words.

/// One normalized subject label and the FR/EN keywords that trigger it.
///
/// Labels are already normalized in the table (lowercase, trimmed, non-empty),
/// and each label appears at most once, so the output of [`topic_labels`] is
/// inherently deduplicated and in a stable order (table order).
struct TopicRule {
    /// Canonical label, already lowercase.
    label: &'static str,
    /// Curated keywords; matched case-insensitively as substrings.
    keywords: &'static [&'static str],
}

const TOPIC_RULES: &[TopicRule] = &[
    TopicRule {
        label: "routing",
        keywords: &["routing", "routage", "route", "router"],
    },
    TopicRule {
        label: "provider",
        keywords: &["provider", "fournisseur"],
    },
    TopicRule {
        label: "model",
        keywords: &["model", "modèle", "llm", "gpt", "claude", "gemini"],
    },
    TopicRule {
        label: "auth",
        keywords: &[
            "auth",
            "authentication",
            "authentification",
            "oauth",
            "jwt",
            "login",
            "sso",
            "token",
        ],
    },
    TopicRule {
        label: "neo4j",
        keywords: &["neo4j", "cypher"],
    },
    TopicRule {
        label: "graph",
        keywords: &["graph", "graphe"],
    },
    TopicRule {
        label: "frontend",
        keywords: &[
            "frontend",
            "front-end",
            "front end",
            "ui/ux",
            "react",
            "angular",
            "svelte",
        ],
    },
    TopicRule {
        label: "packaging",
        keywords: &[
            "packaging",
            "package",
            "empaquetage",
            "paquet",
            "crate",
            "cargo",
        ],
    },
    TopicRule {
        label: "docker",
        keywords: &["docker", "container", "conteneur"],
    },
    TopicRule {
        label: "deployment",
        keywords: &[
            "deployment",
            "deploy",
            "déploiement",
            "déployer",
            "ci/cd",
            "pipeline",
            "release",
        ],
    },
    TopicRule {
        label: "tests",
        keywords: &["test", "testing", "assert"],
    },
    TopicRule {
        label: "docs",
        keywords: &["docs", "documentation", "readme", "document"],
    },
    TopicRule {
        label: "chat",
        keywords: &["chat", "conversation", "websocket", "message", "session"],
    },
    TopicRule {
        label: "mcp",
        keywords: &["mcp", "model context protocol"],
    },
    TopicRule {
        label: "vault",
        keywords: &["vault", "secret", "coffre", "credential"],
    },
    TopicRule {
        label: "database",
        keywords: &[
            "database",
            "base de données",
            "postgres",
            "sql",
            "sqlite",
            "mysql",
        ],
    },
    TopicRule {
        label: "api",
        keywords: &["api", "endpoint", "http", "graphql"],
    },
    TopicRule {
        label: "cache",
        keywords: &["cache", "redis", "memoization", "mémoisation"],
    },
    TopicRule {
        label: "config",
        keywords: &["config", "configuration", "settings", "yaml", "toml"],
    },
    TopicRule {
        label: "performance",
        keywords: &[
            "performance",
            "optimize",
            "optimization",
            "optimisation",
            "benchmark",
            "latency",
        ],
    },
];

/// Returns the normalized subject labels present in `message`.
///
/// A label is emitted iff at least one of its curated keywords appears in the
/// message (case-insensitive substring match). The returned labels are always
/// lowercase, trimmed, non-empty, deduplicated, and in a stable order (the
/// order of the table above, never the order in which keywords appear in the
/// message). A message with no known keyword returns an empty vec: no evidence,
/// no invented topic.
pub fn topic_labels(message: &str) -> Vec<String> {
    let lower = message.to_lowercase();

    TOPIC_RULES
        .iter()
        .filter(|rule| rule.keywords.iter().any(|kw| lower.contains(kw)))
        .map(|rule| rule.label.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn french_and_english_map_to_the_same_label() {
        let fr = topic_labels("le routage du provider et l'authentification");
        let en = topic_labels("the routing of the provider and the authentication");
        assert_eq!(
            fr,
            vec![
                "routing".to_string(),
                "provider".to_string(),
                "auth".to_string()
            ]
        );
        assert_eq!(fr, en);
    }

    #[test]
    fn an_unknown_message_is_empty() {
        assert!(topic_labels("the weather in antibes is lovely").is_empty());
        assert!(topic_labels("").is_empty());
    }

    #[test]
    fn normalization_case_and_trim() {
        // Leading/trailing whitespace and casing must not matter; labels come
        // back lowercase.
        assert_eq!(
            topic_labels("  DOCKER and Neo4J  "),
            vec!["neo4j".to_string(), "docker".to_string()]
        );
    }

    #[test]
    fn dedup_and_stable_order() {
        // Repeated keywords collapse to one label, and the order follows the
        // table (neo4j, docker, tests), not the message order.
        assert_eq!(
            topic_labels("docker docker container and tests tests and neo4j"),
            vec![
                "neo4j".to_string(),
                "docker".to_string(),
                "tests".to_string()
            ]
        );
    }

    #[test]
    fn french_accents_are_handled() {
        assert_eq!(
            topic_labels("le DÉPLOIEMENT docker"),
            vec!["docker".to_string(), "deployment".to_string()]
        );
    }
}
