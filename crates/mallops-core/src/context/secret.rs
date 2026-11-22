//! Secret isolation and resolution contracts.
//!
//! Secrets are resolved on demand through [`SecretProvider`] into an opaque
//! [`Secret`] whose [`fmt::Debug`] and [`fmt::Display`] representations are
//! always redacted (`"***"`) and which intentionally does not implement
//! `serde::Serialize` (see ADR-0007 D4, D7 and `docs/SPEC.md` §4.1).

use std::collections::BTreeMap;
use std::env::VarError;
use std::fmt;

use crate::Error;

/// Redacted placeholder rendered by [`Secret`]'s [`fmt::Debug`] and [`fmt::Display`].
const REDACTED: &str = "***";

/// Environment variable prefix read by [`EnvSecretProvider`].
const ENV_PREFIX: &str = "MALLOPS_SECRET_";

/// An opaque secret value that redacts itself in logs and formatting output.
///
/// `Secret` prints `"***"` for both [`fmt::Debug`] and [`fmt::Display`],
/// requires an explicit [`expose`](Self::expose) call to inspect the inner
/// value, and deliberately does not implement `serde::Serialize` so it cannot
/// accidentally be written to `config.json`, cache keys, or lineage sinks.
///
/// ```compile_fail
/// use mallops_core::Secret;
///
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<Secret>();
/// ```
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    /// Wraps a raw secret string in an opaque [`Secret`].
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the underlying secret value.
    ///
    /// Callers must ensure the returned slice is never written to logs,
    /// `Context::config`, cache keys, or lineage events.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// Resolves named secrets on demand for [`Context::secret`](crate::Context::secret).
///
/// Implementations must return [`Error::Secret`] on failure and must never
/// include secret values in error messages.
pub trait SecretProvider: Send + Sync {
    /// Resolves the secret identified by `name`.
    fn get(&self, name: &str) -> Result<Secret, Error>;
}

/// Default [`SecretProvider`] reading environment variables prefixed with `MALLOPS_SECRET_`.
///
/// A lookup for `"API_KEY"` reads the `MALLOPS_SECRET_API_KEY` environment variable.
#[derive(Debug, Default, Clone, Copy)]
pub struct EnvSecretProvider;

impl EnvSecretProvider {
    /// Prefix prepended to secret names when looking up environment variables.
    pub const PREFIX: &'static str = ENV_PREFIX;

    /// Creates a new [`EnvSecretProvider`].
    pub fn new() -> Self {
        Self
    }
}

impl SecretProvider for EnvSecretProvider {
    fn get(&self, name: &str) -> Result<Secret, Error> {
        if name.is_empty() {
            return Err(Error::Secret("secret name must not be empty".to_string()));
        }
        let env_key = format!("{ENV_PREFIX}{name}");
        match std::env::var(&env_key) {
            Ok(value) => Ok(Secret::new(value)),
            Err(VarError::NotPresent) => Err(Error::Secret(format!(
                "secret `{name}` not found in environment variable `{env_key}`"
            ))),
            Err(VarError::NotUnicode(_)) => Err(Error::Secret(format!(
                "secret `{name}` in environment variable `{env_key}` is not valid UTF-8"
            ))),
        }
    }
}

/// In-memory [`SecretProvider`] intended for unit and integration tests.
///
/// Its [`fmt::Debug`] implementation redacts all stored values.
#[derive(Default, Clone)]
pub struct MemorySecretProvider {
    secrets: BTreeMap<String, String>,
}

impl MemorySecretProvider {
    /// Creates an empty [`MemorySecretProvider`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a secret and returns `self` for fluent test setup.
    pub fn with_secret(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.insert(name, value);
        self
    }

    /// Inserts or replaces a named secret in memory.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.secrets.insert(name.into(), value.into());
    }
}

impl fmt::Debug for MemorySecretProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for key in self.secrets.keys() {
            map.entry(key, &REDACTED);
        }
        map.finish()
    }
}

impl SecretProvider for MemorySecretProvider {
    fn get(&self, name: &str) -> Result<Secret, Error> {
        if name.is_empty() {
            return Err(Error::Secret("secret name must not be empty".to_string()));
        }
        self.secrets
            .get(name)
            .map(|value| Secret::new(value.clone()))
            .ok_or_else(|| Error::Secret(format!("secret `{name}` not found")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHILD_SENTINEL_ENV: &str = "MALLOPS_TEST_ENV_SECRET_CHILD";
    const TEST_SECRET_VALUE: &str = "s3cr3t-token-do-not-leak";

    #[test]
    fn secret_redacts_debug_and_display_and_exposes_value() {
        let secret = Secret::new(TEST_SECRET_VALUE);

        assert_eq!(format!("{secret}"), "***");
        assert_eq!(format!("{secret:?}"), "***");
        assert!(!format!("{secret}").contains(TEST_SECRET_VALUE));
        assert!(!format!("{secret:?}").contains(TEST_SECRET_VALUE));
        assert_eq!(secret.expose(), TEST_SECRET_VALUE);
    }

    #[test]
    fn memory_secret_provider_resolves_and_redacts_debug() {
        let provider = MemorySecretProvider::new().with_secret("DB_PASS", TEST_SECRET_VALUE);

        let resolved = provider.get("DB_PASS").expect("secret should resolve");
        assert_eq!(resolved.expose(), TEST_SECRET_VALUE);

        let debug_repr = format!("{provider:?}");
        assert!(debug_repr.contains("DB_PASS"));
        assert!(debug_repr.contains("***"));
        assert!(!debug_repr.contains(TEST_SECRET_VALUE));

        let missing = provider.get("UNKNOWN").expect_err("missing secret");
        assert!(matches!(missing, Error::Secret(_)));
        assert!(!missing.to_string().contains(TEST_SECRET_VALUE));

        let empty = provider.get("").expect_err("empty name");
        assert!(matches!(empty, Error::Secret(_)));
    }

    #[test]
    fn env_secret_provider_rejects_empty_name() {
        let provider = EnvSecretProvider::new();
        let err = provider.get("").expect_err("empty name must fail");
        assert!(matches!(err, Error::Secret(_)));
    }

    #[test]
    fn env_secret_provider_reads_mallops_secret_prefix() {
        let exe = std::env::current_exe().expect("resolve current test binary");
        let output = std::process::Command::new(exe)
            .arg("--exact")
            .arg("context::secret::tests::env_secret_provider_child")
            .arg("--nocapture")
            .env(CHILD_SENTINEL_ENV, "1")
            .env("MALLOPS_SECRET_API_KEY", TEST_SECRET_VALUE)
            .env_remove("MALLOPS_SECRET_MISSING_KEY")
            .output()
            .expect("spawn child test process");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "child test failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        assert!(
            stdout.contains("1 passed"),
            "expected child test to execute:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[test]
    fn env_secret_provider_child() {
        if std::env::var_os(CHILD_SENTINEL_ENV).is_none() {
            return;
        }

        let provider = EnvSecretProvider::new();
        let secret = provider
            .get("API_KEY")
            .expect("MALLOPS_SECRET_API_KEY should resolve");
        assert_eq!(secret.expose(), TEST_SECRET_VALUE);
        assert_eq!(format!("{secret}"), "***");
        assert_eq!(format!("{secret:?}"), "***");

        let err = provider
            .get("MISSING_KEY")
            .expect_err("missing env secret must fail");
        assert!(matches!(err, Error::Secret(_)));
        let msg = err.to_string();
        assert!(msg.contains("MALLOPS_SECRET_MISSING_KEY"));
        assert!(!msg.contains(TEST_SECRET_VALUE));
    }
}
