//! `${env:NAME}` interpolation and `${secret:NAME}` reference validation.

use serde_json::Value;

use crate::Error;
use crate::context::secret::check_name as check_secret_name;

const ENV_PREFIX: &str = "env:";
const SECRET_PREFIX: &str = "secret:";

/// Recursively interpolates `${env:NAME}` references and validates `${secret:NAME}`
/// references in all string values of `value`.
///
/// Object keys and non-string primitives are left untouched. `${secret:NAME}` references
/// are validated with [`crate::context::secret::check_name`] and preserved literally;
/// secret values are never resolved here. Replaced text is not re-scanned.
pub(crate) fn interpolate(
    value: &mut Value,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<(), Error> {
    match value {
        Value::String(s) => interpolate_str(s, env),
        Value::Array(items) => {
            for item in items {
                interpolate(item, env)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for v in map.values_mut() {
                interpolate(v, env)?;
            }
            Ok(())
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(()),
    }
}

fn is_valid_env_name(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn interpolate_str(input: &mut String, env: &dyn Fn(&str) -> Option<String>) -> Result<(), Error> {
    let s = input.as_str();
    let mut cursor = 0;
    let mut out: Option<String> = None;

    while let Some(rel_start) = s[cursor..].find("${") {
        let start = cursor + rel_start;
        let after_open = start + 2;
        let rest = &s[after_open..];

        if let Some(after_prefix) = rest.strip_prefix(ENV_PREFIX) {
            let name_start = after_open + ENV_PREFIX.len();
            let Some(rel_end) = after_prefix.find('}') else {
                return Err(Error::Config(
                    "unterminated `${env:...}` reference: missing `}`".to_string(),
                ));
            };
            let end = name_start + rel_end;
            let name = &s[name_start..end];
            if !is_valid_env_name(name) {
                return Err(Error::Config(format!(
                    "invalid environment variable name `{name}` in `${{env:{name}}}`"
                )));
            }
            let replacement = env(name)
                .ok_or_else(|| Error::Config(format!("missing environment variable `{name}`")))?;

            let buf = match out.as_mut() {
                Some(buf) => {
                    buf.push_str(&s[cursor..start]);
                    buf
                }
                None => {
                    let mut buf = String::with_capacity(s.len());
                    buf.push_str(&s[..start]);
                    out.insert(buf)
                }
            };
            buf.push_str(&replacement);
            cursor = end + 1;
        } else if let Some(after_prefix) = rest.strip_prefix(SECRET_PREFIX) {
            let name_start = after_open + SECRET_PREFIX.len();
            let Some(rel_end) = after_prefix.find('}') else {
                return Err(Error::Config(
                    "unterminated `${secret:...}` reference: missing `}`".to_string(),
                ));
            };
            let end = name_start + rel_end;
            let name = &s[name_start..end];
            check_secret_name(name)?;

            if let Some(buf) = out.as_mut() {
                buf.push_str(&s[cursor..=end]);
            }
            cursor = end + 1;
        } else {
            if let Some(buf) = out.as_mut() {
                buf.push_str(&s[cursor..after_open]);
            }
            cursor = after_open;
        }
    }

    if let Some(mut buf) = out {
        buf.push_str(&s[cursor..]);
        *input = buf;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;

    fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: BTreeMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |name: &str| map.get(name).cloned()
    }

    #[test]
    fn simple_env_substitution() {
        let env = env_from(&[("HOST", "example.com"), ("PORT", "8080")]);

        let mut exact = json!("${env:HOST}");
        interpolate(&mut exact, &env).expect("exact substitution");
        assert_eq!(exact, json!("example.com"));

        let mut embedded = json!("https://${env:HOST}/v1");
        interpolate(&mut embedded, &env).expect("embedded substitution");
        assert_eq!(embedded, json!("https://example.com/v1"));
    }

    #[test]
    fn multiple_occurrences_and_charset() {
        let env = env_from(&[("aZ_09", "alpha"), ("B", "beta")]);

        let mut val = json!("${env:aZ_09}-${env:B}-${env:aZ_09}");
        interpolate(&mut val, &env).expect("multiple substitutions");
        assert_eq!(val, json!("alpha-beta-alpha"));
    }

    #[test]
    fn nested_values_and_unmodified_keys_and_primitives() {
        let env = env_from(&[("DIR", "/data"), ("TAG", "v1")]);

        let mut val = json!({
            "${env:UNRESOLVED_KEY}": {
                "path": "${env:DIR}/file.csv",
                "items": ["è-${env:TAG}", 42, true, null, ["${env:DIR}"]]
            },
            "count": 3,
            "flag": false,
            "empty": null
        });

        interpolate(&mut val, &env).expect("nested interpolation");

        assert_eq!(
            val,
            json!({
                "${env:UNRESOLVED_KEY}": {
                    "path": "/data/file.csv",
                    "items": ["è-v1", 42, true, null, ["/data"]]
                },
                "count": 3,
                "flag": false,
                "empty": null
            })
        );
    }

    #[test]
    fn missing_env_var_reports_variable_name() {
        let env = env_from(&[("OTHER", "1")]);
        let mut val = json!("url=${env:OPENLINEAGE_URL}");

        let err = interpolate(&mut val, &env).expect_err("missing env var must fail");
        assert!(matches!(err, Error::Config(_)), "unexpected error: {err:?}");
        assert!(
            err.to_string().contains("OPENLINEAGE_URL"),
            "error should name the missing variable: {err}"
        );
    }

    #[test]
    fn secret_reference_stays_symbolic_and_never_calls_env() {
        let calls = AtomicUsize::new(0);
        let env = |name: &str| {
            calls.fetch_add(1, Ordering::Relaxed);
            if name == "HOST" {
                Some("api.local".to_string())
            } else {
                Some("SHOULD_NEVER_BE_READ".to_string())
            }
        };

        let mut val = json!({
            "token": "${secret:API_KEY}",
            "header": "Bearer ${secret:API_KEY}",
            "mixed": "https://${env:HOST}?k=${secret:API_KEY}&m=${env:HOST}"
        });

        interpolate(&mut val, &env).expect("secret reference should stay symbolic");

        assert_eq!(
            val,
            json!({
                "token": "${secret:API_KEY}",
                "header": "Bearer ${secret:API_KEY}",
                "mixed": "https://api.local?k=${secret:API_KEY}&m=api.local"
            })
        );
        // Only the two `${env:HOST}` lookups should have invoked `env`.
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn invalid_names_return_error() {
        let env = env_from(&[]);

        for bad in [
            "${env:}",
            "${env:FOO-BAR}",
            "${env:FOO.BAR}",
            "${env:FOO BAR}",
            "${env:FOO$}",
            "${env:café}",
        ] {
            let mut val = json!(bad);
            let err = interpolate(&mut val, &env).expect_err(bad);
            assert!(
                matches!(err, Error::Config(_)),
                "expected Error::Config for {bad}, got {err:?}"
            );
        }

        let mut empty_secret = json!("${secret:}");
        let err = interpolate(&mut empty_secret, &env).expect_err("empty secret name must fail");
        assert!(
            matches!(err, Error::Secret(_)),
            "expected Error::Secret for empty secret name, got {err:?}"
        );
    }

    #[test]
    fn missing_closing_brace_returns_config_error() {
        let env = env_from(&[("FOO", "bar")]);

        for bad in [
            "${env:FOO",
            "${env:",
            "prefix ${env:FOO",
            "${secret:API_KEY",
            "${secret:",
            "prefix ${secret:API_KEY",
        ] {
            let mut val = json!(bad);
            let err = interpolate(&mut val, &env).expect_err(bad);
            assert!(
                matches!(err, Error::Config(_)),
                "expected Error::Config for {bad}, got {err:?}"
            );
        }
    }

    #[test]
    fn other_dollar_braces_remain_literal() {
        let env = env_from(&[("SUB", "data")]);

        let mut val = json!({
            "shell": "${HOME}/bin/${USER:-root}",
            "empty_braces": "${}",
            "unclosed_other": "${HOME",
            "almost_env": "${env_var}",
            "mixed": "${HOME}/${env:SUB}/${OTHER}"
        });

        interpolate(&mut val, &env).expect("non-env/secret placeholders stay literal");

        assert_eq!(
            val,
            json!({
                "shell": "${HOME}/bin/${USER:-root}",
                "empty_braces": "${}",
                "unclosed_other": "${HOME",
                "almost_env": "${env_var}",
                "mixed": "${HOME}/data/${OTHER}"
            })
        );
    }

    #[test]
    fn replaced_text_is_not_rescanned() {
        let env = env_from(&[
            ("A", "${env:B}"),
            ("B", "expanded"),
            ("BAD_SECRET", "${secret:}"),
            ("UNCLOSED", "${env:NO_BRACE"),
        ]);

        let mut val = json!({
            "recursive": "${env:A}",
            "bad_secret": "${env:BAD_SECRET}",
            "unclosed": "${env:UNCLOSED}"
        });

        interpolate(&mut val, &env).expect("replaced text must not be rescanned");

        assert_eq!(
            val,
            json!({
                "recursive": "${env:B}",
                "bad_secret": "${secret:}",
                "unclosed": "${env:NO_BRACE"
            })
        );
    }
}
