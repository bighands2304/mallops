//! `${env:NAME}` interpolation and `${secret:NAME}` reference validation.

use serde_json::Value;

use crate::Error;

const ENV_PREFIX: &str = "env:";
const SECRET_PREFIX: &str = "secret:";

/// Recursively interpolates `${env:NAME}` references and validates `${secret:NAME}`
/// references in all string values of `value`.
///
/// Object keys and non-string primitives are left untouched. `${secret:NAME}` references
/// are validated with the same charset as env names and preserved literally;
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

fn is_valid_name(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// Interpolates `${env:NAME}` in `input` in place; `${secret:NAME}` is validated and kept.
pub(crate) fn interpolate_str(
    input: &mut String,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<(), Error> {
    let s = input.as_str();
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0;

    while let Some(rel) = s[cursor..].find("${") {
        let start = cursor + rel;
        let rest = &s[start + 2..];
        let (kind, is_env) = if rest.starts_with(ENV_PREFIX) {
            (ENV_PREFIX, true)
        } else if rest.starts_with(SECRET_PREFIX) {
            (SECRET_PREFIX, false)
        } else {
            // other `${...}` stays literal
            out.push_str(&s[cursor..start + 2]);
            cursor = start + 2;
            continue;
        };
        let after = &rest[kind.len()..];
        let Some(len) = after.find('}') else {
            return Err(Error::Config(format!(
                "unterminated `${{{kind}...}}` reference: missing `}}`"
            )));
        };
        let name = &after[..len];
        if !is_valid_name(name) {
            return Err(Error::Config(format!(
                "invalid name `{name}` in `${{{kind}{name}}}`: expected [A-Za-z0-9_]+"
            )));
        }
        out.push_str(&s[cursor..start]);
        if is_env {
            let value = env(name)
                .ok_or_else(|| Error::Config(format!("missing environment variable `{name}`")))?;
            out.push_str(&value);
        } else {
            out.push_str(&s[start..start + 2 + kind.len() + len + 1]);
        }
        cursor = start + 2 + kind.len() + len + 1;
    }
    out.push_str(&s[cursor..]);
    *input = out;
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
    fn errors_table() {
        let env = env_from(&[("FOO", "bar")]);
        for bad in [
            "${env:}",
            "${env:FOO-BAR}",
            "${env:FOO$}",
            "${env:café}",
            "${secret:}",
            "${secret:A-B}",
            "${secret:A${env:FOO}}",
            "${env:FOO",
            "${env:",
            "prefix ${secret:API_KEY",
            "${secret:",
        ] {
            let err = interpolate(&mut json!(bad), &env).expect_err(bad);
            assert!(matches!(err, Error::Config(_)), "{bad}: {err:?}");
        }
    }

    #[test]
    fn literals_table() {
        let env = env_from(&[("SUB", "data")]);
        for (input, want) in [
            ("${HOME}/bin/${USER:-root}", "${HOME}/bin/${USER:-root}"),
            ("${}", "${}"),
            ("${HOME", "${HOME"),
            ("${env_var}", "${env_var}"),
            ("abc${", "abc${"),
            ("${HOME}/${env:SUB}/${OTHER}", "${HOME}/data/${OTHER}"),
            ("è–日本 ${env:SUB} ✓", "è–日本 data ✓"),
            ("日本${", "日本${"),
        ] {
            let mut v = json!(input);
            interpolate(&mut v, &env).expect(input);
            assert_eq!(v, json!(want), "{input}");
        }
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
