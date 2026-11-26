//! Typed `mallops.yaml` configuration (SPEC §5).
//!
//! `${env:NAME}` values are interpolated at load time; `${secret:NAME}` references stay
//! symbolic and are resolved on demand through `Context::secret` (ADR-0007 D7).

#[cfg(feature = "yaml")]
mod interp;

use std::collections::BTreeMap;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{BlockRef, Error};

pub(crate) fn default_host() -> String {
    "127.0.0.1".to_owned()
}

pub(crate) fn default_port() -> u16 {
    8080
}

/// Artifact store settings (`store:`); defaults to a local store at `./.mallops`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct StoreConfig {
    /// Store kind (`local`, `memory`, ...).
    pub kind: String,
    /// Store location (used by `local`).
    pub path: String,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            kind: "local".to_owned(),
            path: "./.mallops".to_owned(),
        }
    }
}

/// One lineage sink (`- sink: json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct LineageSinkConfig {
    /// Sink name (`json`, `openlineage`, ...).
    pub sink: String,
    /// Sink-specific configuration.
    #[serde(default)]
    pub with: serde_json::Value,
}

/// Lineage settings: `off` or a list of sinks (default `[{ sink: json }]`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LineageConfig {
    /// Lineage disabled (`lineage: off`).
    Off,
    /// Lineage emitted to these sinks.
    Sinks(Vec<LineageSinkConfig>),
}

impl Default for LineageConfig {
    fn default() -> Self {
        Self::Sinks(vec![LineageSinkConfig {
            sink: "json".to_owned(),
            with: serde_json::Value::Null,
        }])
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for LineageConfig {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "LineageConfig".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        concat!(module_path!(), "::LineageConfig").into()
    }

    fn json_schema(g: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "`off` or a list of lineage sinks",
            "anyOf": [
                { "const": "off" },
                { "type": "array", "items": g.subschema_for::<LineageSinkConfig>() }
            ]
        })
    }
}

/// JSON Schema of `mallops.yaml`, for editor completion.
#[cfg(feature = "schema")]
#[must_use]
pub fn pipeline_config_json_schema() -> serde_json::Value {
    schemars::schema_for!(PipelineConfig).into()
}

impl Serialize for LineageConfig {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Off => s.serialize_str("off"),
            Self::Sinks(v) => v.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for LineageConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = LineageConfig;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("`off` or a list of lineage sinks")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                if v == "off" {
                    Ok(LineageConfig::Off)
                } else {
                    Err(E::custom(format!(
                        "invalid lineage `{v}`: expected `off` or a list of sinks"
                    )))
                }
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut sinks = Vec::new();
                while let Some(s) = seq.next_element::<LineageSinkConfig>()? {
                    sinks.push(s);
                }
                Ok(LineageConfig::Sinks(sinks))
            }
        }
        d.deserialize_any(V)
    }
}

/// Deserializes a map rejecting duplicate keys (YAML would otherwise keep the last one).
fn unique_map<'de, D, V>(d: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct M<V>(std::marker::PhantomData<V>);
    impl<'de, V: Deserialize<'de>> serde::de::Visitor<'de> for M<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a map")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((k, v)) = map.next_entry::<String, V>()? {
                if out.contains_key(&k) {
                    return Err(serde::de::Error::custom(format!("duplicate key `{k}`")));
                }
                out.insert(k, v);
            }
            Ok(out)
        }
    }
    d.deserialize_map(M(std::marker::PhantomData))
}

/// Serde adapter reusing `BlockRef`'s `FromStr`, so reserved prefixes keep their error.
mod block_ref {
    use super::*;

    pub fn serialize<S: Serializer>(b: &BlockRef, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(b)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BlockRef, D::Error> {
        let s = String::deserialize(d)?;
        BlockRef::from_str(&s).map_err(|e| match e {
            Error::Config(m) => serde::de::Error::custom(m),
            other => serde::de::Error::custom(other),
        })
    }
}

/// One block of the batch DAG (`blocks.<id>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct BlockConfig {
    /// Block reference (`use:`).
    #[serde(rename = "use", with = "block_ref")]
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub r#use: BlockRef,
    /// Input wiring: port name to `<block>.<output>` (`in:`).
    #[serde(rename = "in", default, deserialize_with = "unique_map")]
    pub r#in: BTreeMap<String, String>,
    /// Block-specific configuration (`with:`).
    #[serde(default)]
    pub with: serde_json::Value,
    /// `cache: false` bypasses caching; `None` means the default.
    #[serde(default)]
    pub cache: Option<bool>,
    /// Optional seed for deterministic blocks.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Serving settings read by `mallops serve` (not a DAG node).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ServeConfig {
    /// Package reference, `<block>.<output>`.
    pub package: String,
    /// Bind host (default `127.0.0.1`).
    #[serde(default = "default_host")]
    pub host: String,
    /// Bind port (default `8080`).
    #[serde(default = "default_port")]
    pub port: u16,
}

/// Typed `mallops.yaml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PipelineConfig {
    /// Schema version.
    pub version: u32,
    /// Pipeline name.
    pub name: String,
    /// Artifact store.
    #[serde(default)]
    pub store: StoreConfig,
    /// Lineage sinks.
    #[serde(default)]
    pub lineage: LineageConfig,
    /// Blocks of the batch DAG, by id.
    #[serde(deserialize_with = "unique_map")]
    pub blocks: BTreeMap<String, BlockConfig>,
    /// Optional serving section.
    #[serde(default)]
    pub serve: Option<ServeConfig>,
}

#[cfg(feature = "yaml")]
impl PipelineConfig {
    /// Parses a `mallops.yaml` document; `origin` names the source in error messages.
    ///
    /// `env` is the lookup for `${env:NAME}` interpolation.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] as `<origin>:<line>:<column>: <message>` on syntax or schema errors.
    pub fn from_yaml_str(
        src: &str,
        origin: &str,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, Error> {
        let mut cfg: Self = yaml_serde::from_str(src).map_err(|e| {
            Error::Config(match e.location() {
                Some(l) => format!("{origin}:{}:{}: {e}", l.line(), l.column()),
                None => format!("{origin}: {e}"),
            })
        })?;
        if cfg.version != 1 {
            return Err(Error::Config(format!(
                "{origin}: unsupported version {} (expected 1)",
                cfg.version
            )));
        }
        cfg.interpolate_env(origin, env)?;
        Ok(cfg)
    }

    /// Interpolates only `blocks.*.with`, lineage sink `with` and `store.path`.
    fn interpolate_env(
        &mut self,
        origin: &str,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Error> {
        let ctx = |what: String, e: Error| match e {
            Error::Config(m) => Error::Config(format!("{origin}: {what}: {m}")),
            other => other,
        };
        interp::interpolate_str(&mut self.store.path, env)
            .map_err(|e| ctx("store.path".into(), e))?;
        if let LineageConfig::Sinks(sinks) = &mut self.lineage {
            for sink in sinks {
                interp::interpolate(&mut sink.with, env)
                    .map_err(|e| ctx(format!("lineage sink `{}`", sink.sink), e))?;
            }
        }
        for (id, b) in &mut self.blocks {
            interp::interpolate(&mut b.with, env).map_err(|e| ctx(format!("block `{id}`"), e))?;
        }
        Ok(())
    }

    /// Reads and parses a `mallops.yaml` file.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] if the file cannot be read or fails [`Self::from_yaml_str`].
    pub fn from_yaml_file(
        path: impl AsRef<std::path::Path>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        let origin = path.display().to_string();
        let src =
            std::fs::read_to_string(path).map_err(|e| Error::Config(format!("{origin}: {e}")))?;
        Self::from_yaml_str(&src, &origin, env)
    }
}

#[cfg(all(test, feature = "yaml"))]
mod tests {
    use super::*;

    const FULL: &str = r#"
version: 1
name: churn-classifier
store:
  kind: local
  path: ./.mallops
lineage:
  - sink: json
  - sink: openlineage
    with: { url: "http://x" }
blocks:
  raw:
    use: std/ingest.csv
    with:
      path: data/customers.csv
  features:
    use: std/preprocess.tabular
    in: { data: raw.dataset }
    seed: 42
    with:
      steps:
        - impute: { strategy: median }
  model:
    use: std/external
    in: { train: features.train }
    cache: false
serve:
  package: package.package
  host: 127.0.0.1
  port: 8080
"#;

    fn parse(src: &str) -> Result<PipelineConfig, Error> {
        PipelineConfig::from_yaml_str(src, "t.yaml", &|_| None)
    }

    fn err(src: &str) -> String {
        match parse(src) {
            Err(Error::Config(m)) => m,
            other => panic!("expected Error::Config, got {other:?}"),
        }
    }

    #[test]
    fn full_spec_example() {
        let c = parse(FULL).unwrap();
        assert_eq!((c.version, c.name.as_str()), (1, "churn-classifier"));
        assert_eq!(c.store, StoreConfig::default());
        assert!(matches!(&c.lineage, LineageConfig::Sinks(v) if v.len() == 2));
        assert_eq!(c.blocks.len(), 3);
        assert_eq!(c.blocks["features"].r#in["data"], "raw.dataset");
        assert_eq!(c.blocks["features"].seed, Some(42));
        assert_eq!(c.blocks["model"].cache, Some(false));
        assert_eq!(c.blocks["raw"].cache, None);
        let s = c.serve.unwrap();
        assert_eq!((s.host.as_str(), s.port), ("127.0.0.1", 8080));
    }

    #[test]
    fn defaults_when_omitted() {
        let c = parse("version: 1\nname: p\nblocks:\n  a: { use: std/ingest.csv }\n").unwrap();
        assert_eq!(c.store, StoreConfig::default());
        assert_eq!(c.lineage, LineageConfig::default());
        assert!(c.serve.is_none());
        let c = parse("version: 1\nname: p\nblocks: {}\nserve: { package: a.b }\n").unwrap();
        let s = c.serve.unwrap();
        assert_eq!((s.host.as_str(), s.port), ("127.0.0.1", 8080));
    }

    #[test]
    fn lineage_off() {
        let c = parse("version: 1\nname: p\nlineage: off\nblocks: {}\n").unwrap();
        assert_eq!(c.lineage, LineageConfig::Off);
        assert!(err("version: 1\nname: p\nlineage: nope\nblocks: {}\n").contains("off"));
    }

    #[test]
    fn unknown_fields_rejected() {
        assert!(err("version: 1\nname: p\nblocks: {}\nbogus: 1\n").contains("bogus"));
    }

    #[test]
    fn reserved_prefixes() {
        for r in ["wasm:a.wasm", "grpc:h:1", "plugin/x"] {
            let e = err(&format!(
                "version: 1\nname: p\nblocks:\n  a: {{ use: \"{r}\" }}\n"
            ));
            assert!(e.contains("not supported in v0.1"), "{e}");
        }
    }

    /// Asserts `t.yaml:<line>:<col>: ` prefix with numeric line and column.
    fn assert_located(e: &str) {
        let rest = e.strip_prefix("t.yaml:").unwrap_or_else(|| panic!("{e}"));
        let mut it = rest.splitn(3, ':');
        let (l, c) = (it.next().unwrap(), it.next().unwrap());
        assert!(
            l.parse::<usize>().is_ok() && c.parse::<usize>().is_ok(),
            "{e}"
        );
        assert!(it.next().is_some_and(|m| m.starts_with(' ')), "{e}");
    }

    #[test]
    fn malformed_yaml_has_location() {
        assert_located(&err("version: 1\nname: [\n")); // syntax
        let e = err("version: 1\nname: p\nblocks:\n  a: { use: Bad }\n"); // schema
        assert_located(&e);
        assert!(
            e.starts_with("t.yaml:4:") && !e.contains("config error"),
            "{e}"
        );
    }

    #[test]
    fn alias_bomb_is_an_error() {
        let mut s = String::from(
            "version: 1\nname: p\nblocks:\n  a:\n    use: std/x\n    with:\n      x0: &a0 [lol, lol, lol, lol, lol, lol, lol, lol, lol]\n",
        );
        for i in 1..12 {
            let refs = vec![format!("*a{}", i - 1); 9].join(", ");
            s.push_str(&format!("      x{i}: &a{i} [{refs}]\n"));
        }
        assert!(err(&s).contains("repetition limit exceeded"));
    }

    #[test]
    fn deep_nesting_is_an_error() {
        let deep = format!("{}1{}", "[".repeat(200), "]".repeat(200));
        let e = err(&format!(
            "version: 1\nname: p\nblocks: {{a: {{use: std/x, with: {deep}}}}}\n"
        ));
        assert!(e.contains("recursion limit exceeded"), "{e}");
        let deep = format!("{}1{}", "[".repeat(5_000), "]".repeat(5_000));
        assert!(parse(&deep).is_err());
    }

    #[test]
    fn lineage_sink_unknown_field_is_located() {
        let e = err("version: 1\nname: p\nlineage: [{ sink: json, bogus: 1 }]\nblocks: {}\n");
        assert!(e.contains("bogus") && !e.contains("Raw"), "{e}");
        assert_located(&e);
    }

    #[test]
    fn version_and_duplicates_rejected() {
        assert!(err("version: 2\nname: p\nblocks: {}\n").contains("version"));
        let e = err("version: 1\nname: p\nblocks:\n  a: { use: std/x }\n  a: { use: std/y }\n");
        assert!(e.contains("duplicate"), "{e}");
        let e = err("version: 1\nname: p\nblocks:\n  a: { use: std/x, in: { i: b.c, i: b.d } }\n");
        assert!(e.contains("duplicate"), "{e}");
    }

    #[test]
    fn secret_names_are_validated() {
        for bad in ["${secret:}", "${secret:A${env:B}}"] {
            let e = err(&format!(
                "version: 1\nname: p\nblocks:\n  a: {{ use: std/x, with: {{ t: \"{bad}\" }} }}\n"
            ));
            assert!(e.starts_with("t.yaml: block `a`"), "{e}");
        }
    }

    #[test]
    fn missing_file_names_path() {
        let e = PipelineConfig::from_yaml_file("/nonexistent/mallops.yaml", &|_| None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("/nonexistent/mallops.yaml"), "{e}");
    }

    const SPEC_EXAMPLE: &str = r#"
version: 1
name: churn-classifier
store:
  kind: local
  path: ./.mallops
lineage:
  - sink: json
  - sink: openlineage
    with: { url: "${env:OPENLINEAGE_URL}" }
blocks:
  raw:
    use: std/ingest.csv
    with:
      path: data/customers.csv
  features:
    use: std/preprocess.tabular
    in: { data: raw.dataset }
    with:
      target: churned
      steps:
        - impute: { strategy: median }
        - scale: { method: standard }
        - split: { test: 0.2, seed: 42 }
  model:
    use: std/external
    in: { train: features.train }
    with:
      command: ["python", "./train.py"]
      inputs: { train: features }
      outputs: { model: model }
      flavor: sklearn-pickle
      params: { n_estimators: 200 }
  eval:
    use: std/evaluate.classification
    in: { model: model.model, test: features.test }
    with:
      gate: { metric: f1, min: 0.80 }
  package:
    use: std/package
    in: { model: model.model, preprocess: features.transform, metrics: eval.metrics }
serve:
  package: package.package
  host: 127.0.0.1
  port: 8080
"#;

    #[test]
    fn spec_example_interpolates_env() {
        let env = |k: &str| (k == "OPENLINEAGE_URL").then(|| "http://ol:5000".to_owned());
        let c = PipelineConfig::from_yaml_str(SPEC_EXAMPLE, "t.yaml", &env).unwrap();
        let LineageConfig::Sinks(v) = &c.lineage else {
            panic!("expected sinks")
        };
        assert_eq!(v[1].with["url"], "http://ol:5000");
        assert_eq!(c.blocks.len(), 5);
    }

    #[test]
    fn missing_env_names_origin_and_sink() {
        let e = err(SPEC_EXAMPLE);
        assert!(e.contains("t.yaml") && e.contains("openlineage"), "{e}");
        assert!(e.contains("OPENLINEAGE_URL"), "{e}");
    }

    #[test]
    fn env_in_store_path_and_block_with_only() {
        let env = |k: &str| (k == "D").then(|| "/data".to_owned());
        let src = "version: 1\nname: p\nstore: { path: \"${env:D}/s\" }\nblocks:\n  a: { use: std/x, with: { p: \"${env:D}\" }, in: { i: \"${env:NOPE}\" } }\n";
        let c = PipelineConfig::from_yaml_str(src, "t", &env).unwrap();
        assert_eq!(c.store.path, "/data/s");
        assert_eq!(c.blocks["a"].with["p"], "/data");
        assert_eq!(c.blocks["a"].r#in["i"], "${env:NOPE}"); // not interpolated
        let e = PipelineConfig::from_yaml_str(
            "version: 1\nname: p\nblocks:\n  a: { use: std/x, with: { p: \"${env:D}\" } }\n",
            "t",
            &|_| None,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("block `a`"), "{e}");
    }

    #[test]
    fn secret_reference_stays_symbolic() {
        const SENTINEL: &str = "SENTINEL-SECRET-VALUE";
        // the lookup would leak the value if secrets were resolved through it
        let env = |_: &str| Some(SENTINEL.to_owned());
        let src = "version: 1\nname: p\nlineage: [{ sink: s, with: { k: \"${secret:API_TOKEN}\" } }]\nblocks:\n  a: { use: std/x, with: { token: \"${secret:API_TOKEN}\" } }\n";
        let c = PipelineConfig::from_yaml_str(src, "t", &env).unwrap();
        assert_eq!(c.blocks["a"].with["token"], "${secret:API_TOKEN}");
        assert!(
            matches!(&c.lineage, LineageConfig::Sinks(v) if v[0].with["k"] == "${secret:API_TOKEN}")
        );
        let json = serde_json::to_string(&c).unwrap();
        for out in [json, format!("{c:?}")] {
            assert!(!out.contains(SENTINEL));
            assert!(out.contains("${secret:API_TOKEN}") || out.contains("secret:API_TOKEN"));
        }
    }
}

#[cfg(all(test, feature = "schema"))]
mod schema_tests {
    #[test]
    fn schema_is_valid_object() {
        let s = super::pipeline_config_json_schema();
        assert!(s.is_object());
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["blocks"].is_object());
        assert!(s["$defs"]["BlockConfig"]["properties"]["use"]["type"] == "string");
        assert!(s["$defs"]["LineageConfig"]["anyOf"].is_array());
        let req: Vec<_> = s["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        for k in ["version", "name", "blocks"] {
            assert!(req.contains(&k), "{req:?}");
        }
        assert_eq!(s["additionalProperties"], false);
    }
}
