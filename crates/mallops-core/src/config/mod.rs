//! Typed `mallops.yaml` configuration (SPEC §5).
//!
//! `${env:NAME}` values are interpolated at load time; `${secret:NAME}` references stay
//! symbolic and are resolved on demand through `Context::secret` (ADR-0007 D7).

#[allow(dead_code)] // ponytail: wired into the parser by task 5.3
mod interp;

use std::collections::BTreeMap;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::BlockRef;
#[cfg(feature = "yaml")]
use crate::Error;

fn default_store_kind() -> String {
    "local".to_owned()
}

fn default_store_path() -> String {
    "./.mallops".to_owned()
}

fn default_host() -> String {
    "127.0.0.1".to_owned()
}

fn default_port() -> u16 {
    8080
}

/// Artifact store settings (`store:`); defaults to a local store at `./.mallops`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct StoreConfig {
    /// Store kind (`local`, `memory`, ...).
    #[serde(default = "default_store_kind")]
    pub kind: String,
    /// Store location (used by `local`).
    #[serde(default = "default_store_path")]
    pub path: String,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            kind: default_store_kind(),
            path: default_store_path(),
        }
    }
}

/// One lineage sink (`- sink: json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Str(String),
            Sinks(Vec<LineageSinkConfig>),
        }
        match Raw::deserialize(d)? {
            Raw::Str(s) if s == "off" => Ok(Self::Off),
            Raw::Str(s) => Err(serde::de::Error::custom(format!(
                "invalid lineage `{s}`: expected `off` or a list of sinks"
            ))),
            Raw::Sinks(v) => Ok(Self::Sinks(v)),
        }
    }
}

/// Serde adapter reusing `BlockRef`'s `FromStr`, so reserved prefixes keep their error.
mod block_ref {
    use super::*;

    pub fn serialize<S: Serializer>(b: &BlockRef, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(b)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BlockRef, D::Error> {
        let s = String::deserialize(d)?;
        BlockRef::from_str(&s).map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

/// One block of the batch DAG (`blocks.<id>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct BlockConfig {
    /// Block reference (`use:`).
    #[serde(rename = "use", with = "block_ref")]
    pub r#use: BlockRef,
    /// Input wiring: port name to `<block>.<output>` (`in:`).
    #[serde(rename = "in", default)]
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
        let _ = env; // ponytail: interpolazione collegata in 5.3
        yaml_serde::from_str(src).map_err(|e| {
            Error::Config(match e.location() {
                Some(l) => format!("{origin}:{}:{}: {e}", l.line(), l.column()),
                None => format!("{origin}: {e}"),
            })
        })
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
    use: std/train.external
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

    #[test]
    fn malformed_yaml_has_location() {
        let e = err("version: 1\nname: [\n");
        assert!(e.starts_with("t.yaml:"), "{e}");
        let e = err("version: 1\nname: p\nblocks:\n  a: { use: Bad }\n");
        assert!(e.starts_with("t.yaml:4:"), "{e}");
    }

    #[test]
    fn alias_bomb_is_an_error() {
        let mut s = String::from(
            "version: 1\nname: p\nx0: &a0 [lol, lol, lol, lol, lol, lol, lol, lol, lol]\n",
        );
        for i in 1..12 {
            let refs = vec![format!("*a{}", i - 1); 9].join(", ");
            s.push_str(&format!("x{i}: &a{i} [{refs}]\n"));
        }
        s.push_str("blocks: {a: {use: std/x, with: *a11}}\n");
        assert!(parse(&s).is_err());
    }

    #[test]
    fn deep_nesting_is_an_error() {
        let deep = format!("{}1{}", "[".repeat(200), "]".repeat(200));
        assert!(
            parse(&format!(
                "version: 1\nname: p\nblocks: {{a: {{use: std/x, with: {deep}}}}}\n"
            ))
            .is_err()
        );
        let deep = format!("{}1{}", "[".repeat(100_000), "]".repeat(100_000));
        assert!(parse(&deep).is_err());
    }
}
