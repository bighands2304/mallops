//! Format-agnostic artifact model and canonical `ArtifactId` v1 encoding.
//!
//! An [`Artifact`] pairs a content-addressed [`ArtifactId`] with a canonical
//! [`Manifest`]. Per-run provenance is intentionally excluded from [`Manifest`]
//! so identical deterministic runs produce identical [`ArtifactId`] digests
//! (see ADR-0007 D1 and `docs/SPEC.md` §6).

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

use crate::Error;

/// Domain-separation context string for canonical [`ArtifactId`] v1 hashing.
pub const ARTIFACT_ID_V1_CONTEXT: &str = "mallops artifact v1";

const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn parse_lowercase_hex_32(input: &str) -> Result<[u8; 32], String> {
    let raw = input.as_bytes();
    if raw.len() != Digest::HEX_LEN {
        return Err(format!(
            "expected {} lowercase hexadecimal characters, got {} bytes",
            Digest::HEX_LEN,
            raw.len()
        ));
    }

    let mut bytes = [0u8; 32];
    for (i, chunk) in raw.chunks_exact(2).enumerate() {
        let hi = decode_hex_nibble(chunk[0]).ok_or_else(|| {
            format!(
                "invalid character at byte offset {}: expected lowercase hex [0-9a-f]",
                i * 2
            )
        })?;
        let lo = decode_hex_nibble(chunk[1]).ok_or_else(|| {
            format!(
                "invalid character at byte offset {}: expected lowercase hex [0-9a-f]",
                i * 2 + 1
            )
        })?;
        bytes[i] = (hi << 4) | lo;
    }

    Ok(bytes)
}

fn fmt_lowercase_hex(bytes: &[u8; 32], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for &byte in bytes {
        f.write_char(HEX_DIGITS[(byte >> 4) as usize] as char)?;
        f.write_char(HEX_DIGITS[(byte & 0x0f) as usize] as char)?;
    }
    Ok(())
}

fn to_lowercase_hex(bytes: &[u8; 32]) -> String {
    let mut hex = String::with_capacity(Digest::HEX_LEN);
    for &byte in bytes {
        hex.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        hex.push(HEX_DIGITS[(byte & 0x0f) as usize] as char);
    }
    hex
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let code = c as u8;
                out.push_str("\\u00");
                out.push(HEX_DIGITS[(code >> 4) as usize] as char);
                out.push(HEX_DIGITS[(code & 0x0f) as usize] as char);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_canonical_value(out: &mut String, value: &serde_json::Value) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(true) => out.push_str("true"),
        serde_json::Value::Bool(false) => out.push_str("false"),
        serde_json::Value::Number(n) => {
            // Writing to `String` via `fmt::Write` is infallible.
            let _ = write!(out, "{n}");
        }
        serde_json::Value::String(s) => write_json_string(out, s),
        serde_json::Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_canonical_value(out, item);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.as_bytes().cmp(b.as_bytes()));
            out.push('{');
            for (idx, (key, val)) in entries.into_iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_json_string(out, key);
                out.push(':');
                write_canonical_value(out, val);
            }
            out.push('}');
        }
    }
}

/// A 32-byte BLAKE3 cryptographic digest formatted as 64 lowercase hexadecimal characters.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest([u8; 32]);

impl Digest {
    /// Raw byte length of a BLAKE3 digest.
    pub const LEN: usize = 32;

    /// Length of the lowercase hexadecimal string representation.
    pub const HEX_LEN: usize = 64;

    /// Constructs a [`Digest`] from its raw 32-byte value.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Computes the standard BLAKE3 digest of `data`.
    #[must_use]
    pub fn hash(data: &[u8]) -> Self {
        Self(*blake3::hash(data).as_bytes())
    }

    /// Returns the raw 32-byte digest.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Encodes the digest as 64 lowercase hexadecimal characters.
    #[must_use]
    pub fn to_hex(&self) -> String {
        to_lowercase_hex(&self.0)
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_lowercase_hex(&self.0, f)
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Digest(\"")?;
        fmt_lowercase_hex(&self.0, f)?;
        f.write_str("\")")
    }
}

impl FromStr for Digest {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_lowercase_hex_32(s)
            .map(Self)
            .map_err(Error::InvalidDigest)
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse::<Self>().map_err(de::Error::custom)
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for Digest {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Digest".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        concat!(module_path!(), "::Digest").into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": "32-byte BLAKE3 digest encoded as 64 lowercase hexadecimal characters.",
            "minLength": 64,
            "maxLength": 64,
            "pattern": "^[0-9a-f]{64}$"
        })
    }
}

/// Validated content-addressed identifier of an [`Artifact`].
///
/// Represented as a 32-byte canonical BLAKE3 digest encoded as 64 lowercase
/// hexadecimal characters. Cannot be constructed from an unvalidated string.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactId(Digest);

impl ArtifactId {
    /// Creates an [`ArtifactId`] from a validated 32-byte [`Digest`].
    #[must_use]
    pub const fn from_digest(digest: Digest) -> Self {
        Self(digest)
    }

    /// Creates an [`ArtifactId`] from its raw 32-byte digest.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(Digest::from_bytes(bytes))
    }

    /// Returns the underlying 32-byte [`Digest`].
    #[must_use]
    pub const fn digest(&self) -> Digest {
        self.0
    }

    /// Returns the raw 32-byte digest.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }

    /// Encodes this identifier as 64 lowercase hexadecimal characters.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.0.to_hex()
    }

    /// Computes the canonical `ArtifactId` v1 digest from a [`Manifest`] and
    /// pre-digested payload [`PayloadFile`] entries.
    ///
    /// Canonical `ArtifactId` v1 framing (ADR-0007 D1 / `SPEC.md` §6):
    /// 1. Domain-separated BLAKE3 derive-key hasher with context `"mallops artifact v1"`.
    /// 2. `u64::to_le_bytes(manifest_bytes.len()) || manifest_bytes`, where
    ///    `manifest_bytes` is the canonical sorted-key JSON encoding of `manifest`
    ///    with `parents` sorted lexicographically.
    /// 3. `u64::to_le_bytes(file_count)` followed by each file sorted bytewise
    ///    by its validated [`RelPath`]:
    ///    `u64::to_le_bytes(path.len()) || path || u64::to_le_bytes(content_len) || blake3(content)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPath`] if `files` contains duplicate [`RelPath`] entries.
    pub fn compute(manifest: &Manifest, files: &[PayloadFile]) -> Result<Self, Error> {
        let mut sorted: Vec<&PayloadFile> = files.iter().collect();
        sorted.sort_by(|a, b| a.path.as_str().as_bytes().cmp(b.path.as_str().as_bytes()));

        for window in sorted.windows(2) {
            if window[0].path == window[1].path {
                return Err(Error::InvalidPath(format!(
                    "duplicate payload path: {}",
                    window[0].path
                )));
            }
        }

        let manifest_bytes = manifest.canonical_json();
        let mut hasher = blake3::Hasher::new_derive_key(ARTIFACT_ID_V1_CONTEXT);

        hasher.update(&(manifest_bytes.len() as u64).to_le_bytes());
        hasher.update(&manifest_bytes);

        hasher.update(&(sorted.len() as u64).to_le_bytes());
        for file in sorted {
            let path_bytes = file.path.as_str().as_bytes();
            hasher.update(&(path_bytes.len() as u64).to_le_bytes());
            hasher.update(path_bytes);
            hasher.update(&file.len.to_le_bytes());
            hasher.update(file.digest.as_bytes());
        }

        let digest = Digest::from_bytes(*hasher.finalize().as_bytes());
        Ok(Self(digest))
    }

    /// Computes the canonical `ArtifactId` v1 digest from a [`Manifest`] and
    /// in-memory `(RelPath, &[u8])` payload files.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPath`] if `files` contains duplicate [`RelPath`] entries.
    pub fn compute_from_bytes<'a, I>(manifest: &Manifest, files: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (RelPath, &'a [u8])>,
    {
        let entries: Vec<PayloadFile> = files
            .into_iter()
            .map(|(path, content)| PayloadFile::from_bytes(path, content))
            .collect();
        Self::compute(manifest, &entries)
    }
}

impl From<Digest> for ArtifactId {
    fn from(digest: Digest) -> Self {
        Self(digest)
    }
}

impl From<ArtifactId> for Digest {
    fn from(id: ArtifactId) -> Self {
        id.0
    }
}

impl fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_lowercase_hex(self.0.as_bytes(), f)
    }
}

impl fmt::Debug for ArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ArtifactId(\"")?;
        fmt_lowercase_hex(self.0.as_bytes(), f)?;
        f.write_str("\")")
    }
}

impl FromStr for ArtifactId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_lowercase_hex_32(s)
            .map(|bytes| Self(Digest::from_bytes(bytes)))
            .map_err(Error::InvalidArtifactId)
    }
}

impl Serialize for ArtifactId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ArtifactId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse::<Self>().map_err(de::Error::custom)
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for ArtifactId {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ArtifactId".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        concat!(module_path!(), "::ArtifactId").into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": "32-byte canonical BLAKE3 artifact identifier encoded as 64 lowercase hexadecimal characters.",
            "minLength": 64,
            "maxLength": 64,
            "pattern": "^[0-9a-f]{64}$"
        })
    }
}

/// Validated non-empty UTF-8 POSIX relative path using `/` separators.
///
/// Guarantees that the path contains no NUL bytes, backslashes, leading or
/// trailing slashes, empty segments, `.` or `..` traversal segments, or
/// Windows drive prefixes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelPath(String);

impl RelPath {
    /// Parses and validates a relative POSIX path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPath`] if `input` is empty, contains NUL bytes
    /// or backslashes, starts with `/` or a Windows drive prefix (e.g. `C:`),
    /// ends with `/`, or contains empty, `.`, or `..` segments.
    pub fn parse(input: &str) -> Result<Self, Error> {
        if input.is_empty() {
            return Err(Error::InvalidPath("path must not be empty".to_owned()));
        }
        if input.contains('\0') {
            return Err(Error::InvalidPath(
                "path must not contain NUL bytes".to_owned(),
            ));
        }
        if input.contains('\\') {
            return Err(Error::InvalidPath(
                "path must use '/' separators and must not contain '\\'".to_owned(),
            ));
        }
        let bytes = input.as_bytes();
        if bytes[0] == b'/' {
            return Err(Error::InvalidPath(format!(
                "path must be relative (leading '/'): {input}"
            )));
        }
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            return Err(Error::InvalidPath(format!(
                "path must not contain a Windows drive prefix: {input}"
            )));
        }

        for segment in input.split('/') {
            if segment.is_empty() {
                return Err(Error::InvalidPath(format!(
                    "path must not contain empty segments or a trailing '/': {input}"
                )));
            }
            if segment == "." || segment == ".." {
                return Err(Error::InvalidPath(format!(
                    "path must not contain '{segment}' segments: {input}"
                )));
            }
        }

        Ok(Self(input.to_owned()))
    }

    /// Returns the validated path as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the validated path as a [`std::path::Path`].
    #[must_use]
    pub fn as_path(&self) -> &Path {
        Path::new(&self.0)
    }
}

impl AsRef<str> for RelPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RelPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for RelPath {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for RelPath {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for RelPath {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(de::Error::custom)
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for RelPath {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "RelPath".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        concat!(module_path!(), "::RelPath").into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": "Validated non-empty UTF-8 POSIX relative path using '/' separators.",
            "minLength": 1
        })
    }
}

/// Content summary of a single payload file used when computing an [`ArtifactId`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PayloadFile {
    /// Validated relative POSIX path within the artifact payload.
    pub path: RelPath,
    /// Byte length of the file contents.
    pub len: u64,
    /// Raw BLAKE3 digest of the file contents.
    pub digest: Digest,
}

impl PayloadFile {
    /// Creates a [`PayloadFile`] from a validated path, byte length, and precomputed [`Digest`].
    #[must_use]
    pub const fn new(path: RelPath, len: u64, digest: Digest) -> Self {
        Self { path, len, digest }
    }

    /// Creates a [`PayloadFile`] by hashing an in-memory byte slice.
    #[must_use]
    pub fn from_bytes(path: RelPath, content: &[u8]) -> Self {
        Self {
            path,
            len: content.len() as u64,
            digest: Digest::hash(content),
        }
    }
}

/// High-level semantic category of an [`Artifact`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ArtifactKind {
    /// Raw or curated dataset (e.g. Parquet, Arrow IPC, CSV, JSON, images).
    Dataset,
    /// Preprocessed feature splits derived from a dataset.
    Features,
    /// Fitted preprocessor (scaler, imputer, encoder) reused by packaging and serving.
    Transform,
    /// Trained model artifact; `flavor` selects a compatible predictor.
    Model,
    /// Evaluation metrics and optional diagnostic outputs.
    Metrics,
    /// Deployable bundle combining a model, transforms, and manifest.
    Package,
    /// Opaque escape-hatch artifact for arbitrary payloads.
    Blob,
}

impl ArtifactKind {
    /// Returns the canonical lowercase string representation of this kind.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Dataset => "dataset",
            Self::Features => "features",
            Self::Transform => "transform",
            Self::Model => "model",
            Self::Metrics => "metrics",
            Self::Package => "package",
            Self::Blob => "blob",
        }
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Format-agnostic metadata describing an immutable [`Artifact`].
///
/// Per-run provenance (`run`, `node`) is intentionally omitted so identical
/// deterministic runs hash to the same [`ArtifactId`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct Manifest {
    /// Manifest schema version (currently [`Manifest::SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Semantic artifact category.
    pub kind: ArtifactKind,
    /// Payload format tag (e.g. `"parquet"`, `"pickle"`, `"json"`).
    pub format: String,
    /// Optional model flavor tag selecting a compatible predictor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flavor: Option<String>,
    /// Optional input/output schema describing data or model signatures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub io_schema: Option<serde_json::Value>,
    /// Free-form structured metadata associated with the artifact.
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// Content-addressed IDs of upstream artifacts that produced this artifact.
    #[serde(default)]
    pub parents: Vec<ArtifactId>,
}

impl Manifest {
    /// Current manifest schema version (`1`).
    pub const SCHEMA_VERSION: u32 = 1;

    /// Creates a new [`Manifest`] with [`Manifest::SCHEMA_VERSION`], `kind`, and `format`.
    #[must_use]
    pub fn new(kind: ArtifactKind, format: impl Into<String>) -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            kind,
            format: format.into(),
            flavor: None,
            io_schema: None,
            metadata: BTreeMap::new(),
            parents: Vec::new(),
        }
    }

    /// Sets the model `flavor` tag.
    #[must_use]
    pub fn with_flavor(mut self, flavor: impl Into<String>) -> Self {
        self.flavor = Some(flavor.into());
        self
    }

    /// Sets the optional `io_schema` JSON document.
    #[must_use]
    pub fn with_io_schema(mut self, io_schema: serde_json::Value) -> Self {
        self.io_schema = Some(io_schema);
        self
    }

    /// Replaces the `metadata` map.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Inserts a single entry into `metadata`.
    #[must_use]
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Replaces `parents` with the items from `parents`.
    #[must_use]
    pub fn with_parents(mut self, parents: impl IntoIterator<Item = ArtifactId>) -> Self {
        self.parents = parents.into_iter().collect();
        self
    }

    /// Appends a single parent [`ArtifactId`].
    #[must_use]
    pub fn with_parent(mut self, parent: ArtifactId) -> Self {
        self.parents.push(parent);
        self
    }

    /// Serializes this manifest as canonical compact JSON with lexicographically
    /// sorted object keys (at every nesting level) and lexicographically sorted
    /// `parents`.
    #[must_use]
    pub fn canonical_json(&self) -> Vec<u8> {
        let mut out = String::new();
        out.push('{');

        let mut first = true;
        let mut push_key = |out: &mut String, key: &str| {
            if !first {
                out.push(',');
            }
            first = false;
            write_json_string(out, key);
            out.push(':');
        };

        if let Some(ref flavor) = self.flavor {
            push_key(&mut out, "flavor");
            write_json_string(&mut out, flavor);
        }

        push_key(&mut out, "format");
        write_json_string(&mut out, &self.format);

        if let Some(ref io_schema) = self.io_schema {
            push_key(&mut out, "io_schema");
            write_canonical_value(&mut out, io_schema);
        }

        push_key(&mut out, "kind");
        write_json_string(&mut out, self.kind.as_str());

        push_key(&mut out, "metadata");
        out.push('{');
        for (idx, (key, val)) in self.metadata.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            write_json_string(&mut out, key);
            out.push(':');
            write_canonical_value(&mut out, val);
        }
        out.push('}');

        push_key(&mut out, "parents");
        let mut sorted_parents = self.parents.clone();
        sorted_parents.sort();
        out.push('[');
        for (idx, parent) in sorted_parents.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            out.push('"');
            out.push_str(&parent.to_hex());
            out.push('"');
        }
        out.push(']');

        push_key(&mut out, "schema_version");
        let _ = write!(out, "{}", self.schema_version);

        out.push('}');
        out.into_bytes()
    }
}

/// An immutable, content-addressed artifact combining its [`ArtifactId`] and [`Manifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct Artifact {
    /// Validated 32-byte canonical BLAKE3 identifier of this artifact.
    pub id: ArtifactId,
    /// Format-agnostic metadata describing this artifact.
    pub manifest: Manifest,
}

impl Artifact {
    /// Creates a new [`Artifact`] from its content-addressed identifier and manifest.
    #[must_use]
    pub const fn new(id: ArtifactId, manifest: Manifest) -> Self {
        Self { id, manifest }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parent_a() -> ArtifactId {
        ArtifactId::from_bytes([0x11; 32])
    }

    fn parent_b() -> ArtifactId {
        ArtifactId::from_bytes([0xaa; 32])
    }

    fn golden_manifest() -> Manifest {
        Manifest::new(ArtifactKind::Model, "pickle")
            .with_flavor("sklearn-pickle")
            .with_io_schema(json!({
                "outputs": [{"dtype": "f64", "name": "churned"}],
                "inputs": [{"dtype": "f64", "name": "age"}]
            }))
            .with_metadata_entry("framework_version", json!("1.6.0"))
            .with_metadata_entry("framework", json!("scikit-learn"))
            .with_parents([parent_b(), parent_a()])
    }

    /// Golden test vector for canonical `ArtifactId` v1 (ADR-0007 D1 / SPEC §6).
    ///
    /// How this vector was obtained:
    /// 1. Construct `golden_manifest()` (`schema_version = 1`, `kind = model`,
    ///    `format = "pickle"`, `flavor = "sklearn-pickle"`, `io_schema` and
    ///    `metadata` with unsorted insertion order, `parents = [0xaa..aa, 0x11..11]`).
    /// 2. Canonicalize `Manifest` as compact JSON with lexicographically sorted
    ///    keys at every level and lexicographically sorted `parents`:
    ///    `{"flavor":"sklearn-pickle","format":"pickle","io_schema":{"inputs":[{"dtype":"f64","name":"age"}],"outputs":[{"dtype":"f64","name":"churned"}]},"kind":"model","metadata":{"framework":"scikit-learn","framework_version":"1.6.0"},"parents":["1111...11","aaaa...aa"],"schema_version":1}`
    /// 3. Encode the length-prefixed frame:
    ///    - `u64::to_le_bytes(manifest_json.len()) || manifest_json`
    ///    - `u64::to_le_bytes(2)` (two payload files)
    ///    - file 0 (`"config.json"`, `b"{\"seed\":42}"`):
    ///      `u64::to_le_bytes(11) || b"config.json" || u64::to_le_bytes(11) || blake3(b"{\"seed\":42}")`
    ///    - file 1 (`"model/weights.bin"`, `b"\x01\x02\x03\x04"`):
    ///      `u64::to_le_bytes(17) || b"model/weights.bin" || u64::to_le_bytes(4) || blake3(b"\x01\x02\x03\x04")`
    /// 4. Compute `blake3::derive_key("mallops artifact v1", &frame)` and format
    ///    the 32-byte digest as 64 lowercase hexadecimal characters.
    #[test]
    fn golden_artifact_id_v1_vector() {
        let expected_canonical_manifest = concat!(
            r#"{"flavor":"sklearn-pickle","format":"pickle","#,
            r#""io_schema":{"inputs":[{"dtype":"f64","name":"age"}],"outputs":[{"dtype":"f64","name":"churned"}]},"#,
            r#""kind":"model","metadata":{"framework":"scikit-learn","framework_version":"1.6.0"},"#,
            r#""parents":["1111111111111111111111111111111111111111111111111111111111111111","#,
            r#""aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],"#,
            r#""schema_version":1}"#
        );

        let manifest = golden_manifest();
        let manifest_bytes = manifest.canonical_json();
        assert_eq!(
            std::str::from_utf8(&manifest_bytes).unwrap(),
            expected_canonical_manifest
        );

        let config_bytes = br#"{"seed":42}"#;
        let weights_bytes = b"\x01\x02\x03\x04";

        // Independently assemble the exact byte stream and hash via one-shot blake3::derive_key.
        let mut frame = Vec::new();
        frame.extend_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
        frame.extend_from_slice(&manifest_bytes);
        frame.extend_from_slice(&2u64.to_le_bytes());

        // Bytewise sorted order: "config.json" < "model/weights.bin"
        frame.extend_from_slice(&(b"config.json".len() as u64).to_le_bytes());
        frame.extend_from_slice(b"config.json");
        frame.extend_from_slice(&(config_bytes.len() as u64).to_le_bytes());
        frame.extend_from_slice(blake3::hash(config_bytes).as_bytes());

        frame.extend_from_slice(&(b"model/weights.bin".len() as u64).to_le_bytes());
        frame.extend_from_slice(b"model/weights.bin");
        frame.extend_from_slice(&(weights_bytes.len() as u64).to_le_bytes());
        frame.extend_from_slice(blake3::hash(weights_bytes).as_bytes());

        let derived = ArtifactId::from_bytes(blake3::derive_key(ARTIFACT_ID_V1_CONTEXT, &frame));

        let computed = ArtifactId::compute_from_bytes(
            &manifest,
            [
                (
                    RelPath::parse("model/weights.bin").unwrap(),
                    weights_bytes.as_slice(),
                ),
                (
                    RelPath::parse("config.json").unwrap(),
                    config_bytes.as_slice(),
                ),
            ],
        )
        .unwrap();

        const GOLDEN_HEX: &str = "18269ff8ca79e3534cb4329b6b4ebe60a846f63278eba16c7ee3a5023252e804";
        let expected: ArtifactId = GOLDEN_HEX.parse().unwrap();

        assert_eq!(derived, expected);
        assert_eq!(computed, expected);
        assert_eq!(computed.to_hex(), GOLDEN_HEX);

        // Permuting metadata insertion order, nested io_schema key order, parents
        // order, and payload file order must yield the exact same golden digest.
        let reordered_manifest = Manifest::new(ArtifactKind::Model, "pickle")
            .with_flavor("sklearn-pickle")
            .with_io_schema(json!({
                "inputs": [{"name": "age", "dtype": "f64"}],
                "outputs": [{"name": "churned", "dtype": "f64"}]
            }))
            .with_metadata_entry("framework", json!("scikit-learn"))
            .with_metadata_entry("framework_version", json!("1.6.0"))
            .with_parents([parent_a(), parent_b()]);

        let reordered_id = ArtifactId::compute_from_bytes(
            &reordered_manifest,
            [
                (
                    RelPath::parse("config.json").unwrap(),
                    config_bytes.as_slice(),
                ),
                (
                    RelPath::parse("model/weights.bin").unwrap(),
                    weights_bytes.as_slice(),
                ),
            ],
        )
        .unwrap();
        assert_eq!(reordered_id, expected);
    }

    #[test]
    fn canonical_id_is_independent_of_insertion_orders() {
        let mut meta_a = BTreeMap::new();
        meta_a.insert("z_key".to_owned(), json!({"b": 2, "a": 1}));
        meta_a.insert("a_key".to_owned(), json!([3, 2, 1]));

        let mut meta_b = BTreeMap::new();
        meta_b.insert("a_key".to_owned(), json!([3, 2, 1]));
        meta_b.insert("z_key".to_owned(), json!({"a": 1, "b": 2}));

        let manifest_1 = Manifest::new(ArtifactKind::Model, "onnx")
            .with_flavor("onnx")
            .with_io_schema(json!({"z": true, "a": {"y": 1, "x": 2}}))
            .with_metadata(meta_a)
            .with_parents([parent_b(), parent_a()]);

        let manifest_2 = Manifest::new(ArtifactKind::Model, "onnx")
            .with_flavor("onnx")
            .with_io_schema(json!({"a": {"x": 2, "y": 1}, "z": true}))
            .with_metadata(meta_b)
            .with_parents([parent_a(), parent_b()]);

        assert_eq!(manifest_1.canonical_json(), manifest_2.canonical_json());

        let p1 = RelPath::parse("a/first.bin").unwrap();
        let p2 = RelPath::parse("b/second.bin").unwrap();
        let p3 = RelPath::parse("a/alpha.bin").unwrap();

        let id_1 = ArtifactId::compute_from_bytes(
            &manifest_1,
            [
                (p2.clone(), b"two".as_slice()),
                (p1.clone(), b"one".as_slice()),
                (p3.clone(), b"alpha".as_slice()),
            ],
        )
        .unwrap();

        let id_2 = ArtifactId::compute_from_bytes(
            &manifest_2,
            [
                (p3, b"alpha".as_slice()),
                (p1, b"one".as_slice()),
                (p2, b"two".as_slice()),
            ],
        )
        .unwrap();

        assert_eq!(id_1, id_2);
    }

    #[test]
    fn length_prefix_prevents_payload_and_manifest_boundary_collisions() {
        let manifest = Manifest::new(ArtifactKind::Blob, "raw");

        // 1. Content boundary shift across two files: a="ab", b="" vs a="a", b="b".
        let id_ab_empty = ArtifactId::compute_from_bytes(
            &manifest,
            [
                (RelPath::parse("a").unwrap(), b"ab".as_slice()),
                (RelPath::parse("b").unwrap(), b"".as_slice()),
            ],
        )
        .unwrap();
        let id_a_b = ArtifactId::compute_from_bytes(
            &manifest,
            [
                (RelPath::parse("a").unwrap(), b"a".as_slice()),
                (RelPath::parse("b").unwrap(), b"b".as_slice()),
            ],
        )
        .unwrap();
        assert_ne!(id_ab_empty, id_a_b);

        // 2. Path/content boundary shift: path="ab", content="" vs path="a", content="b".
        let id_path_ab = ArtifactId::compute_from_bytes(
            &manifest,
            [(RelPath::parse("ab").unwrap(), b"".as_slice())],
        )
        .unwrap();
        let id_path_a = ArtifactId::compute_from_bytes(
            &manifest,
            [(RelPath::parse("a").unwrap(), b"b".as_slice())],
        )
        .unwrap();
        assert_ne!(id_path_ab, id_path_a);

        // 3. Manifest/payload boundary shift: verify that appending extra bytes to the
        //    manifest vs shifting them into the payload section produces distinct hashes.
        let manifest_with_meta =
            Manifest::new(ArtifactKind::Blob, "raw").with_metadata_entry("k", json!("v"));
        let id_meta_no_files = ArtifactId::compute_from_bytes(&manifest_with_meta, []).unwrap();
        let id_empty_with_file = ArtifactId::compute_from_bytes(
            &manifest,
            [(RelPath::parse("k").unwrap(), b"v".as_slice())],
        )
        .unwrap();
        assert_ne!(id_meta_no_files, id_empty_with_file);

        // 4. Direct raw-frame boundary shift check: two distinct (manifest_len, manifest, rest)
        //    splits of the exact same concatenated bytes must hash differently when length-prefixed.
        let raw_a = manifest.canonical_json();
        let raw_b = manifest_with_meta.canonical_json();
        assert_ne!(raw_a, raw_b);

        // Even if payload content literally equals the trailing bytes of manifest_b, the
        // manifest length prefix prevents any collision.
        let mut h1 = blake3::Hasher::new_derive_key(ARTIFACT_ID_V1_CONTEXT);
        h1.update(&(3u64).to_le_bytes());
        h1.update(b"abc");
        h1.update(&(2u64).to_le_bytes());
        h1.update(b"de");

        let mut h2 = blake3::Hasher::new_derive_key(ARTIFACT_ID_V1_CONTEXT);
        h2.update(&(2u64).to_le_bytes());
        h2.update(b"ab");
        h2.update(&(3u64).to_le_bytes());
        h2.update(b"cde");

        assert_ne!(h1.finalize(), h2.finalize());
    }

    #[test]
    fn compute_rejects_duplicate_rel_paths() {
        let manifest = Manifest::new(ArtifactKind::Blob, "raw");
        let dup = RelPath::parse("data/part.bin").unwrap();
        let err = ArtifactId::compute_from_bytes(
            &manifest,
            [
                (dup.clone(), b"first".as_slice()),
                (dup, b"second".as_slice()),
            ],
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidPath(_)));
    }

    #[test]
    fn artifact_id_from_str_validation() {
        let valid_hex = "0123456789abcdef".repeat(4);
        let parsed: ArtifactId = valid_hex.parse().unwrap();
        assert_eq!(parsed.to_string(), valid_hex);
        assert_eq!(parsed.to_hex(), valid_hex);

        let invalid_cases = [
            "",
            "0123456789abcdef",                // too short (16)
            &"0".repeat(63),                   // too short (63)
            &"0".repeat(65),                   // too long (65)
            &"A".repeat(64),                   // uppercase hex
            &format!("{}F", "a".repeat(63)),   // single uppercase char
            "../",                             // path traversal
            &format!("../{}", "a".repeat(61)), // 64 bytes with path traversal
            &format!("0x{}", "a".repeat(62)),  // 0x prefix
            &"g".repeat(64),                   // non-hex ASCII
            &format!("{} ", "a".repeat(63)),   // trailing whitespace
            &format!("{}\0", "a".repeat(63)),  // NUL byte
        ];

        for case in invalid_cases {
            let err = ArtifactId::from_str(case).unwrap_err();
            assert!(
                matches!(err, Error::InvalidArtifactId(_)),
                "expected InvalidArtifactId for {case:?}, got {err:?}"
            );

            // Serde deserialization must apply the exact same validation.
            let json_str = serde_json::to_string(case).unwrap();
            assert!(
                serde_json::from_str::<ArtifactId>(&json_str).is_err(),
                "serde accepted invalid ArtifactId {case:?}"
            );
        }
    }

    #[test]
    fn digest_round_trip_and_validation() {
        let digest = Digest::hash(b"mallops");
        let hex = digest.to_hex();
        assert_eq!(hex.len(), Digest::HEX_LEN);
        assert_eq!(hex.parse::<Digest>().unwrap(), digest);

        let serialized = serde_json::to_string(&digest).unwrap();
        let deserialized: Digest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, digest);

        assert!(matches!(
            "ABCDEF".parse::<Digest>().unwrap_err(),
            Error::InvalidDigest(_)
        ));
    }

    #[test]
    fn rel_path_parse_accepts_valid_and_rejects_invalid() {
        let valid_cases = [
            "model.bin",
            "a/b/c.parquet",
            ".hidden",
            "dir/.gitignore",
            "v1.0/foo..bar.safetensors",
            "données/modèle.bin",
        ];

        for case in valid_cases {
            let path = RelPath::parse(case).unwrap();
            assert_eq!(path.as_str(), case);
            assert_eq!(path.to_string(), case);
            assert_eq!(path.as_path(), Path::new(case));

            let json = serde_json::to_string(&path).unwrap();
            let back: RelPath = serde_json::from_str(&json).unwrap();
            assert_eq!(back, path);
        }

        let invalid_cases = [
            "", "..", "../a", "a/../b", "a/..", ".", "./a", "a/./b", "a/.", "/abs", "/", "a//b",
            "a/", r"C:\x", "C:/x", "c:foo", r"a\b", "a\0b", "\0",
        ];

        for case in invalid_cases {
            let err = RelPath::parse(case).unwrap_err();
            assert!(
                matches!(err, Error::InvalidPath(_)),
                "expected InvalidPath for {case:?}, got {err:?}"
            );

            let json_str = serde_json::to_string(case).unwrap();
            assert!(
                serde_json::from_str::<RelPath>(&json_str).is_err(),
                "serde accepted invalid RelPath {case:?}"
            );
        }
    }

    #[test]
    fn manifest_serde_round_trip_and_no_produced_by() {
        let manifest = Manifest::new(ArtifactKind::Transform, "json")
            .with_io_schema(json!({
                "inputs": [{"name": "age", "dtype": "f64"}],
                "outputs": [{"name": "age_scaled", "dtype": "f64"}]
            }))
            .with_metadata_entry("scaler", json!("standard"))
            .with_parent(parent_a());

        let value = serde_json::to_value(&manifest).unwrap();
        let obj = value.as_object().unwrap();

        assert_eq!(obj.get("schema_version"), Some(&json!(1)));
        assert_eq!(obj.get("kind"), Some(&json!("transform")));
        assert_eq!(obj.get("format"), Some(&json!("json")));
        assert!(obj.contains_key("io_schema"));
        assert!(!obj.contains_key("flavor"));
        assert!(!obj.contains_key("produced_by"));
        assert!(!obj.contains_key("signature"));

        let encoded = serde_json::to_string(&manifest).unwrap();
        let decoded: Manifest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, manifest);

        // Verify all ArtifactKind variants round-trip as lowercase strings.
        let kinds = [
            (ArtifactKind::Dataset, "dataset"),
            (ArtifactKind::Features, "features"),
            (ArtifactKind::Transform, "transform"),
            (ArtifactKind::Model, "model"),
            (ArtifactKind::Metrics, "metrics"),
            (ArtifactKind::Package, "package"),
            (ArtifactKind::Blob, "blob"),
        ];
        for (kind, expected) in kinds {
            assert_eq!(kind.as_str(), expected);
            assert_eq!(kind.to_string(), expected);
            assert_eq!(serde_json::to_value(kind).unwrap(), json!(expected));
            assert_eq!(
                serde_json::from_value::<ArtifactKind>(json!(expected)).unwrap(),
                kind
            );
        }
    }

    #[test]
    fn artifact_serde_round_trip() {
        let manifest = Manifest::new(ArtifactKind::Dataset, "parquet")
            .with_metadata_entry("rows", json!(1000));
        let id = ArtifactId::compute_from_bytes(
            &manifest,
            [(RelPath::parse("data.parquet").unwrap(), b"PAR1".as_slice())],
        )
        .unwrap();
        let artifact = Artifact::new(id, manifest);

        let encoded = serde_json::to_string(&artifact).unwrap();
        let decoded: Artifact = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, artifact);
    }

    #[test]
    fn manifest_minimal_deserialization_and_canonical_escaping() {
        let minimal: Manifest =
            serde_json::from_str(r#"{"schema_version":1,"kind":"dataset","format":"csv"}"#)
                .unwrap();
        assert_eq!(minimal, Manifest::new(ArtifactKind::Dataset, "csv"));

        let escaped = Manifest::new(ArtifactKind::Blob, "text")
            .with_metadata_entry("quote", json!("a\"b\\c\nd\te\u{0008}\u{000c}\u{0000}é"))
            .with_metadata_entry("num", json!(-42))
            .with_metadata_entry("flag", json!(false))
            .with_metadata_entry("nil", json!(null));
        let canonical = escaped.canonical_json();
        let parsed: Manifest = serde_json::from_slice(&canonical).unwrap();
        assert_eq!(parsed, escaped);
    }

    #[cfg(feature = "schema")]
    #[test]
    fn manifest_generates_valid_json_schema() {
        let schema = schemars::schema_for!(Manifest);
        let value = serde_json::to_value(&schema).unwrap();
        let root = value.as_object().expect("root schema must be an object");

        assert!(root.contains_key("$schema"));
        let properties = root
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("Manifest schema must define properties");

        for expected_prop in [
            "schema_version",
            "kind",
            "format",
            "flavor",
            "io_schema",
            "metadata",
            "parents",
        ] {
            assert!(
                properties.contains_key(expected_prop),
                "missing property {expected_prop} in Manifest JSON Schema"
            );
        }
        assert!(!properties.contains_key("produced_by"));

        // Verify companion core types also generate valid schemas.
        assert!(serde_json::to_value(schemars::schema_for!(Artifact)).is_ok());
        assert!(serde_json::to_value(schemars::schema_for!(ArtifactId)).is_ok());
        assert!(serde_json::to_value(schemars::schema_for!(Digest)).is_ok());
        assert!(serde_json::to_value(schemars::schema_for!(RelPath)).is_ok());
    }
}
