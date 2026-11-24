//! Training blocks (`std/train.*`).

#[cfg(feature = "train-external")]
pub(crate) mod external;
#[cfg(feature = "train-smartcore")]
pub(crate) mod smartcore;

/// Whether any string (key or value) in `value` contains `${secret:`; JSON never escapes `$`,
/// `{` or `:`.
pub(crate) fn has_secret_ref(value: &impl serde::Serialize) -> bool {
    serde_json::to_string(value).is_ok_and(|json| json.contains("${secret:"))
}
