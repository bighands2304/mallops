//! Training blocks (`std/train.*`).

#[cfg(feature = "train-external")]
pub(crate) mod external;
#[cfg(feature = "train-smartcore")]
pub(crate) mod smartcore;
