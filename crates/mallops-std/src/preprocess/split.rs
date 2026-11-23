//! Seeded deterministic row splitting into `train` / optional `validation` / `test`.
//!
//! Uses an internal SplitMix64 PRNG and Fisher-Yates shuffle (no external RNG crate):
//! the first `floor(n * test)` indices of the permutation form `test`, the next
//! `floor(n * validation)` form `validation` (if configured), and the remainder form `train`.
//! Keeping `test` first ensures the `test` partition is unchanged when `validation` is toggled
//! with the same seed. Each partition is sorted by original row index so relative row order is
//! preserved.

use arrow_array::{RecordBatch, UInt64Array};
use arrow_select::take::take_record_batch;
use mallops_core::Error;

use super::SplitStep;

/// Partitioned batches produced by [`split_batch`].
pub(crate) struct SplitBatches {
    pub(crate) train: RecordBatch,
    pub(crate) validation: Option<RecordBatch>,
    pub(crate) test: RecordBatch,
}

/// Minimal deterministic SplitMix64 PRNG.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Unbiased uniform index in `0..bound` (`bound > 0`).
    fn gen_range(&mut self, bound: usize) -> usize {
        let bound_u64 = bound as u64;
        let threshold = bound_u64.wrapping_neg() % bound_u64;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return (r % bound_u64) as usize;
            }
        }
    }
}

fn take_rows(batch: &RecordBatch, indices: Vec<u64>) -> Result<RecordBatch, Error> {
    let idx = UInt64Array::from(indices);
    take_record_batch(batch, &idx)
        .map_err(|e| Error::Runtime(format!("split: failed to slice rows: {e}")))
}

/// Splits `batch` deterministically according to `step`.
///
/// # Errors
///
/// [`Error::Runtime`] if the dataset is empty or any requested partition (`train`, `test`, or
/// configured `validation`) would have zero rows.
pub(crate) fn split_batch(batch: &RecordBatch, step: SplitStep) -> Result<SplitBatches, Error> {
    let n = batch.num_rows();
    if n == 0 {
        return Err(Error::Runtime(
            "split: cannot split an empty dataset (0 rows)".into(),
        ));
    }

    let test_len = ((n as f64) * step.test).floor() as usize;
    if test_len == 0 {
        return Err(Error::Runtime(format!(
            "split produced an empty `test` partition (rows={n}, test={})",
            step.test
        )));
    }

    let val_len = match step.validation {
        Some(ratio) => {
            let len = ((n as f64) * ratio).floor() as usize;
            if len == 0 {
                return Err(Error::Runtime(format!(
                    "split produced an empty `validation` partition (rows={n}, validation={ratio})"
                )));
            }
            Some(len)
        }
        None => None,
    };

    let taken = test_len.saturating_add(val_len.unwrap_or(0));
    if n <= taken {
        return Err(Error::Runtime(format!(
            "split produced an empty `train` partition (rows={n}, test={}, validation={:?})",
            step.test, step.validation
        )));
    }

    let mut perm: Vec<u64> = (0..n as u64).collect();
    let mut rng = SplitMix64::new(step.seed);
    for i in (1..n).rev() {
        let j = rng.gen_range(i + 1);
        perm.swap(i, j);
    }

    let mut test_idx = perm[..test_len].to_vec();
    test_idx.sort_unstable();

    let (val_idx, train_start) = match val_len {
        Some(vl) => {
            let mut v = perm[test_len..test_len + vl].to_vec();
            v.sort_unstable();
            (Some(v), test_len + vl)
        }
        None => (None, test_len),
    };

    let mut train_idx = perm[train_start..].to_vec();
    train_idx.sort_unstable();

    let train = take_rows(batch, train_idx)?;
    let validation = match val_idx {
        Some(v) => Some(take_rows(batch, v)?),
        None => None,
    };
    let test = take_rows(batch, test_idx)?;

    Ok(SplitBatches {
        train,
        validation,
        test,
    })
}
