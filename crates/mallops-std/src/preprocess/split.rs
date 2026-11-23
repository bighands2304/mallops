//! Seeded deterministic row splitting into `train` / optional `validation` / `test`.
//!
//! Uses an internal SplitMix64 PRNG and Fisher-Yates shuffle (no external RNG crate):
//! the first `floor(n * test)` indices of the permutation form `test`, the next
//! `floor(n * validation)` form `validation` (if configured), and the remainder form `train`.
//! Keeping `test` first ensures the `test` partition is unchanged when `validation` is toggled
//! with the same seed. Each partition is sorted by original row index so relative row order is
//! preserved.

use std::ops::Range;

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

impl SplitBatches {
    /// Applies `f` to every partition (`train`, `test`, and `validation` if present).
    pub(crate) fn map<F>(&mut self, mut f: F) -> Result<(), Error>
    where
        F: FnMut(&RecordBatch) -> Result<RecordBatch, Error>,
    {
        self.train = f(&self.train)?;
        self.test = f(&self.test)?;
        if let Some(val) = self.validation.as_ref() {
            self.validation = Some(f(val)?);
        }
        Ok(())
    }

    /// Ensures `train`, `test`, and any configured `validation` partition are non-empty.
    pub(crate) fn check_non_empty(&self) -> Result<(), Error> {
        ensure_non_empty("train", self.train.num_rows())?;
        ensure_non_empty("test", self.test.num_rows())?;
        if let Some(val) = &self.validation {
            ensure_non_empty("validation", val.num_rows())?;
        }
        Ok(())
    }
}

/// Returns [`Error::Runtime`] if `rows == 0` for the named partition.
pub(crate) fn ensure_non_empty(partition: &str, rows: usize) -> Result<(), Error> {
    if rows == 0 {
        return Err(Error::Runtime(format!(
            "empty `{partition}` partition (0 rows)"
        )));
    }
    Ok(())
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
    ensure_non_empty("test", test_len)?;

    let val_len = match step.validation {
        Some(ratio) => {
            let len = ((n as f64) * ratio).floor() as usize;
            ensure_non_empty("validation", len)?;
            len
        }
        None => 0,
    };

    ensure_non_empty("train", n.saturating_sub(test_len + val_len))?;

    let mut perm: Vec<u64> = (0..n as u64).collect();
    let mut rng = SplitMix64::new(step.seed);
    for i in (1..n).rev() {
        let j = rng.gen_range(i + 1);
        perm.swap(i, j);
    }

    let slice = |range: Range<usize>| -> Result<RecordBatch, Error> {
        let mut idx = perm[range].to_vec();
        idx.sort_unstable();
        take_rows(batch, idx)
    };

    let test = slice(0..test_len)?;
    let validation = step
        .validation
        .map(|_| slice(test_len..test_len + val_len))
        .transpose()?;
    let train = slice(test_len + val_len..n)?;

    Ok(SplitBatches {
        train,
        validation,
        test,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{ArrayRef, Int64Array};

    use super::*;

    fn id_batch(n: usize) -> RecordBatch {
        let ids: Vec<i64> = (0..n as i64).collect();
        RecordBatch::try_from_iter([("id", Arc::new(Int64Array::from(ids)) as ArrayRef)]).unwrap()
    }

    fn ids(batch: &RecordBatch) -> Vec<i64> {
        batch
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec()
    }

    #[test]
    fn preprocess_steps_split_disjoint_complete_and_floor_counts() {
        let b = id_batch(15);
        let all: BTreeSet<i64> = (0..15).collect();

        // floor(15 * 0.2) = 3 test, floor(15 * 0.25) = 3 validation, 9 train.
        let s = split_batch(
            &b,
            SplitStep {
                test: 0.2,
                validation: Some(0.25),
                seed: 7,
            },
        )
        .unwrap();
        let train = ids(&s.train);
        let val = ids(s.validation.as_ref().unwrap());
        let test = ids(&s.test);

        assert_eq!(test.len(), 3);
        assert_eq!(val.len(), 3);
        assert_eq!(train.len(), 9);
        assert!(train.windows(2).all(|w| w[0] < w[1]));
        assert!(val.windows(2).all(|w| w[0] < w[1]));
        assert!(test.windows(2).all(|w| w[0] < w[1]));

        let mut seen = BTreeSet::new();
        for id in train.iter().chain(&val).chain(&test) {
            assert!(seen.insert(*id), "duplicate id {id}");
        }
        assert_eq!(seen, all);

        // Without validation, same seed produces the exact same test partition.
        let s2 = split_batch(
            &b,
            SplitStep {
                test: 0.2,
                validation: None,
                seed: 7,
            },
        )
        .unwrap();
        assert_eq!(ids(&s2.test), test);
        assert!(s2.validation.is_none());
        assert_eq!(s2.train.num_rows(), 12);
    }
}
