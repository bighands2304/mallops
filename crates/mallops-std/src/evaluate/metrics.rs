//! Pure metric functions of `std/evaluate.*` (no I/O, no ML library).
//!
//! Conventions follow scikit-learn: a division by zero yields `0` for precision/recall/F1
//! (`zero_division=0`), R² of a constant target is `1` for a perfect fit and `0` otherwise
//! (`force_finite`), and the default positive class of a binary task is the larger label.

use std::collections::{BTreeMap, BTreeSet};

use mallops_core::Error;
use serde_json::Value;

/// Metrics of a classification task, in the order they are documented.
pub(crate) const CLASSIFICATION_METRICS: &[&str] =
    &["accuracy", "precision", "recall", "f1", "roc_auc"];
/// Metrics of a regression task.
pub(crate) const REGRESSION_METRICS: &[&str] = &["rmse", "mae", "r2"];

/// A class label. Variants are never mixed within one evaluation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Label {
    I64(i64),
    Bool(bool),
    Str(String),
}

impl Label {
    /// Parses a JSON scalar (integer, boolean or string); `None` for anything else.
    pub(crate) fn from_json(value: &Value) -> Option<Self> {
        match value {
            Value::Number(n) => n.as_i64().map(Self::I64),
            Value::Bool(b) => Some(Self::Bool(*b)),
            Value::String(s) => Some(Self::Str(s.clone())),
            _ => None,
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        match self {
            Self::I64(v) => Value::from(*v),
            Self::Bool(v) => Value::from(*v),
            Self::Str(v) => Value::from(v.as_str()),
        }
    }
}

/// Result of [`classification`].
#[derive(Debug, PartialEq)]
pub(crate) struct ClassificationMetrics {
    /// Metric name → finite value; only the metrics that could be computed.
    pub(crate) values: BTreeMap<String, f64>,
    /// The positive class, for binary tasks.
    pub(crate) positive: Option<Label>,
}

fn ratio(num: f64, den: f64) -> f64 {
    if den == 0.0 { 0.0 } else { num / den }
}

fn f1(precision: f64, recall: f64) -> f64 {
    ratio(2.0 * precision * recall, precision + recall)
}

/// `(precision, recall, f1)` of `class` as the positive class.
fn class_prf(y_true: &[Label], y_pred: &[Label], class: &Label) -> (f64, f64, f64) {
    let (mut tp, mut fp, mut fneg) = (0.0, 0.0, 0.0);
    for (t, p) in y_true.iter().zip(y_pred) {
        match (t == class, p == class) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (true, false) => fneg += 1.0,
            (false, false) => {}
        }
    }
    let (precision, recall) = (ratio(tp, tp + fp), ratio(tp, tp + fneg));
    (precision, recall, f1(precision, recall))
}

/// Area under the ROC curve by ranks (Mann–Whitney U, ties get the average rank).
/// `None` unless both a positive and a negative sample exist.
fn roc_auc(scores: &[f64], is_pos: &[bool]) -> Option<f64> {
    let n_pos = is_pos.iter().filter(|p| **p).count() as f64;
    let n_neg = is_pos.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return None;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        rank_sum_pos += order[i..=j].iter().filter(|&&k| is_pos[k]).count() as f64 * avg_rank;
        i = j + 1;
    }
    Some((rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))
}

fn ensure_finite(values: &BTreeMap<String, f64>) -> Result<(), Error> {
    match values.iter().find(|(_, v)| !v.is_finite()) {
        Some((name, v)) => Err(Error::Runtime(format!(
            "metric `{name}` is not finite ({v}); check the predictions"
        ))),
        None => Ok(()),
    }
}

/// Classification metrics (D15-6).
///
/// Classes are the sorted distinct values of `y_true ∪ y_pred ∪ {positive}`. Up to two classes
/// the task is binary and precision/recall/F1 refer to the positive class (`positive`, else the
/// larger class); more classes are macro-averaged and `positive` is rejected. `proba[i]` is
/// P(larger class) for row `i`; `roc_auc` is computed only for a binary task with `proba` and
/// both classes present in `y_true`.
///
/// # Errors
///
/// [`Error::Runtime`] on empty or mismatched inputs, mixed label types, a binary task whose
/// positive class cannot be inferred, `positive` on a multiclass task, or a non-finite metric.
pub(crate) fn classification(
    y_true: &[Label],
    y_pred: &[Label],
    proba: Option<&[f64]>,
    positive: Option<&Label>,
) -> Result<ClassificationMetrics, Error> {
    let n = y_true.len();
    if n == 0 || y_pred.len() != n || proba.is_some_and(|p| p.len() != n) {
        return Err(Error::Runtime(
            "classification needs the same non-zero number of labels and predictions".into(),
        ));
    }
    let mut classes: BTreeSet<&Label> = y_true.iter().chain(y_pred).chain(positive).collect();
    let first = classes.first().map(|l| std::mem::discriminant(*l));
    if classes
        .iter()
        .any(|l| Some(std::mem::discriminant(*l)) != first)
    {
        return Err(Error::Runtime(
            "labels, predictions and `positive` must all have the same type".into(),
        ));
    }
    let correct = y_true.iter().zip(y_pred).filter(|(t, p)| t == p).count();
    let mut values = BTreeMap::from([("accuracy".to_owned(), correct as f64 / n as f64)]);

    let positive_class = if classes.len() <= 2 {
        let larger = classes.pop_last().cloned();
        let pos = positive.cloned().or_else(|| {
            // A lone class is ambiguous: sklearn would assume label 1, we refuse to guess.
            if classes.is_empty() {
                None
            } else {
                larger.clone()
            }
        });
        let Some(pos) = pos else {
            return Err(Error::Runtime(
                "cannot infer the positive class; set with.positive".into(),
            ));
        };
        let (p, r, f) = class_prf(y_true, y_pred, &pos);
        values.extend([
            ("precision".to_owned(), p),
            ("recall".to_owned(), r),
            ("f1".to_owned(), f),
        ]);
        if let (Some(proba), Some(larger)) = (proba, larger) {
            let flip = pos != larger;
            let scores: Vec<f64> = proba
                .iter()
                .map(|p| if flip { 1.0 - p } else { *p })
                .collect();
            let is_pos: Vec<bool> = y_true.iter().map(|t| *t == pos).collect();
            if let Some(auc) = roc_auc(&scores, &is_pos) {
                values.insert("roc_auc".to_owned(), auc);
            }
        }
        Some(pos)
    } else {
        if positive.is_some() {
            return Err(Error::Runtime(
                "with.positive is only valid for binary classification".into(),
            ));
        }
        let (mut p, mut r, mut f) = (0.0, 0.0, 0.0);
        for class in &classes {
            let (cp, cr, cf) = class_prf(y_true, y_pred, class);
            (p, r, f) = (p + cp, r + cr, f + cf);
        }
        let k = classes.len() as f64;
        values.extend([
            ("precision".to_owned(), p / k),
            ("recall".to_owned(), r / k),
            ("f1".to_owned(), f / k),
        ]);
        None
    };
    ensure_finite(&values)?;
    Ok(ClassificationMetrics {
        values,
        positive: positive_class,
    })
}

/// Regression metrics: `rmse`, `mae`, `r2`.
///
/// # Errors
///
/// [`Error::Runtime`] on empty or mismatched inputs, or a non-finite metric (e.g. overflow of
/// the squared error on absurd predictions).
pub(crate) fn regression(y_true: &[f64], y_pred: &[f64]) -> Result<BTreeMap<String, f64>, Error> {
    let n = y_true.len();
    if n == 0 || y_pred.len() != n {
        return Err(Error::Runtime(
            "regression needs the same non-zero number of targets and predictions".into(),
        ));
    }
    let nf = n as f64;
    let mean = y_true.iter().sum::<f64>() / nf;
    let (mut ss_res, mut ss_tot, mut abs) = (0.0, 0.0, 0.0);
    for (t, p) in y_true.iter().zip(y_pred) {
        ss_res += (t - p) * (t - p);
        ss_tot += (t - mean) * (t - mean);
        abs += (t - p).abs();
    }
    let r2 = if ss_tot == 0.0 {
        if ss_res == 0.0 { 1.0 } else { 0.0 }
    } else {
        1.0 - ss_res / ss_tot
    };
    let values = BTreeMap::from([
        ("mae".to_owned(), abs / nf),
        ("r2".to_owned(), r2),
        ("rmse".to_owned(), (ss_res / nf).sqrt()),
    ]);
    ensure_finite(&values)?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: &[i64]) -> Vec<Label> {
        v.iter().map(|x| Label::I64(*x)).collect()
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn binary_default_positive_is_larger_class() {
        let m = classification(&ints(&[0, 0, 1, 1]), &ints(&[0, 1, 1, 1]), None, None).unwrap();
        assert_eq!(m.positive, Some(Label::I64(1)));
        close(m.values["accuracy"], 0.75);
        close(m.values["precision"], 2.0 / 3.0);
        close(m.values["recall"], 1.0);
        close(m.values["f1"], 0.8);
        assert!(!m.values.contains_key("roc_auc"));
    }

    #[test]
    fn binary_positive_override_and_auc_flip() {
        let proba = [0.1, 0.6, 0.4, 0.9]; // P(class 1)
        let y = ints(&[0, 0, 1, 1]);
        let p = ints(&[0, 1, 1, 1]);
        let m = classification(&y, &p, Some(&proba), None).unwrap();
        close(m.values["roc_auc"], 0.75);
        let m = classification(&y, &p, Some(&proba), Some(&Label::I64(0))).unwrap();
        close(m.values["precision"], 1.0);
        close(m.values["recall"], 0.5);
        close(m.values["f1"], 2.0 / 3.0);
        close(m.values["roc_auc"], 0.75); // symmetric: flipping scores and classes
    }

    #[test]
    fn auc_ties_get_average_rank() {
        let m = classification(
            &ints(&[0, 1, 0, 1]),
            &ints(&[0, 1, 0, 1]),
            Some(&[0.5, 0.5, 0.5, 0.5]),
            None,
        )
        .unwrap();
        close(m.values["roc_auc"], 0.5);
    }

    #[test]
    fn auc_omitted_when_one_class_in_truth() {
        let m = classification(&ints(&[1, 1]), &ints(&[0, 1]), Some(&[0.2, 0.9]), None).unwrap();
        assert!(!m.values.contains_key("roc_auc"));
    }

    #[test]
    fn multiclass_is_macro_averaged() {
        let m = classification(&ints(&[0, 1, 2, 2]), &ints(&[0, 2, 2, 2]), None, None).unwrap();
        assert_eq!(m.positive, None);
        close(m.values["accuracy"], 0.75);
        close(m.values["precision"], 5.0 / 9.0);
        close(m.values["recall"], 2.0 / 3.0);
        close(m.values["f1"], 0.6);
        assert!(!m.values.contains_key("roc_auc"));
        let e = classification(
            &ints(&[0, 1, 2]),
            &ints(&[0, 1, 2]),
            None,
            Some(&Label::I64(1)),
        );
        assert!(matches!(e, Err(Error::Runtime(_))));
    }

    #[test]
    fn string_and_bool_labels() {
        let s = |v: &[&str]| {
            v.iter()
                .map(|x| Label::Str((*x).into()))
                .collect::<Vec<_>>()
        };
        let m = classification(&s(&["a", "b", "b"]), &s(&["a", "b", "a"]), None, None).unwrap();
        assert_eq!(m.positive, Some(Label::Str("b".into())));
        close(m.values["recall"], 0.5);
        let b = [Label::Bool(false), Label::Bool(true)];
        let m = classification(&b, &b, None, None).unwrap();
        close(m.values["f1"], 1.0);
    }

    #[test]
    fn single_class_needs_positive() {
        let y = ints(&[0, 0]);
        let e = classification(&y, &y, None, None).unwrap_err();
        assert!(e.to_string().contains("with.positive"), "{e}");
        // an explicit positive that never occurs is a class too
        let m = classification(&y, &y, None, Some(&Label::I64(1))).unwrap();
        close(m.values["precision"], 0.0);
        close(m.values["accuracy"], 1.0);
    }

    #[test]
    fn mixed_label_types_are_errors() {
        let e = classification(
            &ints(&[0, 1]),
            &[Label::Bool(false), Label::Bool(true)],
            None,
            None,
        );
        assert!(matches!(e, Err(Error::Runtime(_))));
        let e = classification(
            &ints(&[0, 1]),
            &ints(&[0, 1]),
            None,
            Some(&Label::Str("1".into())),
        );
        assert!(matches!(e, Err(Error::Runtime(_))));
    }

    #[test]
    fn regression_analytic() {
        let m = regression(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 6.0]).unwrap();
        close(m["mae"], 0.5);
        close(m["rmse"], 1.0);
        close(m["r2"], 1.0 - 4.0 / 5.0);
    }

    #[test]
    fn regression_constant_target_r2() {
        close(regression(&[2.0, 2.0], &[2.0, 2.0]).unwrap()["r2"], 1.0);
        close(regression(&[2.0, 2.0], &[2.0, 3.0]).unwrap()["r2"], 0.0);
    }

    #[test]
    fn regression_overflow_is_an_error() {
        let e = regression(&[1.0, 2.0], &[1e200, -1e200]).unwrap_err();
        assert!(matches!(e, Error::Runtime(_)), "{e}");
    }
}
