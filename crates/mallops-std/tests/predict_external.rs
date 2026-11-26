//! Integration tests of the external Python predictor (`harness = false`, see `Cargo.toml`).
//!
//! With `--worker` this binary acts as the mock worker (filled in by the external predictor task).

fn main() {
    if std::env::args().any(|a| a == "--worker") {
        return;
    }
    println!("0 tests");
}
