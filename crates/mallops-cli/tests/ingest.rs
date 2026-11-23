//! End-to-end CLI tests for standard ingest blocks (`std/ingest.csv`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_mallops");

/// Tempdir wrapper that makes read-only store trees writable before cleanup.
struct WorkDir(tempfile::TempDir);

impl WorkDir {
    fn new() -> Self {
        Self(tempfile::tempdir().unwrap())
    }

    fn path(&self) -> &Path {
        self.0.path()
    }

    fn write(&self, rel: &str, content: &str) -> PathBuf {
        let p = self.path().join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, content).unwrap();
        p
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        make_tree_writable(self.path());
    }
}

#[cfg(unix)]
fn make_tree_writable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    let mut stack = vec![path.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let Ok(meta) = fs::symlink_metadata(&cur) else {
            continue;
        };
        let ft = meta.file_type();
        if ft.is_dir() {
            let _ = fs::set_permissions(&cur, fs::Permissions::from_mode(0o755));
            if let Ok(entries) = fs::read_dir(&cur) {
                for entry in entries.flatten() {
                    stack.push(entry.path());
                }
            }
        } else if ft.is_file() {
            let _ = fs::set_permissions(&cur, fs::Permissions::from_mode(0o644));
        }
    }
}

#[cfg(not(unix))]
fn make_tree_writable(path: &Path) {
    let mut stack = vec![path.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let Ok(meta) = fs::symlink_metadata(&cur) else {
            continue;
        };
        if meta.is_file() {
            let mut perms = meta.permissions();
            perms.set_readonly(false);
            let _ = fs::set_permissions(&cur, perms);
        } else if meta.is_dir() {
            if let Ok(entries) = fs::read_dir(&cur) {
                for entry in entries.flatten() {
                    stack.push(entry.path());
                }
            }
        }
    }
}

fn run_cli(cwd: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("failed to execute mallops binary")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("json file must exist")).unwrap()
}

#[test]
fn ingest_e2e_csv_relative_path_cache_and_invalidation() {
    let proj = WorkDir::new();
    let caller_cwd = WorkDir::new();

    proj.write("data.csv", "id,name,score\n1,alice,9.5\n2,bob,8.0\n");
    let yaml = proj.write(
        "mallops.yaml",
        r#"version: 1
name: ingest-e2e
store:
  kind: local
  path: ./.mallops
blocks:
  raw:
    use: std/ingest.csv
    with:
      path: data.csv
"#,
    );
    let yaml_arg = yaml.to_str().unwrap();

    // 1. First run from a different cwd: relative `data.csv` and `store.path` resolve against yaml dir.
    let first = run_cli(caller_cwd.path(), &["-f", yaml_arg, "run"]);
    assert_eq!(first.status.code(), Some(0), "stderr: {}", stderr(&first));
    let out1 = stdout(&first);
    assert!(
        out1.contains("raw succeeded ") && !out1.contains("(cached)"),
        "stdout:\n{out1}"
    );
    assert!(
        out1.contains(": 1 executed, 0 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout:\n{out1}"
    );
    assert!(
        !caller_cwd.path().join(".mallops").exists(),
        "store must be created in yaml directory, not caller cwd"
    );

    let latest_path = proj.path().join(".mallops/meta/runs/latest.json");
    let latest1 = read_json(&latest_path);
    assert_eq!(latest1["v"], 1);
    assert_eq!(latest1["pipeline"], "ingest-e2e");
    let id1 = latest1["outputs"]["raw"]["dataset"]
        .as_str()
        .expect("dataset artifact id in latest.json")
        .to_owned();

    let artifact_dir1 = proj.path().join(".mallops/artifacts").join(&id1);
    let manifest1 = read_json(&artifact_dir1.join("manifest.json"));
    assert_eq!(manifest1["kind"], "dataset");
    assert_eq!(manifest1["format"], "parquet");
    let payload1 = artifact_dir1.join("payload/data.parquet");
    assert!(
        payload1.is_file() && fs::metadata(&payload1).unwrap().len() > 0,
        "expected non-empty payload/data.parquet"
    );

    let verify1 = run_cli(
        caller_cwd.path(),
        &["-f", yaml_arg, "store", "verify", &id1],
    );
    assert_eq!(
        verify1.status.code(),
        Some(0),
        "stderr: {}",
        stderr(&verify1)
    );

    // 2. Second run with unchanged CSV: 0 executed, served from cache with same id.
    let second = run_cli(caller_cwd.path(), &["-f", yaml_arg, "run"]);
    assert_eq!(second.status.code(), Some(0), "stderr: {}", stderr(&second));
    let out2 = stdout(&second);
    assert!(out2.contains("raw succeeded (cached) "), "stdout:\n{out2}");
    assert!(
        out2.contains(": 0 executed, 1 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout:\n{out2}"
    );
    let latest2 = read_json(&latest_path);
    assert_eq!(
        latest2["outputs"]["raw"]["dataset"].as_str(),
        Some(id1.as_str())
    );

    // 3. Modify data.csv: third run re-executes the node and produces a new artifact id.
    proj.write(
        "data.csv",
        "id,name,score\n1,alice,9.5\n2,bob,8.0\n3,carol,7.5\n",
    );
    let third = run_cli(caller_cwd.path(), &["-f", yaml_arg, "run"]);
    assert_eq!(third.status.code(), Some(0), "stderr: {}", stderr(&third));
    let out3 = stdout(&third);
    assert!(
        out3.contains("raw succeeded ") && !out3.contains("(cached)"),
        "stdout:\n{out3}"
    );
    assert!(
        out3.contains(": 1 executed, 0 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout:\n{out3}"
    );

    let latest3 = read_json(&latest_path);
    let id2 = latest3["outputs"]["raw"]["dataset"]
        .as_str()
        .expect("updated dataset artifact id in latest.json");
    assert_ne!(id2, id1, "modified CSV must produce a new artifact id");

    let manifest2 = read_json(
        &proj
            .path()
            .join(".mallops/artifacts")
            .join(id2)
            .join("manifest.json"),
    );
    assert_eq!(manifest2["kind"], "dataset");
    assert_eq!(manifest2["format"], "parquet");

    let verify2 = run_cli(caller_cwd.path(), &["-f", yaml_arg, "store", "verify", id2]);
    assert_eq!(
        verify2.status.code(),
        Some(0),
        "stderr: {}",
        stderr(&verify2)
    );
}

#[test]
fn ingest_e2e_csv_schema_mismatch_exits_1() {
    let proj = WorkDir::new();
    let caller_cwd = WorkDir::new();

    proj.write("data.csv", "id,name\n1,alice\n");
    let yaml = proj.write(
        "mallops.yaml",
        r#"version: 1
name: ingest-bad-schema
store:
  kind: local
  path: ./.mallops
blocks:
  raw:
    use: std/ingest.csv
    with:
      path: data.csv
      schema:
        - name: id
          type: int64
        - name: wrong_column
          type: string
"#,
    );
    let yaml_arg = yaml.to_str().unwrap();

    let out = run_cli(caller_cwd.path(), &["-f", yaml_arg, "run"]);
    assert_eq!(out.status.code(), Some(1), "stdout:\n{}", stdout(&out));
    assert!(
        stdout(&out).contains("raw failed ") && stdout(&out).contains("schema mismatch"),
        "stdout:\n{}",
        stdout(&out)
    );
    assert!(
        stderr(&out).contains("run failed"),
        "stderr:\n{}",
        stderr(&out)
    );
    assert!(
        !proj.path().join(".mallops/meta/runs/latest.json").exists(),
        "failed run must not write runs/latest.json"
    );
}
