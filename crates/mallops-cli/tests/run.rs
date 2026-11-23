//! Integration tests for `mallops run` and `mallops store verify` (task 10.2).

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

fn run_cli(dir: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .current_dir(dir)
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

/// Two-node DAG where `z` runs before `a` (topological order != alphabetical order).
const TWO_NODE_PIPELINE: &str = r#"version: 1
name: two-node
blocks:
  z:
    use: core/noop
  a:
    use: core/noop
    in:
      in: z.out
"#;

#[test]
fn run_init_validate_run_and_cached_rerun() {
    let wd = WorkDir::new();

    let init = run_cli(wd.path(), &["init"]);
    assert_eq!(init.status.code(), Some(0), "stderr: {}", stderr(&init));

    let validate = run_cli(wd.path(), &["validate"]);
    assert_eq!(
        validate.status.code(),
        Some(0),
        "stderr: {}",
        stderr(&validate)
    );

    let first = run_cli(wd.path(), &["run"]);
    assert_eq!(first.status.code(), Some(0), "stderr: {}", stderr(&first));
    assert!(
        stdout(&first).contains(": 2 executed, 0 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout:\n{}",
        stdout(&first)
    );

    let second = run_cli(wd.path(), &["run"]);
    assert_eq!(second.status.code(), Some(0), "stderr: {}", stderr(&second));
    assert!(
        stdout(&second).contains(": 0 executed, 2 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout:\n{}",
        stdout(&second)
    );
}

#[test]
fn run_executes_and_second_run_uses_cache() {
    let wd = WorkDir::new();
    wd.write("mallops.yaml", TWO_NODE_PIPELINE);

    let first = run_cli(wd.path(), &["run"]);
    assert!(
        first.status.success(),
        "first run failed: {}",
        stderr(&first)
    );
    let out1 = stdout(&first);
    let lines1: Vec<&str> = out1.lines().collect();
    assert_eq!(lines1.len(), 3, "stdout:\n{out1}");
    // Topological order: `z` precedes `a`.
    assert!(lines1[0].starts_with("z succeeded "), "stdout:\n{out1}");
    assert!(!lines1[0].contains("(cached)"), "stdout:\n{out1}");
    assert!(lines1[1].starts_with("a succeeded "), "stdout:\n{out1}");
    assert!(!lines1[1].contains("(cached)"), "stdout:\n{out1}");
    assert!(
        lines1[2].starts_with("run ")
            && lines1[2].ends_with(": 2 executed, 0 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout:\n{out1}"
    );

    let second = run_cli(wd.path(), &["run"]);
    assert!(
        second.status.success(),
        "second run failed: {}",
        stderr(&second)
    );
    let out2 = stdout(&second);
    let lines2: Vec<&str> = out2.lines().collect();
    assert_eq!(lines2.len(), 3, "stdout:\n{out2}");
    assert!(
        lines2[0].starts_with("z succeeded (cached) "),
        "stdout:\n{out2}"
    );
    assert!(
        lines2[1].starts_with("a succeeded (cached) "),
        "stdout:\n{out2}"
    );
    assert!(
        lines2[2].starts_with("run ")
            && lines2[2].ends_with(": 0 executed, 2 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout:\n{out2}"
    );
}

#[test]
fn run_no_cache_and_jobs_one() {
    let wd = WorkDir::new();
    wd.write("mallops.yaml", TWO_NODE_PIPELINE);

    let first = run_cli(wd.path(), &["run", "--jobs", "1"]);
    assert!(first.status.success(), "stderr: {}", stderr(&first));

    let second = run_cli(wd.path(), &["run", "--no-cache", "--jobs", "1"]);
    assert!(second.status.success(), "stderr: {}", stderr(&second));
    let out = stdout(&second);
    assert!(!out.contains("(cached)"), "stdout:\n{out}");
    assert!(
        out.contains(": 2 executed, 0 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout:\n{out}"
    );
}

#[test]
fn run_records_latest() {
    let wd = WorkDir::new();
    wd.write("mallops.yaml", TWO_NODE_PIPELINE);

    let ok_run = run_cli(wd.path(), &["run"]);
    assert!(ok_run.status.success(), "stderr: {}", stderr(&ok_run));

    let latest_path = wd.path().join(".mallops/meta/runs/latest.json");
    let before_bytes = fs::read(&latest_path).expect("runs/latest.json must exist");
    let latest: Value = serde_json::from_slice(&before_bytes).unwrap();
    assert_eq!(latest["v"], 1);
    assert_eq!(latest["pipeline"], "two-node");
    let run_id = latest["run"].as_str().expect("run id string");
    assert!(!run_id.is_empty());
    assert!(
        stdout(&ok_run).contains(&format!("run {run_id}:")),
        "stdout: {}",
        stdout(&ok_run)
    );

    let outputs = latest["outputs"].as_object().expect("outputs object");
    assert_eq!(outputs.len(), 2);
    for node in ["a", "z"] {
        let id = outputs[node]["out"]
            .as_str()
            .expect("artifact id for out port");
        let verify = run_cli(wd.path(), &["store", "verify", id]);
        assert!(
            verify.status.success(),
            "verify {node}.out ({id}) failed: {}",
            stderr(&verify)
        );
        assert_eq!(stdout(&verify).trim(), format!("ok: {id}"));
    }

    // A run with a skipped node exits 0 but must leave runs/latest.json unchanged.
    wd.write(
        "mallops.yaml",
        r#"version: 1
name: two-node
blocks:
  z:
    use: core/noop
    with:
      skip: "gate closed"
  a:
    use: core/noop
    in:
      in: z.out
"#,
    );
    let skipped = run_cli(wd.path(), &["run"]);
    assert_eq!(
        skipped.status.code(),
        Some(0),
        "stderr: {}",
        stderr(&skipped)
    );
    let skip_out = stdout(&skipped);
    assert!(
        skip_out.contains("z skipped ") && skip_out.contains("gate closed"),
        "stdout:\n{skip_out}"
    );
    assert!(
        skip_out.contains(": 0 executed, 1 cached, 0 failed, 1 skipped, 0 cancelled"),
        "stdout:\n{skip_out}"
    );
    assert!(
        skip_out.contains("latest.json not updated: 1 skipped"),
        "stdout:\n{skip_out}"
    );
    assert_eq!(
        before_bytes,
        fs::read(&latest_path).unwrap(),
        "skipped run must leave runs/latest.json unchanged"
    );

    // A run with a failing node and a downstream cancelled node exits 1 and leaves latest.json unchanged.
    wd.write(
        "mallops.yaml",
        r#"version: 1
name: two-node
blocks:
  z:
    use: core/noop
    with:
      fail: "boom"
  a:
    use: core/noop
    in:
      in: z.out
"#,
    );
    let failed = run_cli(wd.path(), &["run"]);
    assert_eq!(failed.status.code(), Some(1), "stderr: {}", stderr(&failed));
    let fail_out = stdout(&failed);
    assert!(fail_out.contains("z failed "), "stdout:\n{fail_out}");
    assert!(fail_out.contains("boom"), "stdout:\n{fail_out}");
    assert!(fail_out.contains("a cancelled "), "stdout:\n{fail_out}");
    assert!(
        fail_out.contains(": 0 executed, 0 cached, 1 failed, 0 skipped, 1 cancelled"),
        "stdout:\n{fail_out}"
    );
    assert!(
        stderr(&failed).contains("run failed"),
        "{}",
        stderr(&failed)
    );

    let after_bytes = fs::read(&latest_path).expect("runs/latest.json must remain");
    assert_eq!(
        before_bytes, after_bytes,
        "failed run must leave runs/latest.json unchanged"
    );
}

#[test]
fn run_store_verify_ok_and_detects_tampering() {
    let wd = WorkDir::new();
    wd.write("mallops.yaml", TWO_NODE_PIPELINE);

    // Before any run, `store verify` on a missing local store fails without creating `.mallops`.
    let missing_id = "0".repeat(64);
    let no_store = run_cli(wd.path(), &["store", "verify", &missing_id]);
    assert_eq!(no_store.status.code(), Some(1));
    assert!(
        stderr(&no_store).contains("store not found:"),
        "stderr: {}",
        stderr(&no_store)
    );
    assert!(
        !wd.path().join(".mallops").exists(),
        "store verify must not create a missing local store directory"
    );

    let out = run_cli(wd.path(), &["run"]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let latest: Value = serde_json::from_slice(
        &fs::read(wd.path().join(".mallops/meta/runs/latest.json")).unwrap(),
    )
    .unwrap();
    let id = latest["outputs"]["z"]["out"].as_str().unwrap();

    // Intact artifact verifies cleanly.
    let ok = run_cli(wd.path(), &["store", "verify", id]);
    assert!(ok.status.success(), "stderr: {}", stderr(&ok));
    assert_eq!(stdout(&ok).trim(), format!("ok: {id}"));

    // Invalid ArtifactId exits 1.
    let invalid = run_cli(wd.path(), &["store", "verify", "not-a-valid-id"]);
    assert_eq!(invalid.status.code(), Some(1));

    // Missing ArtifactId in an existing store exits 1.
    let missing = run_cli(wd.path(), &["store", "verify", &missing_id]);
    assert_eq!(missing.status.code(), Some(1));

    // Tamper with payload/data on disk (stored read-only 0444 on Unix).
    let payload_file = wd
        .path()
        .join(".mallops/artifacts")
        .join(id)
        .join("payload/data");
    make_tree_writable(&payload_file);
    fs::write(&payload_file, b"tampered").unwrap();

    let tampered = run_cli(wd.path(), &["store", "verify", id]);
    assert!(
        !tampered.status.success(),
        "verify must fail after tampering; stdout: {}",
        stdout(&tampered)
    );
}

/// Unix only: on Windows `Command::new("git")` resolves only `git.exe`, not a script shim.
#[cfg(unix)]
#[test]
fn run_spawns_git() {
    let wd = WorkDir::new();
    let fake_bin = wd.path().join("fake-bin");
    fs::create_dir_all(&fake_bin).unwrap();
    let marker = wd.path().join("git-ran.marker");
    let fake_commit = "0123456789abcdef0123456789abcdef01234567";

    {
        use std::os::unix::fs::PermissionsExt;
        let script = fake_bin.join("git");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\necho ran > \"{}\"\necho {fake_commit}\n",
                marker.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    }
    // Place mallops.yaml in a nested directory to also verify relative store.path resolution.
    let yaml = wd.write("proj/mallops.yaml", TWO_NODE_PIPELINE);
    let out = Command::new(BIN)
        .current_dir(wd.path())
        .env("PATH", &fake_bin)
        .args(["-f", yaml.to_str().unwrap(), "run"])
        .output()
        .expect("failed to execute mallops binary");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(marker.exists(), "expected fake git marker to be created");

    let latest_path = wd.path().join("proj/.mallops/meta/runs/latest.json");
    let latest: Value = serde_json::from_slice(&fs::read(&latest_path).unwrap()).unwrap();
    let run_id = latest["run"].as_str().unwrap();

    let lineage_path = wd
        .path()
        .join(format!("proj/.mallops/meta/runs/{run_id}/lineage.json"));
    let lineage: Value = serde_json::from_slice(&fs::read(&lineage_path).unwrap()).unwrap();
    let events = lineage["events"].as_array().unwrap();
    let started = events
        .iter()
        .find(|e| e["event"] == "run_started")
        .expect("run_started event");
    assert_eq!(started["git_commit"], fake_commit);
}

#[test]
fn run_lineage_off_writes_no_lineage_file() {
    let wd = WorkDir::new();
    wd.write(
        "mallops.yaml",
        r#"version: 1
name: no-lineage
lineage: off
blocks:
  only:
    use: core/noop
"#,
    );

    let out = run_cli(wd.path(), &["run"]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));

    let latest_path = wd.path().join(".mallops/meta/runs/latest.json");
    let latest: Value = serde_json::from_slice(&fs::read(&latest_path).unwrap()).unwrap();
    let run_id = latest["run"].as_str().unwrap();
    let lineage_path = wd
        .path()
        .join(format!(".mallops/meta/runs/{run_id}/lineage.json"));
    assert!(
        !lineage_path.exists(),
        "lineage: off must not write lineage.json"
    );
}

#[test]
fn run_memory_store_succeeds_without_creating_disk_store() {
    let wd = WorkDir::new();
    wd.write(
        "mallops.yaml",
        r#"version: 1
name: mem-only
store:
  kind: memory
blocks:
  only:
    use: core/noop
"#,
    );

    let out = run_cli(wd.path(), &["run"]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(
        stdout(&out).contains(": 1 executed, 0 cached, 0 failed, 0 skipped, 0 cancelled"),
        "stdout: {}",
        stdout(&out)
    );
    assert!(!wd.path().join(".mallops").exists());
}

#[test]
fn run_rejects_unknown_store_and_lineage_backends() {
    let wd = WorkDir::new();
    wd.write(
        "mallops.yaml",
        r#"version: 1
name: bad-store
store:
  kind: s3
blocks:
  only:
    use: core/noop
"#,
    );

    let run_out = run_cli(wd.path(), &["run"]);
    assert_eq!(run_out.status.code(), Some(1));
    assert!(stderr(&run_out).contains("s3"), "{}", stderr(&run_out));

    let fake_id = "0".repeat(64);
    let verify_out = run_cli(wd.path(), &["store", "verify", &fake_id]);
    assert_eq!(verify_out.status.code(), Some(1));
    assert!(
        stderr(&verify_out).contains("s3"),
        "{}",
        stderr(&verify_out)
    );

    wd.write(
        "mallops.yaml",
        r#"version: 1
name: bad-lineage
lineage:
  - sink: openlineage
blocks:
  only:
    use: core/noop
"#,
    );
    let lineage_out = run_cli(wd.path(), &["run"]);
    assert_eq!(lineage_out.status.code(), Some(1));
    assert!(
        stderr(&lineage_out).contains("openlineage"),
        "{}",
        stderr(&lineage_out)
    );
}
