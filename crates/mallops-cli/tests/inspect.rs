//! Integration and snapshot tests for `init`, `validate`, `graph`, and `blocks list`.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use insta::assert_snapshot;

const BIN: &str = env!("CARGO_BIN_EXE_mallops");

fn run_cli(dir: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .current_dir(dir)
        .args(args)
        .output()
        .expect("failed to run mallops binary")
}

#[test]
fn inspect_init_creates_runnable_yaml_and_refuses_overwrite() {
    let dir = tempfile::tempdir().unwrap();

    let out = run_cli(dir.path(), &["init"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("created "), "unexpected stdout: {stdout}");
    let yaml_path = dir.path().join("mallops.yaml");
    assert!(yaml_path.is_file());
    let original = fs::read_to_string(&yaml_path).unwrap();

    // Validate the generated pipeline.
    let val = run_cli(dir.path(), &["validate"]);
    assert_eq!(
        val.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&val.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&val.stdout),
        "ok: hello (2 blocks)\n"
    );

    // Re-running init must fail with exit 1 and never overwrite.
    let again = run_cli(dir.path(), &["init"]);
    assert_eq!(again.status.code(), Some(1));
    assert_eq!(fs::read_to_string(&yaml_path).unwrap(), original);
}

#[test]
fn inspect_graph_mermaid_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let init = run_cli(dir.path(), &["init"]);
    assert_eq!(init.status.code(), Some(0));

    let out = run_cli(dir.path(), &["graph", "--format", "mermaid"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_snapshot!("graph_mermaid", String::from_utf8(out.stdout).unwrap());
}

#[test]
fn inspect_graph_dot_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let init = run_cli(dir.path(), &["init"]);
    assert_eq!(init.status.code(), Some(0));

    let out = run_cli(dir.path(), &["graph", "--format", "dot"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_snapshot!("graph_dot", String::from_utf8(out.stdout).unwrap());
}

#[test]
fn inspect_blocks_list_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_cli(dir.path(), &["blocks", "list"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_snapshot!("blocks_list", String::from_utf8(out.stdout).unwrap());
}

#[test]
fn inspect_validate_errors_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let yaml = "\
version: 1
name: broken

blocks:
  good:
    use: core/noop
  unknown_block:
    use: std/missing
  bad_input:
    use: core/noop
    in:
      in: nonexistent.out
      extra: good.out
";
    fs::write(dir.path().join("mallops.yaml"), yaml).unwrap();

    let out = run_cli(dir.path(), &["validate"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty(), "expected empty stdout on failure");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert_snapshot!("validate_errors", stderr);
}

#[test]
fn inspect_validate_parse_error_exits_1() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("mallops.yaml"), "version: [not a number\n").unwrap();

    let out = run_cli(dir.path(), &["validate"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("mallops.yaml:"),
        "expected file:line:col in stderr: {stderr}"
    );
}

#[test]
fn inspect_no_execution_on_validate_and_graph() {
    let dir = tempfile::tempdir().unwrap();
    let init = run_cli(dir.path(), &["init"]);
    assert_eq!(init.status.code(), Some(0));

    let fake_bin = tempfile::tempdir().unwrap();
    let marker = dir.path().join("git-was-called.marker");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let git_path = fake_bin.path().join("git");
        let script = format!(
            "#!/bin/sh\necho touched > \"{}\"\nexit 0\n",
            marker.display()
        );
        fs::write(&git_path, script).unwrap();
        let mut perms = fs::metadata(&git_path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&git_path, perms).unwrap();
    }

    #[cfg(windows)]
    {
        let git_path = fake_bin.path().join("git.cmd");
        let script = format!("@echo touched > \"{}\"\r\n@exit /b 0\r\n", marker.display());
        fs::write(&git_path, script).unwrap();
    }

    for args in [
        &["validate"][..],
        &["graph", "--format", "mermaid"][..],
        &["graph", "--format", "dot"][..],
    ] {
        let out = Command::new(BIN)
            .current_dir(dir.path())
            .env("PATH", fake_bin.path())
            .args(args)
            .output()
            .expect("failed to run mallops binary");
        assert_eq!(
            out.status.code(),
            Some(0),
            "args {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    assert!(
        !marker.exists(),
        "validate/graph must never spawn external processes"
    );
    assert!(
        !dir.path().join(".mallops").exists(),
        "validate/graph must never create the artifact store"
    );
}
