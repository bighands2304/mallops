//! Shared test utilities for CLI integration tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

pub const BIN: &str = env!("CARGO_BIN_EXE_mallops");

/// Tempdir wrapper that makes read-only store trees writable before cleanup.
pub struct WorkDir(tempfile::TempDir);

impl WorkDir {
    pub fn new() -> Self {
        Self(tempfile::tempdir().unwrap())
    }

    pub fn path(&self) -> &Path {
        self.0.path()
    }

    pub fn write(&self, rel: &str, content: &str) -> PathBuf {
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
pub fn make_tree_writable(path: &Path) {
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
pub fn make_tree_writable(path: &Path) {
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

pub fn run_cli(cwd: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("failed to execute mallops binary")
}

pub fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

pub fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[allow(dead_code)] // used by ingest.rs
pub fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("json file must exist")).unwrap()
}
