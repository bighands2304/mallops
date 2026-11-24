//! Ctrl-C during `mallops run` cancels the run and kills the `std/train.external` process group
//! (issue #13).

#![cfg(unix)]

#[allow(dead_code)] // only part of the shared helpers is used here
mod common;

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{BIN, WorkDir};

/// Polls `cond` every 50 ms until it holds or `limit` elapses.
fn wait_until(limit: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    cond()
}

fn alive(pid: &str) -> bool {
    Command::new("kill")
        .args(["-0", pid])
        .stderr(Stdio::null())
        .status()
        .unwrap()
        .success()
}

fn read_pid(path: &Path) -> Option<String> {
    let pid = std::fs::read_to_string(path).ok()?;
    let pid = pid.trim();
    (!pid.is_empty()).then(|| pid.to_owned())
}

#[test]
fn ctrl_c_cancels_run_and_kills_trainer_process_group() {
    let wd = WorkDir::new();
    let pid_file = wd.path().join("grandchild.pid");
    let script = format!("sleep 30 & echo $! > {}; wait", pid_file.display());
    let yaml = format!(
        "version: 1\nname: ctrl-c\nblocks:\n  train:\n    use: std/train.external\n    with:\n      \
         command: {}\n      inputs: {{}}\n",
        serde_json::json!(["sh", "-c", script])
    );
    wd.write("mallops.yaml", &yaml);

    let mut mallops = Command::new(BIN)
        .current_dir(wd.path())
        .arg("run")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut grandchild = None;
    let started = wait_until(Duration::from_secs(30), || {
        grandchild = read_pid(&pid_file);
        grandchild.is_some()
    });
    if !started {
        let _ = mallops.kill();
        let out = mallops.wait_with_output().unwrap();
        panic!(
            "trainer never started; stderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let grandchild = grandchild.unwrap();

    let sigint = Command::new("kill")
        .args(["-INT", &mallops.id().to_string()])
        .status()
        .unwrap();
    assert!(sigint.success());

    let exited = wait_until(Duration::from_secs(10), || {
        mallops.try_wait().unwrap().is_some()
    });
    if !exited {
        let _ = mallops.kill();
        let _ = Command::new("kill").args(["-9", &grandchild]).status();
        panic!("mallops did not exit within 10 s of SIGINT");
    }
    let out = mallops.wait_with_output().unwrap();
    assert_ne!(
        out.status.code(),
        Some(0),
        "stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("1 cancelled"),
        "stdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );

    let dead = wait_until(Duration::from_secs(5), || !alive(&grandchild));
    if !dead {
        let _ = Command::new("kill").args(["-9", &grandchild]).status();
        panic!("grandchild {grandchild} survived Ctrl-C");
    }
}
