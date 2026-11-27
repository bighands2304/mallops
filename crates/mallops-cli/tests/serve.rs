//! `mallops serve` end to end (issue #17): run a minimal pipeline, serve its package, survive
//! SIGHUP, stop on SIGTERM.

#[allow(dead_code)] // only part of the shared helpers is used here
mod common;

use common::{WorkDir, run_cli, stderr};

const PIPELINE: &str = r#"version: 1
name: serve-e2e
blocks:
  raw: { use: std/ingest.csv, with: { path: d.csv } }
  features:
    use: std/preprocess.tabular
    in: { data: raw.dataset }
    with: { target: y, steps: [ { split: { test: 0.25, seed: 1 } } ] }
  model:
    use: std/train.smartcore
    in: { train: features.train }
    with: { algorithm: logistic_regression, target: y }
  package:
    use: std/package
    in: { model: model.model, preprocess: features.transform }
serve: { package: package.package, port: 0 }
"#;

fn csv() -> String {
    let mut s = String::from("x1,x2,y\n");
    for i in 0..40 {
        let y = i % 2;
        s += &format!(
            "{},{},{y}\n",
            f64::from(y) * 2.0 + f64::from(i) / 40.0,
            f64::from(i % 7) / 7.0
        );
    }
    s
}

#[test]
fn serve_errors_without_run_or_with_memory_store() {
    let wd = WorkDir::new();
    wd.write("mallops.yaml", PIPELINE);
    std::fs::create_dir(wd.path().join(".mallops")).unwrap();
    let out = run_cli(wd.path(), &["serve"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("error: no successful run: run `mallops run` first"),
        "{}",
        stderr(&out)
    );

    let wd = WorkDir::new();
    wd.write("mallops.yaml", PIPELINE);
    let out = run_cli(wd.path(), &["serve"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("error: store not found: "),
        "{}",
        stderr(&out)
    );
    assert!(
        !wd.path().join(".mallops").exists(),
        "serve created the store"
    );

    let wd = WorkDir::new();
    wd.write(
        "mallops.yaml",
        &PIPELINE.replace("blocks:", "store: { kind: memory }\nblocks:"),
    );
    let out = run_cli(wd.path(), &["serve"]);
    assert!(
        stderr(&out).contains("error: serve needs a persistent store"),
        "{}",
        stderr(&out)
    );
}

#[cfg(unix)]
#[test]
fn serve_e2e_health_sighup_sigterm() {
    use std::io::{BufRead as _, BufReader, Read as _, Write as _};
    use std::net::TcpStream;
    use std::process::{Command, Stdio};
    use std::time::Duration;

    let wd = WorkDir::new();
    wd.write("d.csv", &csv());
    wd.write("mallops.yaml", PIPELINE);
    let run = run_cli(wd.path(), &["run"]);
    assert_eq!(run.status.code(), Some(0), "{}", stderr(&run));

    let mut child = Command::new(common::BIN)
        .current_dir(wd.path())
        .args(["serve", "--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let pid = child.id().to_string();
    let kill = |sig: &str| {
        assert!(
            Command::new("kill")
                .args([sig, &pid])
                .status()
                .unwrap()
                .success()
        );
    };
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let rest = line
        .trim()
        .strip_prefix("serving ")
        .unwrap_or_else(|| panic!("stdout: {line:?}"));
    let (id, addr) = rest.split_once(" on http://").unwrap();
    // Only now: the handler is installed (it precedes the load), so the signal cannot be lost.
    kill("-HUP");

    let health = || {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        s.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut body = String::new();
        s.read_to_string(&mut body).unwrap();
        body
    };
    let body = health();
    assert!(body.starts_with("HTTP/1.1 200"), "{body}");
    assert!(body.contains(id), "{body}");

    // A reload with an unchanged latest.json keeps the server up.
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        child.try_wait().unwrap().is_none(),
        "SIGHUP killed the server"
    );
    assert!(health().starts_with("HTTP/1.1 200"));

    kill("-TERM");
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(0));
}
