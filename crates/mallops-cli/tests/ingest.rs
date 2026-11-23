//! End-to-end CLI tests for standard ingest blocks (`std/ingest.csv`, `std/ingest.parquet`).

mod common;

use std::fs;

use common::{WorkDir, read_json, run_cli, stderr, stdout};

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

    // 2. Second run with unchanged CSV: served from cache with the same artifact id.
    let second = run_cli(caller_cwd.path(), &["-f", yaml_arg, "run"]);
    assert_eq!(second.status.code(), Some(0), "stderr: {}", stderr(&second));
    assert!(
        stdout(&second).contains("0 executed, 1 cached"),
        "stdout:\n{}",
        stdout(&second)
    );
    assert_eq!(
        read_json(&latest_path)["outputs"]["raw"]["dataset"].as_str(),
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
fn ingest_e2e_relative_file_flag_from_other_cwd() {
    let root = WorkDir::new();
    root.write("sub/proj/data.csv", "id,label\n1,a\n2,b\n");
    root.write(
        "sub/proj/mallops.yaml",
        r#"version: 1
name: rel-flag
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

    let out = run_cli(root.path(), &["-f", "sub/proj/mallops.yaml", "run"]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    assert!(
        !root.path().join(".mallops").exists(),
        "store must live in sub/proj, not caller cwd"
    );
    let latest = read_json(&root.path().join("sub/proj/.mallops/meta/runs/latest.json"));
    let id = latest["outputs"]["raw"]["dataset"]
        .as_str()
        .expect("dataset id");
    let verify = run_cli(
        root.path(),
        &["-f", "sub/proj/mallops.yaml", "store", "verify", id],
    );
    assert_eq!(verify.status.code(), Some(0), "stderr: {}", stderr(&verify));
}

#[test]
fn ingest_e2e_parquet_reads_validates_and_caches() {
    // Produce a real Parquet file via `std/ingest.csv`, then feed it to `std/ingest.parquet`.
    let seed = WorkDir::new();
    seed.write("seed.csv", "id,name\n1,alice\n2,bob\n");
    seed.write(
        "mallops.yaml",
        r#"version: 1
name: seed-parquet
blocks:
  raw:
    use: std/ingest.csv
    with:
      path: seed.csv
"#,
    );
    let seed_run = run_cli(seed.path(), &["run"]);
    assert_eq!(
        seed_run.status.code(),
        Some(0),
        "stderr: {}",
        stderr(&seed_run)
    );
    let seed_latest = read_json(&seed.path().join(".mallops/meta/runs/latest.json"));
    let seed_id = seed_latest["outputs"]["raw"]["dataset"].as_str().unwrap();
    let seed_parquet = seed
        .path()
        .join(".mallops/artifacts")
        .join(seed_id)
        .join("payload/data.parquet");

    let proj = WorkDir::new();
    fs::copy(&seed_parquet, proj.path().join("input.parquet")).unwrap();
    proj.write(
        "mallops.yaml",
        r#"version: 1
name: ingest-parquet-e2e
blocks:
  raw:
    use: std/ingest.parquet
    with:
      path: input.parquet
      schema:
        - name: id
          type: int64
        - name: name
          type: string
"#,
    );

    let first = run_cli(proj.path(), &["run"]);
    assert_eq!(first.status.code(), Some(0), "stderr: {}", stderr(&first));
    let latest_path = proj.path().join(".mallops/meta/runs/latest.json");
    let id1 = read_json(&latest_path)["outputs"]["raw"]["dataset"]
        .as_str()
        .expect("dataset id")
        .to_owned();
    let manifest = read_json(
        &proj
            .path()
            .join(".mallops/artifacts")
            .join(&id1)
            .join("manifest.json"),
    );
    assert_eq!(manifest["kind"], "dataset");
    assert_eq!(manifest["format"], "parquet");

    let second = run_cli(proj.path(), &["run"]);
    assert_eq!(second.status.code(), Some(0), "stderr: {}", stderr(&second));
    assert!(
        stdout(&second).contains("0 executed, 1 cached"),
        "stdout:\n{}",
        stdout(&second)
    );

    // Mismatched schema fails with exit 1 and leaves latest.json unchanged.
    proj.write(
        "mallops.yaml",
        r#"version: 1
name: ingest-parquet-e2e
blocks:
  raw:
    use: std/ingest.parquet
    with:
      path: input.parquet
      schema:
        - name: id
          type: float64
        - name: name
          type: string
"#,
    );
    let bad = run_cli(proj.path(), &["run"]);
    assert_eq!(bad.status.code(), Some(1), "stdout:\n{}", stdout(&bad));
    assert!(
        stdout(&bad).contains("schema mismatch"),
        "stdout:\n{}",
        stdout(&bad)
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
