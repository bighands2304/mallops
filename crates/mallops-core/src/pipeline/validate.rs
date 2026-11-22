//! The single-pass pipeline validator behind [`PipelineConfig::validate`].

use std::collections::{BTreeMap, BTreeSet};

use super::{ValidatedNode, ValidatedPipeline};
use crate::config::PipelineConfig;
use crate::{
    ArtifactKind, Block, BlockRegistry, Error, Port, ResolvedPorts, Stage, ValidationError,
};
use std::sync::Arc;

/// A node whose block is known and whose ports resolved.
struct Resolved {
    block: Arc<dyn Block>,
    ports: ResolvedPorts,
}

/// Maps an error from `validate_config`/`resolve_ports` to paths under `blocks.<id>.with`.
fn with_errors(id: &str, e: Error) -> Vec<ValidationError> {
    let base = format!("blocks.{id}.with");
    match e {
        Error::Validation(v) => v
            .into_iter()
            .map(|mut ve| {
                ve.path = if ve.path.is_empty() {
                    base.clone()
                } else {
                    format!("{base}.{}", ve.path)
                };
                ve
            })
            .collect(),
        other => vec![ValidationError::new(base, other.to_string())],
    }
}

/// Splits `<node>.<output>` at the first dot; both parts must be non-empty.
fn split_ref(r: &str) -> Option<(&str, &str)> {
    r.split_once('.')
        .filter(|(n, p)| !n.is_empty() && !p.is_empty())
}

fn describe(p: &Port) -> String {
    format!(
        "kind `{}`, formats {:?}, flavors {:?}",
        p.kind.as_str(),
        p.formats,
        p.flavors
    )
}

/// Which of `kind`/`format`/`flavor` make `input` reject `output` (empty if it accepts).
fn mismatches(input: &Port, output: &Port) -> Vec<&'static str> {
    let mut dims = Vec::new();
    if input.kind != output.kind {
        dims.push("kind");
        return dims;
    }
    let mut no_flavor = input.clone();
    no_flavor.flavors.clear();
    if !no_flavor.accepts(output) {
        dims.push("format");
    }
    let mut no_format = input.clone();
    no_format.formats.clear();
    if !no_format.accepts(output) {
        dims.push("flavor");
    }
    dims
}

/// Returns the strongly connected component of `n` (reachable both ways) in `adj`.
fn scc<'a>(
    n: &'a str,
    fwd: &BTreeMap<&'a str, BTreeSet<&'a str>>,
    bwd: &BTreeMap<&'a str, BTreeSet<&'a str>>,
) -> BTreeSet<&'a str> {
    let reach = |adj: &BTreeMap<&'a str, BTreeSet<&'a str>>| {
        let mut seen = BTreeSet::from([n]);
        let mut stack = vec![n];
        while let Some(x) = stack.pop() {
            for &y in adj.get(x).into_iter().flatten() {
                if seen.insert(y) {
                    stack.push(y);
                }
            }
        }
        seen
    };
    reach(fwd).intersection(&reach(bwd)).copied().collect()
}

impl PipelineConfig {
    /// Validates this pipeline against `registry` without running any block.
    ///
    /// One pass, all errors aggregated; nodes of the result are in topological order
    /// (Kahn, ties broken by node id).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] with every problem found, each with a dotted path
    /// (e.g. `blocks.eval.in.test`, `blocks.eval.with.gate.min`, `serve.package`).
    pub fn validate(&self, registry: &BlockRegistry) -> Result<ValidatedPipeline, Error> {
        let mut errs: Vec<ValidationError> = Vec::new();

        // 1. blocks, `with:` config and ports.
        let mut ok: BTreeMap<&str, Resolved> = BTreeMap::new();
        for (id, cfg) in &self.blocks {
            let Some(block) = registry.get(&cfg.r#use) else {
                errs.push(ValidationError::new(
                    format!("blocks.{id}.use"),
                    format!("unknown block `{}`", cfg.r#use),
                ));
                continue;
            };
            if block.spec().stage == Stage::Serve {
                errs.push(ValidationError::new(
                    format!("blocks.{id}.use"),
                    format!(
                        "`{}` is a serve block: serve is not a DAG node, use the top-level `serve:`",
                        cfg.r#use
                    ),
                ));
                continue;
            }
            let mut good = true;
            if let Err(e) = block.validate_config(&cfg.with) {
                errs.extend(with_errors(id, e));
                good = false;
            }
            match block.resolve_ports(&cfg.with) {
                Ok(ports) if good => {
                    ok.insert(id, Resolved { block, ports });
                }
                Ok(_) => {}
                Err(e) => errs.extend(with_errors(id, e)),
            }
        }

        // 2. wiring.
        let mut fwd: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut bwd: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for (id, cfg) in &self.blocks {
            for (port, r) in &cfg.r#in {
                let path = format!("blocks.{id}.in.{port}");
                let Some((up, out)) = split_ref(r) else {
                    errs.push(ValidationError::new(
                        path,
                        format!("invalid reference `{r}`: expected `<node>.<output>`"),
                    ));
                    continue;
                };
                let Some((up_id, _)) = self.blocks.get_key_value(up) else {
                    errs.push(ValidationError::new(
                        path,
                        format!("unknown upstream node `{up}`"),
                    ));
                    continue;
                };
                fwd.entry(up_id.as_str()).or_default().insert(id);
                bwd.entry(id).or_default().insert(up_id.as_str());
                // no cascading errors: port checks need both ends resolved.
                let input = ok
                    .get(id.as_str())
                    .map(|n| (n, n.ports.inputs.iter().find(|p| p.name == port.as_str())));
                if let Some((_, None)) = input {
                    errs.push(ValidationError::new(
                        path.clone(),
                        format!("input port `{port}` is not declared by the block"),
                    ));
                }
                let Some(up_node) = ok.get(up) else { continue };
                let Some(output) = up_node.ports.outputs.iter().find(|p| p.name == out) else {
                    errs.push(ValidationError::new(
                        path,
                        format!("unknown output port `{out}` on node `{up}`"),
                    ));
                    continue;
                };
                if let Some((_, Some(inp))) = input {
                    let dims = mismatches(inp, output);
                    if !dims.is_empty() {
                        errs.push(ValidationError::new(
                            path,
                            format!(
                                "{} mismatch: `{r}` produces {}, input `{port}` accepts {}",
                                dims.join("+"),
                                describe(output),
                                describe(inp)
                            ),
                        ));
                    }
                }
            }
            if let Some(n) = ok.get(id.as_str()) {
                for p in n.ports.inputs.iter().filter(|p| p.required) {
                    if !cfg.r#in.contains_key(&*p.name) {
                        errs.push(ValidationError::new(
                            format!("blocks.{id}.in.{}", p.name),
                            "required input is not connected",
                        ));
                    }
                }
            }
        }

        // 3. topological order (Kahn, ties by id) and cycles.
        let mut indeg: BTreeMap<&str, usize> = self
            .blocks
            .keys()
            .map(|k| (k.as_str(), bwd.get(k.as_str()).map_or(0, BTreeSet::len)))
            .collect();
        let mut ready: BTreeSet<&str> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(k, _)| *k)
            .collect();
        let mut order: Vec<&str> = Vec::new();
        while let Some(n) = ready.pop_first() {
            order.push(n);
            for &m in fwd.get(n).into_iter().flatten() {
                if let Some(d) = indeg.get_mut(m) {
                    *d -= 1;
                    if *d == 0 {
                        ready.insert(m);
                    }
                }
            }
        }
        if order.len() < self.blocks.len() {
            let done: BTreeSet<&str> = order.iter().copied().collect();
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            for n in self
                .blocks
                .keys()
                .map(String::as_str)
                .filter(|n| !done.contains(n))
            {
                if seen.contains(n) {
                    continue;
                }
                let comp = scc(n, &fwd, &bwd);
                seen.extend(&comp);
                let self_loop = fwd.get(n).is_some_and(|s| s.contains(n));
                if comp.len() > 1 {
                    let list = comp.iter().map(|c| format!("`{c}`")).collect::<Vec<_>>();
                    errs.push(ValidationError::new(
                        format!("blocks.{n}"),
                        format!("dependency cycle between nodes {}", list.join(", ")),
                    ));
                } else if self_loop {
                    errs.push(ValidationError::new(
                        format!("blocks.{n}"),
                        format!("dependency cycle: node `{n}` depends on itself"),
                    ));
                }
            }
        }

        // 4. serve.package.
        if let Some(serve) = &self.serve {
            let path = "serve.package";
            match split_ref(&serve.package) {
                None => errs.push(ValidationError::new(
                    path,
                    format!(
                        "invalid reference `{}`: expected `<node>.<output>`",
                        serve.package
                    ),
                )),
                Some((node, _)) if !self.blocks.contains_key(node) => {
                    errs.push(ValidationError::new(path, format!("unknown node `{node}`")))
                }
                Some((node, out)) => {
                    if let Some(n) = ok.get(node) {
                        match n.ports.outputs.iter().find(|p| p.name == out) {
                            None => errs.push(ValidationError::new(
                                path,
                                format!("unknown output port `{out}` on node `{node}`"),
                            )),
                            Some(p) if p.kind != ArtifactKind::Package => {
                                errs.push(ValidationError::new(
                                    path,
                                    format!(
                                        "`{}` has kind `{}`, expected `package`",
                                        serve.package,
                                        p.kind.as_str()
                                    ),
                                ));
                            }
                            Some(_) => {}
                        }
                    }
                }
            }
        }

        if !errs.is_empty() {
            return Err(Error::Validation(errs));
        }
        // No errors: every node resolved and `order` covers all of them.
        let nodes = order
            .into_iter()
            .filter_map(|id| {
                let r = ok.remove(id)?;
                Some(ValidatedNode {
                    id: id.to_owned(),
                    block: r.block,
                    config: self.blocks.get(id)?.clone(),
                    ports: r.ports,
                })
            })
            .collect();
        Ok(ValidatedPipeline {
            config: self.clone(),
            nodes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BlockConfig, ServeConfig};
    use crate::{BlockOutcome, BoxFuture, Context, Inputs};

    /// Block whose `run` panics: validation must never call it.
    struct T {
        spec: crate::BlockSpec,
        cfg_err: Option<fn() -> Error>,
        ports_err: bool,
        /// Output flavor taken from `with.flavor` when set.
        dynamic: bool,
    }

    impl Block for T {
        fn spec(&self) -> &crate::BlockSpec {
            &self.spec
        }
        fn validate_config(&self, _: &serde_json::Value) -> Result<(), Error> {
            self.cfg_err.map_or(Ok(()), |f| Err(f()))
        }
        fn resolve_ports(&self, c: &serde_json::Value) -> Result<ResolvedPorts, Error> {
            if self.ports_err {
                return Err(Error::Config("bad ports".into()));
            }
            let mut r = ResolvedPorts::from(&self.spec);
            if self.dynamic {
                let f = c["flavor"].as_str().unwrap_or("none").to_owned();
                r.outputs = vec![Port::new("out", ArtifactKind::Model).with_flavor(f)];
            }
            Ok(r)
        }
        fn run<'a>(
            &'a self,
            _: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            panic!("validate must never run a block")
        }
    }

    fn t(id: &'static str, stage: Stage, ins: Vec<Port>, outs: Vec<Port>) -> T {
        let mut spec = crate::BlockSpec::new(id, "1", stage);
        spec.inputs = ins;
        spec.outputs = outs;
        T {
            spec,
            cfg_err: None,
            ports_err: false,
            dynamic: false,
        }
    }

    fn ds(n: &'static str) -> Port {
        Port::new(n, ArtifactKind::Dataset)
    }

    fn registry() -> BlockRegistry {
        let mut r = BlockRegistry::new();
        let blocks = [
            t("t/src", Stage::Ingest, vec![], vec![ds("out")]),
            t("t/mid", Stage::Preprocess, vec![ds("in")], vec![ds("out")]),
            t(
                "t/two",
                Stage::Preprocess,
                vec![ds("a"), ds("b")],
                vec![ds("out")],
            ),
            t(
                "t/opt",
                Stage::Preprocess,
                vec![ds("in").optional()],
                vec![ds("out")],
            ),
            t(
                "t/model",
                Stage::Train,
                vec![],
                vec![
                    Port::new("out", ArtifactKind::Model)
                        .with_format("onnx")
                        .with_flavor("tabular"),
                ],
            ),
            t(
                "t/wantonnx",
                Stage::Evaluate,
                vec![Port::new("in", ArtifactKind::Model).with_format("onnx")],
                vec![],
            ),
            t(
                "t/wantpt",
                Stage::Evaluate,
                vec![Port::new("in", ArtifactKind::Model).with_format("pt")],
                vec![],
            ),
            t(
                "t/wantvision",
                Stage::Evaluate,
                vec![Port::new("in", ArtifactKind::Model).with_flavor("vision")],
                vec![],
            ),
            t(
                "t/pkg",
                Stage::Package,
                vec![],
                vec![Port::new("out", ArtifactKind::Package)],
            ),
            t("t/serve", Stage::Serve, vec![], vec![]),
        ];
        for b in blocks {
            r.register(Arc::new(b)).unwrap();
        }
        let mut d = t(
            "t/dyn",
            Stage::Train,
            vec![],
            vec![Port::new("out", ArtifactKind::Model)],
        );
        d.dynamic = true;
        r.register(Arc::new(d)).unwrap();
        let mut c = t("t/badcfg", Stage::Evaluate, vec![], vec![]);
        c.cfg_err = Some(|| Error::validation("gate.min", "must be >= 0"));
        r.register(Arc::new(c)).unwrap();
        let mut c = t("t/rootcfg", Stage::Evaluate, vec![], vec![]);
        c.cfg_err = Some(|| Error::validation("", "bad"));
        r.register(Arc::new(c)).unwrap();
        let mut c = t("t/other", Stage::Evaluate, vec![], vec![]);
        c.cfg_err = Some(|| Error::Config("boom".into()));
        r.register(Arc::new(c)).unwrap();
        let mut c = t("t/portsfail", Stage::Evaluate, vec![ds("in")], vec![]);
        c.ports_err = true;
        r.register(Arc::new(c)).unwrap();
        r
    }

    fn node(use_: &str, ins: &[(&str, &str)]) -> BlockConfig {
        BlockConfig {
            r#use: use_.parse().unwrap(),
            r#in: ins
                .iter()
                .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
                .collect(),
            with: serde_json::Value::Null,
            cache: None,
            seed: None,
        }
    }

    fn pipe(nodes: Vec<(&str, BlockConfig)>, serve: Option<&str>) -> PipelineConfig {
        PipelineConfig {
            version: 1,
            name: "p".into(),
            store: Default::default(),
            lineage: Default::default(),
            blocks: nodes.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
            serve: serve.map(|p| ServeConfig {
                package: p.into(),
                host: "h".into(),
                port: 1,
            }),
        }
    }

    fn errors(p: &PipelineConfig) -> Vec<ValidationError> {
        match p.validate(&registry()) {
            Err(Error::Validation(v)) => v,
            other => panic!("expected validation errors, got {other:?}"),
        }
    }

    fn paths(p: &PipelineConfig) -> Vec<String> {
        errors(p).into_iter().map(|e| e.path).collect()
    }

    fn msg(p: &PipelineConfig, path: &str) -> String {
        errors(p)
            .into_iter()
            .find(|e| e.path == path)
            .unwrap()
            .message
    }

    #[test]
    fn validate_ok_topological_order_with_id_tiebreak() {
        let p = pipe(
            vec![
                ("z", node("t/src", &[])),
                ("a", node("t/mid", &[("in", "z.out")])),
                ("m", node("t/src", &[])),
                ("b", node("t/mid", &[("in", "a.out")])),
            ],
            None,
        );
        let v = p.validate(&registry()).unwrap();
        let ids: Vec<_> = v.nodes().iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, ["m", "z", "a", "b"]);
        assert!(v.serve().is_none());
    }

    #[test]
    fn validate_ok_with_serve_and_optional_input_unconnected() {
        let p = pipe(
            vec![("o", node("t/opt", &[])), ("pkg", node("t/pkg", &[]))],
            Some("pkg.out"),
        );
        assert_eq!(
            p.validate(&registry()).unwrap().serve().unwrap().package,
            "pkg.out"
        );
    }

    #[test]
    fn validate_self_loop() {
        let p = pipe(vec![("a", node("t/mid", &[("in", "a.out")]))], None);
        let e = errors(&p);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].path, "blocks.a");
        assert!(e[0].message.contains("itself"));
    }

    #[test]
    fn validate_two_node_cycle_one_error_on_first_node() {
        let p = pipe(
            vec![
                ("b", node("t/mid", &[("in", "a.out")])),
                ("a", node("t/mid", &[("in", "b.out")])),
            ],
            None,
        );
        let e = errors(&p);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].path, "blocks.a");
        assert!(e[0].message.contains("`a`") && e[0].message.contains("`b`"));
    }

    #[test]
    fn validate_three_node_cycle_with_tail_and_second_cycle() {
        let p = pipe(
            vec![
                ("a", node("t/mid", &[("in", "c.out")])),
                ("b", node("t/mid", &[("in", "a.out")])),
                ("c", node("t/mid", &[("in", "b.out")])),
                ("tail", node("t/mid", &[("in", "c.out")])),
                ("x", node("t/mid", &[("in", "y.out")])),
                ("y", node("t/mid", &[("in", "x.out")])),
            ],
            None,
        );
        let e = errors(&p);
        assert_eq!(paths(&p), ["blocks.a", "blocks.x"]);
        assert!(
            ["`a`", "`b`", "`c`"]
                .iter()
                .all(|n| e[0].message.contains(n))
        );
        assert!(!e[0].message.contains("tail"));
    }

    #[test]
    fn validate_unknown_block_and_unknown_namespace() {
        let p = pipe(
            vec![("a", node("t/nope", &[])), ("b", node("zz/src", &[]))],
            None,
        );
        assert_eq!(paths(&p), ["blocks.a.use", "blocks.b.use"]);
        assert!(msg(&p, "blocks.a.use").contains("t/nope"));
    }

    #[test]
    fn validate_serve_stage_block_in_blocks() {
        let p = pipe(vec![("s", node("t/serve", &[]))], None);
        assert_eq!(paths(&p), ["blocks.s.use"]);
    }

    #[test]
    fn validate_unknown_upstream_node_and_output_port() {
        let p = pipe(
            vec![
                ("src", node("t/src", &[])),
                ("a", node("t/mid", &[("in", "ghost.out")])),
                ("b", node("t/mid", &[("in", "src.nope")])),
            ],
            None,
        );
        assert_eq!(paths(&p), ["blocks.a.in.in", "blocks.b.in.in"]);
        assert!(msg(&p, "blocks.a.in.in").contains("ghost"));
        assert!(msg(&p, "blocks.b.in.in").contains("nope"));
    }

    #[test]
    fn validate_reference_without_dot_or_empty_parts() {
        let p = pipe(
            vec![
                ("src", node("t/src", &[])),
                ("a", node("t/mid", &[("in", "src")])),
                ("b", node("t/mid", &[("in", ".out")])),
                ("c", node("t/mid", &[("in", "src.")])),
            ],
            None,
        );
        assert_eq!(
            paths(&p),
            ["blocks.a.in.in", "blocks.b.in.in", "blocks.c.in.in"]
        );
        assert!(msg(&p, "blocks.a.in.in").contains("<node>.<output>"));
    }

    #[test]
    fn validate_reference_splits_at_first_dot() {
        // node `src` exists, output port named `out.x` does not.
        let p = pipe(
            vec![
                ("src", node("t/src", &[])),
                ("a", node("t/mid", &[("in", "src.out.x")])),
            ],
            None,
        );
        assert!(msg(&p, "blocks.a.in.in").contains("`out.x`"));
    }

    #[test]
    fn validate_unknown_input_port() {
        let p = pipe(
            vec![
                ("src", node("t/src", &[])),
                (
                    "a",
                    node("t/mid", &[("in", "src.out"), ("extra", "src.out")]),
                ),
            ],
            None,
        );
        assert_eq!(paths(&p), ["blocks.a.in.extra"]);
    }

    #[test]
    fn validate_missing_required_input() {
        let p = pipe(
            vec![
                ("src", node("t/src", &[])),
                ("a", node("t/two", &[("a", "src.out")])),
            ],
            None,
        );
        assert_eq!(paths(&p), ["blocks.a.in.b"]);
    }

    #[test]
    fn validate_port_mismatch_kind() {
        let p = pipe(
            vec![
                ("m", node("t/model", &[])),
                ("a", node("t/mid", &[("in", "m.out")])),
            ],
            None,
        );
        assert_eq!(paths(&p), ["blocks.a.in.in"]);
        assert!(msg(&p, "blocks.a.in.in").starts_with("kind mismatch"));
    }

    #[test]
    fn validate_port_mismatch_format() {
        let p = pipe(
            vec![
                ("m", node("t/model", &[])),
                ("a", node("t/wantpt", &[("in", "m.out")])),
            ],
            None,
        );
        assert!(msg(&p, "blocks.a.in.in").starts_with("format mismatch"));
        let ok = pipe(
            vec![
                ("m", node("t/model", &[])),
                ("a", node("t/wantonnx", &[("in", "m.out")])),
            ],
            None,
        );
        assert!(ok.validate(&registry()).is_ok());
    }

    #[test]
    fn validate_port_mismatch_flavor() {
        let p = pipe(
            vec![
                ("m", node("t/model", &[])),
                ("a", node("t/wantvision", &[("in", "m.out")])),
            ],
            None,
        );
        assert!(msg(&p, "blocks.a.in.in").starts_with("flavor mismatch"));
    }

    #[test]
    fn validate_dynamic_flavor_from_resolve_ports() {
        let mut d = node("t/dyn", &[]);
        d.with = serde_json::json!({"flavor": "tabular"});
        let p = pipe(
            vec![
                ("d", d.clone()),
                ("a", node("t/wantvision", &[("in", "d.out")])),
            ],
            None,
        );
        assert_eq!(paths(&p), ["blocks.a.in.in"]);
        assert!(msg(&p, "blocks.a.in.in").starts_with("flavor mismatch"));
        d.with = serde_json::json!({"flavor": "vision"});
        let p = pipe(
            vec![("d", d), ("a", node("t/wantvision", &[("in", "d.out")]))],
            None,
        );
        assert!(p.validate(&registry()).is_ok());
    }

    #[test]
    fn validate_config_error_paths_are_prefixed() {
        let p = pipe(
            vec![
                ("e", node("t/badcfg", &[])),
                ("r", node("t/rootcfg", &[])),
                ("o", node("t/other", &[])),
            ],
            None,
        );
        assert_eq!(
            paths(&p),
            ["blocks.e.with.gate.min", "blocks.o.with", "blocks.r.with"]
        );
        assert!(msg(&p, "blocks.o.with").contains("boom"));
    }

    #[test]
    fn validate_resolve_ports_failure_and_no_cascade() {
        let p = pipe(
            vec![
                ("src", node("t/src", &[])),
                // ports unresolved: no input-port/required errors, only the with error.
                ("f", node("t/portsfail", &[("whatever", "src.out")])),
                // downstream of an unresolved node: its own checks are skipped too.
                ("n", node("t/nope", &[("in", "src.out")])),
                ("a", node("t/mid", &[("in", "n.out")])),
            ],
            None,
        );
        assert_eq!(paths(&p), ["blocks.f.with", "blocks.n.use"]);
    }

    #[test]
    fn validate_serve_package_errors() {
        let base = |s| {
            pipe(
                vec![("src", node("t/src", &[])), ("pkg", node("t/pkg", &[]))],
                Some(s),
            )
        };
        assert_eq!(paths(&base("ghost.out")), ["serve.package"]);
        assert!(msg(&base("ghost.out"), "serve.package").contains("ghost"));
        assert!(msg(&base("pkg.nope"), "serve.package").contains("nope"));
        assert!(msg(&base("src.out"), "serve.package").contains("expected `package`"));
        assert!(msg(&base("nodot"), "serve.package").contains("<node>.<output>"));
        assert!(base("pkg.out").validate(&registry()).is_ok());
    }

    #[test]
    fn validate_aggregates_three_distinct_errors_without_running() {
        let p = pipe(
            vec![
                ("src", node("t/src", &[])),
                ("a", node("t/nope", &[])),
                ("b", node("t/two", &[("a", "src.out")])),
                ("c", node("t/mid", &[("in", "ghost.out")])),
            ],
            None,
        );
        assert_eq!(
            paths(&p),
            ["blocks.a.use", "blocks.b.in.b", "blocks.c.in.in"]
        );
        match p.validate(&registry()) {
            Err(Error::Validation(v)) => assert_eq!(v.len(), 3),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn validate_never_runs_block_even_on_success() {
        // every registry block panics in `run`; a successful validation must not reach it.
        let p = pipe(
            vec![
                ("src", node("t/src", &[])),
                ("a", node("t/mid", &[("in", "src.out")])),
            ],
            None,
        );
        assert!(p.validate(&registry()).is_ok());
    }
}
