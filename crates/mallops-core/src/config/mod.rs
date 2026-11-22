//! Typed `mallops.yaml` configuration (SPEC §5).
//!
//! `${env:NAME}` values are interpolated at load time; `${secret:NAME}` references stay
//! symbolic and are resolved on demand through `Context::secret` (ADR-0007 D7).

#[allow(dead_code)] // ponytail: wired into the parser by task 5.3
mod interp;
