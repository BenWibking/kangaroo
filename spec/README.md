# Kangaroo Functional Specification

This directory defines the required observable behavior of Kangaroo.

The goal is to make the system reproducible by an implementation agent without requiring access to the original source code. The specification is normative for externally visible behavior and intentionally non-prescriptive about internal implementation choices.

## Intended Audience

This specification assumes the reader is technically fluent in:
- Distributed systems fundamentals.
- Adaptive Mesh Refinement (AMR) concepts.
- Scientific array data processing.

It does not assume prior knowledge of Kangaroo-specific terminology. See `spec/glossary.md` for project-specific definitions.

## Scope

This specification covers:
- Public Python API behavior.
- Plan model and runtime execution semantics.
- AMR metadata and chunk identity conventions.
- Dataset backend behavior for supported source types.
- Operator semantics for field, AMR, and particle workflows.
- Dashboard and script-level user-visible behavior.
- Validation and error semantics required for compatibility.

## Out of Scope

This specification does not require:
- Matching internal class layout or module structure.
- Matching C++/Python implementation language details.
- Matching scheduler internals, threading model internals, or memory allocator choice.
- Matching exact micro-optimization strategy.

A conforming implementation may differ internally as long as the externally observable behavior matches this specification.

## Conformance Definition

An implementation is conformant if all of the following hold:
- Public API signatures and return-shape semantics are compatible.
- Plan decoding, validation, and execution outcomes are equivalent.
- Operator numerical behavior and masking semantics are equivalent.
- Required error conditions produce explicit failures.
- Script and dashboard behavior is compatible with documented inputs/outputs.

## Conformance Priority

If two parts of this specification appear to conflict, precedence is:
1. `spec/validation-and-errors.md` (safety and rejection behavior)
2. `spec/operators.md` (numerical and workflow semantics)
3. `spec/python-api.md` (user-facing API contracts)
4. `spec/core-architecture.md` and `spec/data-models.md`
5. `spec/backends-and-io.md` and `spec/dashboard-and-cli.md`
6. `spec/glossary.md` (terminology guidance)

## Compatibility Target

The compatibility target is behavioral equivalence with the Kangaroo repository at the time this spec set was authored, including behavior validated by the repository test suite.

## Specification Map

- `spec/design-overview.md`: non-normative architecture narrative and mental model.
- `spec/core-architecture.md`: execution model, stage/template semantics, scheduling and event model.
- `spec/data-models.md`: metadata schema, geometry/index conventions, chunk identity and domain semantics.
- `spec/python-api.md`: Python-facing API contract (`Runtime`, `Dataset`, `Pipeline`, helpers).
- `spec/operators.md`: required kernels and semantics for all major operators.
- `spec/backends-and-io.md`: backend capability contract and reader behavior.
- `spec/dashboard-and-cli.md`: dashboard lifecycle and script-level behavior contract.
- `spec/validation-and-errors.md`: normative validation and failure requirements.
- `spec/glossary.md`: definitions for uncommon or overloaded terms.
- `spec/performance-requirements.md`: normative requirements for a performant implementation profile (GPU offload, distributed-memory execution, memory-capped streaming/out-of-core).

## Writing Convention

Normative keywords are used as follows:
- `MUST`: mandatory for conformance.
- `MUST NOT`: forbidden for conformance.
- `SHOULD`: recommended unless a documented compatibility reason prevents it.
- `MAY`: optional behavior.
