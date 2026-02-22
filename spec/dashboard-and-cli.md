# Dashboard and CLI Specification

## 1. Dashboard Process Contract

The dashboard is a live monitoring/inspection service for runtime execution.

Requirements:
- Runs as a Bokeh server application.
- Exposes configurable port and websocket origin policy.
- Rejects unsupported Python runtime versions that are known-incompatible.

## 2. Dashboard Configuration Inputs

Dashboard accepts:
- Optional metrics/event log JSONL path.
- Optional plan JSON path for DAG rendering.
- Optional workflow command to run under dashboard monitoring.
- Update interval, history window, title, and port.

If workflow mode is enabled without explicit metrics/plan paths:
- Temporary file paths MUST be created for both artifacts.
- Workflow environment MUST receive these paths.

## 3. Event Log Reader Behavior

The log reader MUST:
- Tail file incrementally from last offset.
- Detect file replacement/rotation by inode change and restart reading from beginning.
- Ignore empty lines and malformed JSON lines.
- Yield only valid dict-like event payloads.

## 4. Metric Sampling Behavior

Dashboard MUST support two metric sources:
- Runtime-emitted metric events from log.
- Local process sampling fallback when runtime metrics are absent.

Fallback sampling is active only while workflow is running.

## 5. Task Lifecycle Tracking

Task stream model MUST:
- Track task records by stable task ID.
- Handle start and terminal status updates.
- Preserve timing windows for display and rollover constraints.
- Drop stale tasks outside configured history when workflow is still active.

Terminal status semantics:
- Ended tasks are immutable except for completion metadata updates.
- Error status MUST be represented distinctly from successful completion.

## 6. Dashboard Visual Components

Dashboard MUST provide:
- Memory time-series panel.
- I/O throughput time-series panel.
- DAG panel from plan structure.
- Task stream timeline by worker lane.
- Aggregate flamegraph of total task durations by task name.
- KPI/status indicators for runtime state.

## 7. DAG Construction Semantics

### 7.1 Node Construction

Nodes are generated from plan stages/templates/domains.

Rules:
- Template with explicit block list produces one node per block.
- Template without explicit blocks produces one aggregate node.
- Stage with no templates still produces a stage node.

### 7.2 Edge Construction

For chunk-to-chunk parent/child stages:
- Prefer dataflow edges by matching parent outputs to child inputs on field/version.
- Respect input domain restrictions for block correspondence.
- When block correspondence can be inferred, create block-matched edges.
- If no data edges found, fallback to full parent-stage to child-stage expansion.

This behavior is mandatory for compatibility with dashboard DAG tests.

## 8. Workflow Launch Controls

If workflow command is configured:
- UI MUST include start action.
- Optional threads-per-locality setting MUST inject runtime thread config only when absent.
- Workflow stdout/stderr stream MUST be surfaced in dashboard process output with clear prefixing.

## 9. CLI Script Coverage

The following scripts are part of the behavior contract:
- `scripts/plotfile_slice.py`
- `scripts/plotfile_projection.py`
- `scripts/plotfile_projection_cic_stellar.py`
- `scripts/plotfile_particle_mass_histogram.py`
- `scripts/plotfile_fab_minmax.py`
- `scripts/plotfile_slice_yt.py`
- `scripts/smoke_demo.py`
- `scripts/slice_operator_demo.py`
- `scripts/kangaroo_dashboard.py`

## 10. Common CLI Requirements

Scripts MUST:
- Parse documented arguments.
- Validate argument formats and ranges.
- Route unknown runtime arguments where applicable.
- Produce explicit errors for invalid user input.
- Emit output artifacts and terminal summaries consistent with script purpose.

## 11. Script-Specific Required Behaviors

### 11.1 Slice and projection scripts

- Use dataset field resolution and geometry helpers.
- Build and execute pipeline operators.
- Retrieve output arrays and optionally render plots.
- Respect requested axis/zoom/resolution/output paths.

### 11.2 Particle histogram script

Must support:
- Explicit or inferred histogram range.
- Log-space edge construction.
- Optional density normalization.
- Optional CSV/NPZ output.
- Top-k exact mode extraction path.
- Warning and fallback behavior when runtime histogram returns empty while in-range data exists.

### 11.3 `smoke_demo.py`

Exit code contract:
- Distinct non-zero code for runtime initialization failure.
- Distinct non-zero code for runtime execution failure after initialization.
- Zero on successful run.

### 11.4 `scripts/kangaroo_dashboard.py`

- Accepts `--threads-per-locality`.
- Injects runtime thread config only when workflow execution mode is active and no thread config already supplied.
- Rejects invalid thread counts (< 1).
