# Chunk Buffer Implementation Plan

## Status

- **State:** implemented and verified on 2026-07-10
- **Scope:** replace context-dependent raw chunk bytes with self-describing **Chunk Buffers**
- **Primary modules:** Python plan/lowering, plan IR, executor, data service, dataset backends, kernel interface, Python materialization
- **Compatibility posture:** atomic internal plan-format migration; no long-lived legacy packed-plan compatibility
- **Performance posture:** zero-copy backend reads remain supported; numeric kernels use variadic `float`/`double` dtype packs through arity 10

Verification snapshot:

- Release build: `KANGAROO_BUILD_DIR=.pixi/build-hpx pixi run -e pixi-hpx build`
- Full suite: 104 passed, 1 expected xfail
- Clang 19 ARM64 dtype-pack fixture: arity 9 compiled in 5 s; arity 10 in 12 s
- Scalar-proxy assembly fixture contains no out-of-line `memcpy`
- Section 15 deletion searches return no production matches

The project terminology used here is defined in [`CONTEXT.md`](../CONTEXT.md).

## 1. Objective

Deepen the chunk-storage module so dtype, logical shape, physical layout, allocation, validation, serialization, and typed access have one owner.

After this refactor:

- every transported chunk is a self-describing **Chunk Buffer**;
- scientific kernel parameters contain scientific configuration, not storage metadata;
- the executor allocates and estimates outputs from a **Buffer Specification** rather than decoding kernel-private parameters;
- kernels obtain typed arrays or **Block Grids** without manual byte arithmetic, dtype branches, layout branches, or `reinterpret_cast`;
- mixed `float32` and `float64` inputs are correct by construction;
- Python materializes NumPy arrays from the same descriptor used by C++;
- plotfile component slices remain zero-copy even when their physical layout is KJI;
- dynamic numeric and opaque outputs are explicit rather than represented by `output_bytes=[0]` or misleading one-element allocations.

This is a separation-of-concerns refactor. Numerical algorithms, task topology, field identity, and dataset semantics should remain unchanged except where existing raw-byte assumptions are already incorrect.

## 2. Evidence and Current Architectural Friction

### 2.1 Storage facts have no single owner

The current storage contract is split across:

- `SharedByteBuffer` and `HostView` in `cpp/include/kangaroo/kernel.hpp`;
- caller-provided `nx`, `ny`, `nz`, and `bytes_per_value` in `HostGridView3D`;
- `output_bytes` in `analysis/plan.py` and `TaskTemplateIR`;
- dtype and byte-width parameters in `analysis/ops.py` and `analysis/pipeline.py`;
- executor-side parsing of kernel parameters in `cpp/src/executor.cpp`;
- manual NumPy dtype/shape reconstruction in `analysis/runtime.py`, `analysis/dataset.py`, and `analysis/plotfile.py`;
- repeated output allocation and pointer casts in `cpp/src/runtime.cpp` and backend implementations.

The deletion test is strong: removing the proposed module would recreate byte-count validation, dtype dispatch, layout indexing, COW rules, allocation, serialization, and NumPy conversion across dozens of callers.

### 2.2 Existing correctness failures

The current single-`bytes_per_value` convention silently misreads valid mixed-width inputs in at least:

- one- and two-dimensional histograms;
- spherical and cylindrical flux-surface integrals;
- any future multi-field kernel that infers width from only one participating field.

The refactor must make per-input scalar type intrinsic to the buffer. A lowerer-level check that merely rejects mixed inputs is not the target architecture.

### 2.3 Existing allocation ambiguity

Examples include:

- executor allocation from `output_bytes` followed by kernel-side resize;
- particle masks declared as one output byte even though their output length matches the input chunk;
- particle filters declared as one `float64` even though their output is data-dependent;
- particle loads whose lengths are known only after backend reads;
- packed AMR patch payloads whose serialized size is dynamic;
- memory estimates derived by parsing kernel-specific msgpack.

The new plan contract must distinguish static, input-derived, and dynamic outputs.

## 3. Accepted Design Decisions

These decisions are settled for this implementation.

### 3.1 Chunk metadata travels with the bytes

A field catalog may help construct plans, but it is not the source of truth for a transported chunk. The concrete descriptor travels with every **Chunk Buffer** because buffers may be:

- independently sliced from a shared plotfile FAB;
- copied or detached through COW;
- fetched as subboxes;
- created as temporary task outputs;
- transported between HPX localities;
- materialized in Python independently of their original field registration.

### 3.2 Logical indexing is independent of physical layout

A **Block Grid** is always indexed logically as `(i, j, k)`.

Physical layout is represented by resolved strides in the concrete descriptor:

- runtime IJK: `{ny * nz, nz, 1}` elements;
- plotfile KJI: `{1, nx, nx * ny}` elements.

Layout selection and validation happen once when a view is constructed. The inner loop has no layout branch.

### 3.3 Numeric dtype dispatch is variadic through arity 10

Numeric grid kernels support independently typed `float32` and `float64` inputs through a variadic visitor with a maximum arity of 10.

For `N` independently typed real inputs, the runtime contains up to `2^N` typed loop variants. This code-size and compile-time cost is accepted.

Measured on 2026-07-10 with the repository's Clang 19 toolchain, C++20, and `-O3`:

| Numeric inputs | Typed variants | Non-combinatorial baseline | Variadic compile time | Object size |
|---:|---:|---:|---:|---:|
| 6 | 64 | 0.33 s | 1.53 s | 171 KiB |
| 7 | 128 | 0.33 s | 3.02-3.46 s | 362 KiB |
| 8 | 256 | 0.33 s | 6.31-7.06 s | 766 KiB |
| 9 | 512 | 0.39 s | 12.89-13.37 s | 1.49 MiB |
| 10 | 1,024 | 0.34 s | 21.94-22.81 s | 2.34 MiB |

Integer and mask arrays do not join this combinatorial visitor. They use exact typed array access.

### 3.4 Byte storage remains standards-safe

Typed scalar access over byte-owned storage uses fixed-size `memcpy` loads and stores hidden by an inline scalar reference/proxy.

On the same Clang 19 ARM64 toolchain, an AXPY loop using these helpers compiled to native scalar load/FMA/store instructions without out-of-line `memcpy`. Unlike direct `double*` access, a general runtime-strided `TensorView` is not required to vectorize because its unit-stride and aliasing properties are unavailable at compile time. This avoids exposing alignment and object-lifetime assumptions while keeping scalar access free of helper calls; direct-pointer-equivalent vectorization is reserved for a future precompiled contiguous-layout specialization.

Generated assembly must remain a performance gate during implementation.

### 3.5 The shape language is closed

The plan supports only the output shapes needed by the current runtime:

1. **Block-shaped**: derived from the executing block's logical extents, optionally with a fixed trailing component extent.
2. **Fixed-shaped**: literal extents known during plan construction.
3. **Like-input**: logical shape copied from one task input, optionally changing scalar type.
4. **Dynamic one-dimensional**: actual length determined during execution, with a declared upper-bound rule.

There is no general shape-expression DSL in this refactor.

### 3.6 Opaque payloads remain explicit

Packed msgpack records and heterogeneous serialized records use `ScalarType::kOpaque` and a one-dimensional byte extent.

An **Opaque Payload** may expose bytes but cannot expose a numeric array or **Block Grid**. Do not invent fake scalar metadata for:

- packed AMR neighbor patch collections;
- serialized particle-value count maps;
- other heterogeneous record encodings.

### 3.7 Static and dynamic allocation have different owners

- The executor is the sole allocator for statically resolvable outputs.
- A dynamic **Buffer Specification** authorizes a kernel to choose the final length without exceeding its resolved upper bound.
- The descriptor is updated atomically when a dynamic output commits its final length.
- Kernels may not resize statically specified outputs.

### 3.8 Plan migration is atomic at the supported interface

`Runtime.run()` constructs and serializes plans immediately before execution. There is no repository-supported persisted-plan read workflow.

Implementation may temporarily accept both representations while patches are developed, but the completed refactor will:

- emit only **Buffer Specifications**;
- reject templates that use `output_bytes`;
- remove executor inference from kernel parameters;
- update dashboard JSON to the new representation.

No permanent legacy adapter is part of the target architecture.

## 4. Target C++ Chunk Buffer Module

### 4.1 Files

Create:

- `cpp/include/kangaroo/chunk_buffer.hpp`
- `cpp/src/chunk_buffer.cpp` if non-template validation/formatting code warrants a source file

Update or eventually simplify:

- `cpp/include/kangaroo/kernel.hpp`
- `cpp/include/kangaroo/data_service.hpp`
- `cpp/include/kangaroo/runtime.hpp`

### 4.2 Scalar types

Start with the types used by the live runtime:

```cpp
enum class ScalarType : std::uint8_t {
  kOpaque,
  kU8,
  kI64,
  kF32,
  kF64,
};
```

Do not add unused integer widths speculatively. Adding a new scalar requires:

- C++ type mapping;
- scalar size mapping;
- Python dtype mapping;
- HPX serialization coverage;
- exact typed access tests;
- a decision about whether it participates in numeric visitation.

Only `kF32` and `kF64` participate in the variadic real-grid visitor.

### 4.3 Concrete buffer descriptor

Use a concrete descriptor with resolved physical strides:

```cpp
inline constexpr std::size_t kMaxBufferRank = 4;

struct BufferDesc {
  ScalarType scalar = ScalarType::kOpaque;
  std::uint8_t rank = 1;
  std::array<std::uint64_t, kMaxBufferRank> extents{};
  std::array<std::int64_t, kMaxBufferRank> strides_bytes{};

  std::uint64_t element_count() const;
  std::uint64_t required_bytes() const;
  void validate(std::size_t visible_storage_bytes) const;

  static BufferDesc contiguous(ScalarType, std::span<const std::uint64_t> extents);
  static BufferDesc runtime_grid(ScalarType, std::array<std::uint64_t, 3> extents);
  static BufferDesc plotfile_grid(ScalarType, std::array<std::uint64_t, 3> extents);
};
```

Rules:

- rank must be between 1 and 4;
- unused extents and strides are zero;
- numeric extents must be positive;
- opaque buffers use rank 1 and extent equal to byte count;
- extent, stride, and byte calculations use checked arithmetic;
- supported descriptors are dense positive-stride permutations without padding or broadcast axes;
- arbitrary public stride construction is not required for ordinary callers;
- backend adapters use named factories or an internal validated constructor;
- descriptor validation occurs after construction and HPX deserialization.

The maximum rank covers:

- rank 1 particle/reduction arrays;
- rank 2 images and histograms;
- rank 3 scalar block fields;
- rank 4 block fields with a trailing fixed component dimension, such as gradients.

### 4.4 Chunk Buffer ownership

Replace/deepen `HostView` into a self-describing buffer:

```cpp
class ChunkBuffer {
 public:
  static ChunkBuffer allocate(BufferDesc, InitPolicy);
  static ChunkBuffer wrap(SharedByteBuffer, BufferDesc);

  const BufferDesc& desc() const noexcept;
  std::size_t bytes() const noexcept;
  std::span<const std::uint8_t> byte_view() const noexcept;
  std::span<std::uint8_t> mutable_byte_view();

  template <typename T, std::size_t Rank>
  TensorView<const T, Rank> view() const;

  template <typename T, std::size_t Rank>
  TensorView<T, Rank> mutable_view();

  template <typename T>
  ArrayView<const T> array() const;

  template <typename T>
  ArrayView<T> mutable_array();

  void commit_dynamic_extent(std::uint64_t elements);

  template <typename Archive>
  void serialize(Archive&, unsigned);

 private:
  SharedByteBuffer storage_;
  BufferDesc desc_;
  std::optional<std::uint64_t> dynamic_capacity_elements_;
};
```

Naming may retain a temporary `using HostView = ChunkBuffer` during migration, but `HostView` is removed from the final interface.

### 4.5 Copy-on-write invariants

- Const byte or typed views never detach storage.
- The first mutable view acquisition performs COW detachment if storage is shared or sliced.
- View acquisition completes all validation and detachment before returning a borrowed view.
- A borrowed view is invalidated by buffer resize, replacement, move, or another operation that can detach storage.
- Element access never checks or changes ownership state.
- HPX serialization of a slice materializes only the visible slice, matching current behavior.

### 4.6 Typed views

`TensorView<T, Rank>` stores only:

- a borrowed byte pointer;
- logical extents;
- resolved byte strides.

It provides:

- unchecked `operator()` for hot loops;
- optional checked `at()` for tests and non-hot paths;
- extent observers;
- no allocation, virtual dispatch, ownership mutation, dtype branch, or layout branch during indexing.

Since storage is byte-oriented, element reads and writes use inline fixed-size `memcpy` helpers. The optimizer is expected to reduce these to native scalar/vector memory operations.

Define convenience aliases or wrappers:

```cpp
template <typename T>
using BlockGrid = TensorView<T, 3>;

template <typename T>
using ComponentBlockGrid = TensorView<T, 4>;
```

### 4.7 Variadic dtype visitation

Provide one central helper for independently typed real inputs:

```cpp
template <std::size_t MaxInputs = 10, typename F>
decltype(auto) visit_real_buffers(std::span<const ChunkBuffer> inputs, F&& fn);
```

Implementation requirements:

- recursively dispatch `kF32` versus `kF64` once per input;
- instantiate a callable with a typed parameter pack;
- reject zero inputs if the caller requires at least one;
- reject more than 10 inputs with an explicit contract error;
- reject non-real scalar types before entering the numerical loop;
- keep the visitor in one module so individual kernels do not reinvent recursion;
- make compile-time and object-size behavior measurable by the benchmark described below.

Kernels with optional inputs should build the exact participating input span before visitation. Do not instantiate absent optional fields as dummy buffers.

### 4.8 Error model

Introduce a single `BufferContractError : std::runtime_error` with structured reason categories internally:

- invalid rank;
- invalid extent;
- arithmetic overflow;
- descriptor/storage mismatch;
- scalar mismatch;
- rank mismatch;
- invalid dynamic resize;
- dynamic upper-bound violation;
- unsupported real visitation dtype;
- visitor arity exceeded;
- opaque payload used as numeric data.

The calling layer adds kernel name, input/output index, field, and chunk identity where available.

Contract errors must propagate through HPX futures. Do not convert them into empty outputs or silent fallback values.

## 5. Plan-Time Buffer Specification

### 5.1 Python representation

Create `analysis/buffer.py` with immutable plan types:

```python
class DType(str, Enum):
    OPAQUE = "opaque"
    U8 = "u8"
    I64 = "i64"
    F32 = "f32"
    F64 = "f64"

class InitPolicy(str, Enum):
    UNINITIALIZED = "uninitialized"
    ZERO = "zero"

@dataclass(frozen=True)
class BlockShape:
    components: int = 1

@dataclass(frozen=True)
class FixedShape:
    extents: tuple[int, ...]

@dataclass(frozen=True)
class LikeInputShape:
    input_index: int

@dataclass(frozen=True)
class DynamicShape:
    upper_bound: DynamicUpperBound

@dataclass(frozen=True)
class BufferSpec:
    dtype: DType
    shape: BlockShape | FixedShape | LikeInputShape | DynamicShape
    init: InitPolicy = InitPolicy.UNINITIALIZED
```

`BlockShape(components=3)` resolves to logical extents `(nx, ny, nz, 3)` and contiguous runtime layout.

Opaque output specifications must use `DType.OPAQUE` and `DynamicShape` or a one-dimensional `FixedShape`.

### 5.2 Output references own output specifications

Replace the parallel `outputs` and `output_bytes` lists with one output structure:

```python
@dataclass(frozen=True)
class OutputRef:
    field: FieldRef
    buffer: BufferSpec
```

`TaskTemplate.outputs` becomes `list[OutputRef]`.

This prevents output identity and output allocation metadata from becoming misaligned.

### 5.3 C++ plan IR

Add corresponding IR types in `cpp/include/kangaroo/plan_ir.hpp`:

```cpp
enum class ShapeRuleKind : std::uint8_t {
  kBlock,
  kFixed,
  kLikeInput,
  kDynamic,
};

struct BufferSpecIR {
  ScalarType scalar;
  ShapeRuleKind shape_kind;
  std::vector<std::uint64_t> fixed_extents;
  std::uint32_t block_components = 1;
  std::int32_t like_input_index = -1;
  DynamicUpperBoundIR dynamic_upper_bound;
  InitPolicy init = InitPolicy::kUninitialized;
};

struct OutputRefIR {
  FieldRefIR field;
  BufferSpecIR buffer;
};
```

`TaskTemplateIR.outputs` becomes `std::vector<OutputRefIR>`, and `output_bytes` is removed.

### 5.4 Serialization format

Each serialized output contains:

```json
{
  "field": 123,
  "version": 0,
  "buffer": {
    "dtype": "f64",
    "shape": {"kind": "block", "components": 1},
    "init": "uninitialized"
  }
}
```

Fixed example:

```json
{
  "field": 456,
  "version": 0,
  "buffer": {
    "dtype": "f64",
    "shape": {"kind": "fixed", "extents": [256, 256]},
    "init": "zero"
  }
}
```

The decoder validates all buffer specifications before task expansion.

### 5.5 Dynamic upper bounds

Support upper-bound sources needed by current kernels:

- literal element or byte bound;
- same element count as an input;
- backend-provided chunk estimate;
- computed AMR subbox-pack bound.

Every dynamic output must have a conservative upper bound before the old storage path is deleted.

Required mappings:

- particle filter: input value count;
- particle field load: backend particle chunk estimate;
- packed particle count map: input count times maximum encoded record size plus header;
- AMR patch pack: sum of requested subbox byte bounds plus serialization overhead;
- other dynamic outputs: document and implement an equivalent conservative rule.

When memory-capped execution is active, an unresolved upper bound is an explicit admission error.

## 6. Executor and Data-Service Responsibilities

### 6.1 Descriptor resolution

Add a resolver that converts `BufferSpecIR` plus task context into a concrete `BufferDesc` or dynamic allocation contract.

Inputs include:

- template domain;
- executing block;
- `LevelMeta` box extents;
- already fetched input descriptors for `LikeInputShape`;
- backend estimates for dynamic sources.

The resolver is the only implementation of plan-shape semantics in C++.

### 6.2 Output allocation

For static outputs:

1. resolve the concrete descriptor;
2. compute exact required bytes with checked arithmetic;
3. allocate through `DataService::alloc_host` using descriptor and initialization policy;
4. pass the fully described buffer to the kernel;
5. reject kernel attempts to change its descriptor or size.

For dynamic outputs:

1. resolve and reserve the upper-bound capacity;
2. mark the buffer dynamic and initially empty;
3. allow one final extent commit from the kernel;
4. validate the committed extent against the capacity;
5. publish the concrete descriptor with the output.

### 6.3 Data-service interface

Replace byte-only allocation:

```cpp
virtual HostView alloc_host(const ChunkRef&, std::size_t bytes);
```

with descriptor-aware allocation:

```cpp
virtual ChunkBuffer alloc_host(const ChunkRef&, const ResolvedBufferSpec&);
```

`get_host`, `get_hosts`, and `put_host` transport `ChunkBuffer` unchanged.

The data service must not infer dtype, shape, or layout.

### 6.4 Memory accounting

Remove `input_bytes_per_value_from_params()` and related kernel-parameter inspection.

Memory estimates come from:

- concrete descriptors on dataset or materialized input buffers;
- resolved output descriptors;
- dynamic output upper bounds;
- backend `describe_chunk`/estimate methods before input materialization.

Fix known-output accounting so each output is recorded with its own descriptor and byte size. Never sum all template outputs and apply the sum to each output.

### 6.5 Backend description interface

Extend `DatasetBackend` with a descriptor/estimate query suitable for admission:

```cpp
virtual std::optional<BufferDesc> describe_chunk(const ChunkRef&) const;
virtual std::size_t estimate_chunk_bytes(const ChunkRef&) const;
```

For dynamic particle chunks, add a backend-specific estimate surfaced through a generic runtime query rather than kernel-private parameters.

## 7. Dataset Backend Adapters

### 7.1 Memory backend

The memory backend is the first typed test adapter.

Update Python/C++ write interfaces so callers provide self-describing inputs:

- NumPy input: derive dtype, shape, and contiguous layout automatically;
- raw bytes: require explicit dtype/shape or mark explicitly opaque;
- reject byte-count mismatches before storage;
- preserve step, level, field, version, and block identity unchanged.

Do not infer element width by fetching a chunk and dividing its byte size by a block cell count.

### 7.2 AMReX plotfile backend

Preserve the existing shared-FAB component slicing path:

1. read a contiguous FAB allocation;
2. create one `SharedByteBuffer` over the allocation;
3. slice each requested component without copying;
4. attach scalar type from `FabData::type`;
5. attach logical extents `(nx, ny, nz)`;
6. attach plotfile KJI strides;
7. return the component **Chunk Buffer**.

When the configured non-zero-copy path transposes a component, describe it with runtime IJK strides.

Update `spec/backends-and-io.md`: the required contract is correct logical `(i, j, k)` access, not mandatory eager transposition.

### 7.3 openPMD backend

- derive scalar type from the openPMD datatype;
- attach the post-remap logical XYZ extents;
- describe the actual post-remap physical layout;
- preserve scaling semantics;
- reject unsupported datatypes explicitly.

### 7.4 Parthenon backend

- derive scalar type from `DatasetInfo::type`;
- attach the selected component's logical block extents;
- audit and document the returned component order;
- describe that actual order rather than relying on a global runtime assumption;
- retain component-name registration semantics.

### 7.5 Plotfile reader Python interface

`PlotfileReader.read_fab(..., return_ndarray=True)` and particle readers already receive dtype and shape metadata. Route them through the shared Python dtype mapping so direct-reader and runtime materialization cannot drift.

## 8. Subbox and AMR Patch Handling

### 8.1 Subbox requests

Remove `bytes_per_value` from:

- `ChunkSubboxRef`;
- `SubboxView`;
- Python test bindings;
- `DataServiceLocal::build_subbox_view` callers.

The source **Chunk Buffer** supplies scalar type, extents, and layout.

### 8.2 Subbox outputs

A numeric subbox result is a contiguous runtime-layout **Chunk Buffer** with:

- the same scalar type as the source;
- extents equal to the returned overlap;
- a descriptor validated against copied bytes.

Subbox copying uses typed or scalar-size-generic views from the chunk-buffer module. It does not reimplement KJI/IJK indexing.

### 8.3 Packed AMR neighbor patches

`amr_subbox_fetch_pack` remains an **Opaque Payload**, but each embedded patch record must include its own concrete numeric descriptor instead of a separate `bytes_per_value` integer.

`gradU_stencil` decodes the records and obtains typed **Block Grids** from the embedded descriptors.

The pack format remains kernel-private serialization. It is not promoted to the general **Chunk Buffer** interface.

## 9. Kernel Interface and Migration

### 9.1 Kernel function type

Change the kernel data arguments from `HostView` to `ChunkBuffer`:

```cpp
using KernelFn = std::function<hpx::future<void>(
    const LevelMeta&,
    int32_t block,
    std::span<const ChunkBuffer> inputs,
    const NeighborBuffers& neighbors,
    std::span<ChunkBuffer> outputs,
    std::span<const std::uint8_t> params_msgpack)>;
```

Keep parameter decoding separate. Buffer access does not depend on msgpack.

### 9.2 Kernel I/O convenience module

Add a lightweight non-owning `KernelIO`/`BlockIO` implementation over the spans if it materially reduces repeated validation:

- `input_array<T>(index)`;
- `output_array<T>(index)`;
- `input_grid(index)` for variadic real visitation;
- `output_grid<T>(index)`;
- `opaque_input(index)` and `opaque_output(index)`.

This is an internal adapter over the existing kernel seam, not a new virtual seam.

### 9.3 Migration order by kernel family

#### Family A: mixed-width correctness tracer bullet

Migrate first:

- `histogram1d_accumulate`;
- `histogram2d_accumulate`;
- their reduction/finalization path.

Required result:

- mixed `float32`/`float64` axes and weights produce correct histograms;
- output allocation comes from fixed **Buffer Specifications**;
- no histogram parameter contains `bytes_per_value`;
- the executor does not inspect histogram params for memory estimates.

#### Family B: high-arity numeric kernels

Migrate next:

- `flux_surface_integral_accumulate`;
- `cylindrical_flux_surface_integral_accumulate`;
- Toomre/profile accumulators with many field inputs;
- other multi-field scientific reductions.

Use the accepted variadic real dtype visitor through arity 10.

Verify every `float32`/`float64` pattern for smaller arities and representative mixed patterns for arities 9-10. Full enumeration belongs in compile-time instantiation, not runtime test matrices with 1,024 expensive scientific runs.

#### Family C: ordinary block grids

Migrate:

- uniform slice and projection accumulation/finalization;
- field expressions;
- gradients and vorticity;
- plotfile load/transpose paths;
- other block-local numeric transforms.

Use rank-4 block-shaped outputs for interleaved vector/tensor results instead of manual `components * index + component` byte arithmetic.

#### Family D: particle arrays and reductions

Migrate particle kernels to exact rank-1 types:

- particle values: `f64` initially;
- particle integer values/counts: `i64`;
- masks: `u8`;
- scalar reductions: fixed rank-1 extent 1;
- masks and elementwise results: `LikeInputShape`;
- filters and backend loads: `DynamicShape`.

Do not route `u8` or `i64` through the real-grid variadic visitor.

#### Family E: opaque codecs

Migrate:

- particle value-count map encoding/decoding;
- AMR patch pack encoding/decoding;
- any other heterogeneous packed records.

Use explicit byte access and dynamic opaque specifications.

#### Family F: generic reductions

Replace `uniform_slice_reduce`/`uniform_slice_add` byte-width switches with exact typed array reductions.

The input and output descriptors must match unless a specifically named conversion reduction is introduced.

## 10. Python Runtime and Materialization

### 10.1 Shared dtype mapping

Create one Python mapping between serialized dtype tags and NumPy dtypes. Use it in:

- runtime output materialization;
- dataset particle reads;
- direct plotfile reads;
- memory dataset writes;
- pipeline particle caches.

### 10.2 Runtime output retrieval

Expose bytes plus descriptor metadata from nanobind.

The Python runtime should be able to materialize an array without caller-provided dtype or shape:

```python
arr = runtime.get_task_chunk_array(
    step=...,
    level=...,
    field=...,
    block=...,
    dataset=...,
)
```

Optional caller-provided dtype/shape may be retained only as assertions. A mismatch raises an explicit error.

Provide an explicitly named raw-byte retrieval method for debugging and opaque payloads. Do not make raw bytes the only general retrieval interface.

### 10.3 Pipeline handles

Where useful, carry expected output `BufferSpec` on `FieldHandle`, histogram handles, flux handles, and particle handles so Python can validate graph composition before packing the plan.

The concrete runtime descriptor remains authoritative after execution.

### 10.4 Remove inference APIs

Delete or deprecate after migration:

- `Dataset.infer_bytes_per_value()`;
- `Pipeline._infer_field_bytes_per_value()`;
- `_resolve_bytes_per_value()` in operator lowering;
- public `bytes_per_value` options whose only purpose is storage interpretation.

Keep explicit output dtype selection where it is a meaningful user choice, expressed as dtype rather than byte width.

## 11. Implementation Milestones

Each milestone must leave the branch buildable and have a focused verification target.

### Milestone 0: contract tests and benchmark fixtures

Add:

- `tests/test_chunk_buffer.py` using narrow `_core` test hooks for C++ invariants;
- a compile-time benchmark under `scripts/utils/` reproducing the accepted arity 1-10 dtype-pack measurements;
- a code-generation fixture comparing scalar-proxy and direct typed loops.

The benchmark is a manual performance gate, not part of every `pytest` run.

Exit criteria:

- benchmark commands and reference environment are documented;
- tests fail because the new chunk-buffer interface is not yet present, or land in the same commit as Milestone 1 if the repository must remain green.

### Milestone 1: core Chunk Buffer module

Implement:

- scalar types;
- `BufferDesc` factories and validation;
- checked size arithmetic;
- `ChunkBuffer` allocation/wrapping;
- COW-safe const/mutable access;
- fixed-size scalar proxy;
- rank-1, rank-2, rank-3, and rank-4 views;
- dynamic extent commit;
- HPX serialization;
- variadic real visitation through arity 10.

Keep `HostView` as a temporary alias or wrapper if required to avoid migrating all callers in one commit.

Exit criteria:

- all descriptor, view, COW, serialization, and visitation tests pass;
- scalar-proxy assembly uses native scalar loads/stores and contains no out-of-line `memcpy` in the representative loop;
- existing tests still pass through the temporary compatibility layer.

### Milestone 2: backend and data-service metadata propagation

Implement fully described buffers for:

- memory backend;
- plotfile backend, including zero-copy KJI slices;
- openPMD backend when enabled;
- Parthenon backend when enabled;
- data-service get/put and remote HPX transport;
- subbox fetch.

Exit criteria:

- buffers retain descriptors across local and remote data-service paths;
- plotfile zero-copy reads remain shallow slices;
- KJI and IJK produce identical logical values through **Block Grid** access;
- Python memory inputs must be typed or explicitly opaque.

### Milestone 3: Buffer Specification plan and executor tracer bullet

Implement:

- Python `BufferSpec` and `OutputRef`;
- msgpack encoding;
- C++ IR and decoder validation;
- shape resolution;
- descriptor-aware executor allocation;
- descriptor-based memory estimates;
- dashboard JSON changes.

Use histogram2D as the first complete operator through the new path.

Exit criteria:

- a mixed-width memory dataset executes histogram2D correctly;
- the histogram plan contains no output byte count or storage-width kernel parameter;
- executor memory estimates match concrete descriptors;
- malformed specifications fail during decode or task preparation.

### Milestone 4: high-arity scientific kernels

Migrate flux-surface, cylindrical flux-surface, and profile kernels to variadic dtype packs.

Exit criteria:

- mixed-width regression tests pass;
- homogeneous existing results remain within current numerical tolerances;
- arity 9 and 10 build times remain within the accepted reference envelope;
- no per-element dtype branch appears in optimized assembly.

### Milestone 5: remaining numeric grids and reductions

Migrate ordinary block, projection, gradient, expression, and reduction kernels.

Exit criteria:

- all numeric kernel buffer access uses typed views;
- component fields use rank-4 views where appropriate;
- generic reductions validate exact descriptor agreement;
- existing AMR correctness tests pass.

### Milestone 6: particle and opaque paths

Migrate exact typed particle arrays, dynamic filters/loads, count maps, and AMR patch packs.

Exit criteria:

- masks, integer counts, and real arrays retain exact types;
- dynamic outputs never exceed declared bounds;
- opaque payloads cannot be materialized as numeric arrays;
- particle and AMR neighbor workflows pass end to end.

### Milestone 7: delete the legacy storage surface

Remove:

- `HostGridView3D`;
- `HostView` compatibility alias;
- `TaskTemplate.output_bytes` and `TaskTemplateIR::output_bytes`;
- executor parameter parsing for input widths;
- numeric-buffer `reinterpret_cast` sites in kernels;
- storage-oriented `bytes_per_value` parameters and inference helpers;
- redundant kernel-side allocation for static outputs.

Update specifications and README examples in the same milestone.

Exit criteria:

- deletion checks in Section 15 pass;
- full test suite passes;
- documentation describes only the new interface.

## 12. Test Plan

### 12.1 Descriptor and allocation tests

Cover:

- scalar-size mapping;
- rank 1-4 descriptors;
- runtime IJK and plotfile KJI strides;
- checked extent multiplication;
- invalid rank and zero/overflowing extents;
- descriptor/storage byte mismatch;
- zero versus uninitialized allocation;
- opaque restrictions;
- dynamic extent commit and upper-bound violation.

### 12.2 View tests

Cover:

- exact typed array access for `u8`, `i64`, `f32`, and `f64`;
- scalar mismatch and rank mismatch errors;
- logical equality between IJK and KJI physical layouts;
- rank-4 trailing component access;
- checked versus unchecked indexing behavior;
- mutable writes through scalar proxies.

### 12.3 Copy-on-write and slice tests

Cover:

- cheap const copies;
- const access without detachment;
- mutation detaching a shared full buffer;
- mutation detaching a shared component slice;
- preservation of sibling slices after mutation;
- view invalidation rules around resize/move;
- serialization of only visible slice bytes.

### 12.4 Serialization and distributed tests

Cover:

- HPX archive round trip for all scalar types and ranks;
- post-deserialization validation;
- local data-service put/get descriptor preservation;
- remote action descriptor preservation when distributed tests are available;
- dynamic and opaque buffer transport.

### 12.5 Plan and executor tests

Cover:

- Python-to-msgpack-to-C++ round trip for every shape rule;
- multiple outputs with different descriptors and byte sizes;
- block shape resolution on non-cubic boxes;
- fixed rank-1/rank-2 outputs;
- rank-4 block outputs;
- like-input type-preserving and type-changing outputs;
- dynamic upper-bound resolution;
- descriptor-based input/output memory estimates;
- rejection of `output_bytes` after the atomic migration;
- absence of storage inference from kernel params.

### 12.6 Mixed dtype tests

At minimum:

- histogram1D values f32 and weights f64;
- histogram2D x f32, y f64, and both weight widths;
- spherical flux with density f64 and momentum f32;
- cylindrical flux with representative mixed fields;
- field expression with several mixed inputs;
- one high-arity profile/flux kernel with a nontrivial f32/f64 pattern;
- all-f32 and all-f64 paths.

For the variadic visitor itself, enumerate all type patterns through a small arity such as 4. Compilation of the arity-10 benchmark proves availability of all 1,024 packs without running every scientific combination.

### 12.7 Dynamic and opaque tests

Cover:

- empty particle chunk;
- particle filter selecting none, some, and all values;
- backend particle load actual length smaller than bound;
- value-count map encode/decode;
- AMR patch pack with zero, one, and multiple patches;
- dynamic upper-bound overflow rejection;
- opaque-to-numeric access rejection.

### 12.8 Existing behavioral regression tests

Retain and run:

- projection and slice correctness;
- AMR covered-cell masking;
- gradient and vorticity coarse/fine behavior;
- subbox fetch;
- flux-surface and histogram operators;
- particle API;
- executor streaming and memory-accounting tests;
- runtime output visibility;
- smoke demo.

## 13. Performance Verification

### 13.1 Compile-time benchmark

Add a reproducible utility that:

- generates or compiles the representative variadic kernel for arities 1-10;
- uses the active project compiler and Release flags;
- reports wall time and object size;
- records compiler identity and target architecture;
- provides paired non-combinatorial baselines.

Reference acceptance on the measured Clang 19 ARM64 host:

- arity 9: no more than 20 seconds;
- arity 10: no more than 30 seconds;
- large regressions require investigation but are not necessarily cross-compiler failures.

Do not enforce host-specific wall-clock thresholds in ordinary CI.

### 13.2 Generated-code check

For runtime-strided f64 loops, confirm optimized output contains native scalar loads/stores and no out-of-line `memcpy` call in the inner loop. Vectorization equivalent to direct typed pointers is not required unless the loop uses a future precompiled contiguous-layout specialization.

Repeat for:

- scalar load/store;
- simple AXPY-style loop;
- a strided logical view;
- mixed f32/f64 visitation.

### 13.3 Runtime microbenchmarks

Compare before/after:

- uniform projection on IJK input;
- uniform projection on zero-copy KJI plotfile input;
- histogram2D homogeneous and mixed dtype;
- flux-surface homogeneous and mixed dtype;
- subbox extraction;
- particle mask/filter.

Measure:

- kernel wall time;
- bytes copied;
- allocation count;
- COW detachment count;
- zero-copy slice retention;
- task memory estimates versus actual bytes.

Performance expectations:

- no new per-element virtual call;
- no per-element dtype or layout branch;
- no forced plotfile transpose or copy;
- no extra allocation for static outputs beyond existing executor allocation;
- mixed dtype costs only one dispatch per input before the loop.

## 14. Specification and Documentation Updates

Update these files as the corresponding implementation lands:

### `spec/core-architecture.md`

- replace raw `HostView` kernel arguments with **Chunk Buffers**;
- replace `output_bytes` with output **Buffer Specifications**;
- document static/dynamic allocation ownership;
- state that kernel params contain no storage metadata.

### `spec/data-models.md`

- replace context-dependent raw-byte semantics;
- define concrete descriptors, scalar types, ranks, and logical layout;
- replace subbox bytes-per-value with descriptor propagation;
- define opaque payload semantics.

### `spec/backends-and-io.md`

- require self-described backend chunks;
- describe logical indexing rather than mandatory eager transposition;
- document zero-copy plotfile KJI slices;
- document typed memory-backend writes;
- add backend descriptor/estimate requirements.

### `spec/validation-and-errors.md`

- add descriptor and specification validation;
- add dtype/rank mismatch errors;
- add dynamic bound errors;
- require pre-loop failure rather than silent empty output;
- remove output-byte and inferred-width validation language.

### `spec/performance-requirements.md`

- require descriptor-based memory accounting;
- include dynamic upper bounds in memory-cap admission;
- require logical-layout access without forced copy;
- include COW and buffer allocation telemetry where useful.

### `spec/glossary.md` and `CONTEXT.md`

- keep **Chunk Buffer**, **Buffer Specification**, **Block Grid**, and **Opaque Payload** terminology synchronized;
- remove `HostView` and raw byte-width terminology once migration completes.

### `README.md`

- update runtime output retrieval examples to omit caller-provided shape/dtype when available from descriptors;
- replace byte-width options with dtype choices;
- document explicitly raw/opaque retrieval separately.

## 15. Deletion and Completion Criteria

The refactor is complete only when all of the following are true.

### 15.1 Interface criteria

- Every dataset and produced chunk is a self-describing **Chunk Buffer**.
- Every task output has a **Buffer Specification**.
- Every numeric kernel uses typed views.
- Logical block indexing is independent of physical backend layout.
- Dynamic and opaque outputs are explicit.

### 15.2 Separation criteria

- The executor does not parse kernel-private parameters for storage information.
- Python lowerers do not calculate raw output byte counts.
- Scientific kernel params do not contain `bytes_per_value`, input widths, output widths, shapes, or layouts.
- The data service transports and allocates descriptors but does not infer them.
- Backend adapters are the only modules that translate source storage layout into concrete descriptors.

### 15.3 Deletion checks

These searches should return no production matches, except documented external-file parsing or benchmark fixtures:

```bash
rg "HostGridView3D|using HostView" cpp analysis
rg "output_bytes" cpp analysis
rg "input_bytes_per_value|out_bytes_per_value" cpp analysis
rg '"bytes_per_value"' analysis/ops.py analysis/pipeline.py cpp/src/runtime.cpp cpp/src/executor.cpp
```

Numeric kernel access should not contain raw typed casts:

```bash
rg "reinterpret_cast<(const )?(float|double|int64_t)" cpp/src/runtime.cpp
```

Any surviving match must be justified as an external binary codec and isolated behind an adapter or opaque codec module.

### 15.4 Verification criteria

- Focused chunk-buffer contract tests pass.
- Mixed-width histogram and flux regressions pass.
- AMR gradient, vorticity, projection, and subbox tests pass.
- Particle and opaque payload tests pass.
- Executor memory estimates match descriptors and dynamic bounds.
- Full `pixi run test` passes.
- Release build completes with the accepted arity-10 compile-time envelope.
- Plotfile zero-copy behavior is preserved.
- Generated-code checks show no abstraction penalty in representative inner loops.

## 16. Risks and Mitigations

### Compile-time growth

**Risk:** `2^N` real dtype combinations increase compile time.

**Decision:** accepted through `N=10`.

**Mitigation:** centralize visitation, keep helper code non-templated where it does not depend on input types, monitor with the compile benchmark, and avoid adding scalar alternatives to the real visitor.

### Template code size

**Risk:** high-arity kernels increase `_core` binary size.

**Decision:** a few additional MiB per high-arity translation unit are acceptable.

**Mitigation:** keep common geometry/codec helpers out of the templated loop body when doing so does not reintroduce per-element type erasure.

### Copy-on-write view invalidation

**Risk:** mutable access can detach storage and invalidate existing borrowed views.

**Mitigation:** detach before returning a mutable view, document lifetime rules, and test sibling slices and move/resize cases.

### Dynamic output underestimation

**Risk:** an incorrect upper bound breaks memory accounting or truncates output.

**Mitigation:** checked conservative formulas, explicit overflow errors, actual-versus-estimated telemetry, and focused worst-case tests.

### Plotfile zero-copy regression

**Risk:** attaching descriptors accidentally forces transposition or copy.

**Mitigation:** describe KJI with strides, test shared storage identity/COW behavior, and benchmark bytes copied.

### Metadata drift between Python and C++

**Risk:** dtype or shape encodings diverge.

**Mitigation:** one serialized vocabulary, plan round-trip tests, one Python NumPy mapping, and strict C++ decode validation.

### Long dual-path migration

**Risk:** carrying raw and typed buffer paths creates more complexity than it removes.

**Mitigation:** temporary compatibility is milestone-local, not a supported interface; Milestone 7 deletes the legacy surface before completion.

## 17. Recommended Commit Slices

Keep commits narrow enough to review and bisect:

1. `define chunk buffer contracts and tests`
2. `propagate chunk descriptors through memory backend`
3. `preserve descriptors through data service and HPX`
4. `describe plotfile openpmd and parthenon chunks`
5. `replace output bytes with buffer specifications`
6. `allocate executor outputs from buffer specifications`
7. `migrate mixed-width histogram kernels`
8. `migrate variadic flux and profile kernels`
9. `migrate block grid and reduction kernels`
10. `migrate particle and opaque buffer paths`
11. `remove legacy byte-width storage interfaces`
12. `synchronize specs docs and benchmarks`

Each commit must include the smallest focused verification appropriate to its surface. Do not combine unrelated numerical changes with this refactor.

## 18. Verification Commands

Use the smallest relevant command while developing each milestone, then run the full sequence before declaring the refactor complete.

### Python-only plan and lowering changes

```bash
pixi run pytest -q tests/test_pipeline_api.py tests/test_medium_spec_gaps.py
```

### Buffer, backend, data-service, and subbox changes

```bash
pixi run build
pixi run install
pixi run pytest -q tests/test_data_service_async.py tests/test_subbox_fetch.py tests/test_plotfile_reader.py
```

### Histogram tracer bullet

```bash
pixi run build
pixi run install
pixi run pytest -q tests/test_histogram_ops.py
```

### Flux and high-arity dtype-pack migration

```bash
pixi run build
pixi run install
pixi run pytest -q tests/test_flux_surface_integral.py tests/test_plotfile_flux_surface.py
```

### AMR grid migration

```bash
pixi run build
pixi run install
pixi run pytest -q \
  tests/test_projection_op.py \
  tests/test_amr_gradient.py \
  tests/test_amr_vorticity.py \
  tests/test_subbox_fetch.py
```

### Particle and opaque payload migration

```bash
pixi run build
pixi run install
pixi run pytest -q tests/test_particle_api.py tests/test_data_service_async.py
```

### Full verification

```bash
pixi run configure
pixi run build
pixi run install
pixi run test
```

Also run the dtype-pack compile benchmark and scalar-proxy code-generation check created in Milestone 0 using the active Release compiler and flags. Record compiler identity, target architecture, arity-9/10 wall time, and resulting object size in the implementation PR.
