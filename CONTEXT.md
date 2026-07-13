# Kangaroo Analysis Runtime

Kangaroo executes distributed analysis plans over dataset fields represented as independently transportable chunks.

## Language

**Chunk Buffer**:
A self-describing host-resident payload for one field chunk, carrying its scalar type, logical shape, layout, and shared storage.
_Avoid_: HostView, raw byte buffer

**Buffer Specification**:
A plan-time storage contract that resolves to the descriptor and allocation policy of a **Chunk Buffer**.
_Avoid_: output bytes, bytes-per-value parameter

**Block Grid**:
A logical three-dimensional field view over a **Chunk Buffer**, indexed in runtime `(i, j, k)` order regardless of physical layout.
_Avoid_: raw grid pointer

**Opaque Payload**:
A **Chunk Buffer** whose packed representation has no numeric scalar or shape semantics at the buffer seam.
_Avoid_: fake numeric array

**AMR Patch Payload**:
An **Opaque Payload** containing validated AMR patch geometry, level identity, and embedded **Chunk Buffers** for stencil sampling.
_Avoid_: packed neighbor bytes

## Relationships

- A field and block identify one **Chunk Buffer** at a particular step, level, and version.
- A **Buffer Specification** resolves to one output **Chunk Buffer** for each task instance.
- A numeric rank-three **Chunk Buffer** exposes a **Block Grid**.
- An **Opaque Payload** exposes bytes but cannot expose a typed **Block Grid** or numeric array.
- An **AMR Patch Payload** transports neighboring **Chunk Buffers** without exposing its wire format to kernels.

## Example dialogue

> **Developer:** "Does this kernel need to know the input field's bytes per value?"
> **Runtime maintainer:** "No. The **Chunk Buffer** carries its scalar type, and the kernel requests a typed **Block Grid**."

## Flagged ambiguities

- "layout" previously mixed physical byte order with logical grid indexing. A **Block Grid** always uses logical `(i, j, k)` indexing; physical layout belongs to the **Chunk Buffer** descriptor.
