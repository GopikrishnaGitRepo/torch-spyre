# Phase 1 Symbolic / Dynamic Shape Support — torch-spyre

**Date:** 2026-10-03  
**Branch:** `runtime_max_alloc`

---

## Overview

Phase 1 adds **compile-once, run-at-many-batch-sizes** support to torch-spyre. A user marks tensor dimension 0 as dynamic with an upper bound; the compiler wraps the operation in a tiled `for_each_tile` loop whose trip count is `batch_size // tile_size`. Because the batch size is unknown at compile time, the loop bound becomes a runtime parameter — a `kDimension` SymbolicArg — that is read from the actual tensor shape at each dispatch.

### User-facing API

```python
import torch
import torch._dynamo as dynamo

x = torch.rand(64, 1024, dtype=torch.float16)
x_device = x.to("spyre")
dynamo.mark_dynamic(x_device, dim=0, min=64, max=576)

compiled_fn = torch.compile(gelu_fn)
result = compiled_fn(x_device)          # batch=64
result = compiled_fn(x_device[:128])    # batch=128, same compiled kernel
result = compiled_fn(x_device[:512])    # batch=512, same compiled kernel
```

Or via the new `dynamic=` kwarg on `.to("spyre")`:

```python
x_device = x.to("spyre", dynamic={0: {"min": 64, "max": 576}})
```

### Scope (Phase 1)

| Supported | Not Supported |
|-----------|---------------|
| Single dynamic outer axis (dim 0) | Inner / non-outermost dynamic axes |
| `map` operations (gelu, softmax, etc.) | Reductions along the dynamic axis |
| One symbolic variable per kernel | Multiple simultaneous dynamic batch dims |
| `for_each_tile` tiled loops | Direct SDSC dynamic-dim path (Phase 2) |

---

## Architecture: Data Flow

```
User: mark_dynamic(t, 0, min=G, max=M)
          │
          ▼
_monkey_patch.py: spyre_to(dynamic={...})
  allocates max-size tensor, calls mark_dynamic
          │
          ▼  torch.compile
decompositions.py: spyre_gelu (and similar)
  _get_dynamic_outer_dim_info(input) → (G, M)
  wraps op in for_each_tile(_body, tile_size=G)
          │
          ▼
for_each_tile_lowering.py
  try_prove_for_each_tile →
  _extract_trip_count → sympy.Symbol s0
  (SymInt.node.expr extracted here)
          │
          ▼
spyre_kernel.py: wrap_op_specs_in_loop(FloorDiv(s0, G))
  populates LoopSpec.symbolic_dim_bounds
  {"s0": (M, G, "s0", tensor_id, 0)}
          │
          ▼
codegen/bundle.py: generate_bundle
  _compile_specs collects loop_dim_kinds
  injects SymbolKind.dimension(gran=G, max=M, pytorch_sym="s0")
  _emit_symbolic_loop_bound:
    arith.constant G : index
    arith.divsi %sym_N_K, %trip_divisor : index
          │
  bundle.mlir:
    func.func @sdsc_bundle(
      %sym_1_1_base: !sdscbundle.input_arg<index, granularity=G, max_value=M>
    ) {
      %sym_1_1 = sdscbundle.input_arg_extract …
      %trip_divisor = arith.constant G : index
      %loop_bound = arith.divsi %sym_1_1, %trip_divisor : index
      scf.for %i = %c0 to %loop_bound step %c1 { … }
    }
          │
          ▼
async_compile.py: generate_bundle() → symbol_kinds
  (NotImplementedError guard removed)
          │
          ▼
kernel_runner.py: SpyreSDSCKernelRunner.__init__
  dimension symbol → None sentinel in _symbolic_args
  _has_dim_args = True
          │
  .run(*args):
    reads args[tensor_id].shape[dim_index]
    builds SymbolicArg(kDimension, tid, dim_idx, value=actual_size)
    calls launch_jobplan(jobplan, args, resolved_sym_args)
          │
          ▼
C++ job_plan.cpp: JobPlanStepHostCompute::construct
  for sym in ctx.symbolic_args:
    kDimension → args.push_back(static_cast<int64_t>(sym.value))
    kAddress   → args.push_back(get_composite_address(…))
  flex::createHostComputeParams(…, args, …)
```

---

## File-by-file Changes

| File | What Changed | Why |
|------|-------------|-----|
| `torch_spyre/_inductor/op_spec.py` | Added `symbolic_dim_bounds: dict` field to `LoopSpec` | Carries `{str(sym): (max_val, gran, pytorch_sym, tensor_id, dim_idx)}` through the compilation pipeline to `bundle.py` |
| `torch_spyre/_inductor/wsr/for_each_tile_lowering.py` | `_extract_trip_count`: handles `torch.SymInt` by extracting `.node.expr` | Dynamo traces the `for_each_tile` condition with a SymInt; the integer-only check rejected it |
| `torch_spyre/_inductor/spyre_kernel.py` | `wrap_op_specs_in_loop`: populates `symbolic_dim_bounds` when count has free symbols; post-processes `tensor_id=-1` sentinel after arg_index assignment | Symbolic trip counts require the dimension bounds to be propagated to `bundle.py`; `arg_index` is assigned later so sentinel needed |
| `torch_spyre/_inductor/codegen/compute_ops.py` | Added `dim_index: int = 0` field to `SymbolKind`; updated `SymbolKind.dimension()` factory to accept `tensor_id` and `dim_index` | `kernel_runner.py` needs to know which tensor and which axis to read at dispatch time |
| `torch_spyre/_inductor/codegen/bundle.py` | `_compile_specs`: gathers `loop_dim_kinds` from `LoopSpec.symbolic_dim_bounds`; `generate_bundle`: injects them as synthetic MLIR params; added `_emit_symbolic_loop_bound` for `arith.divsi` emission; fixed `sympy.Integer`-before-`Rational` divisor extraction bug | Loop-spec dimension symbols are not tied to any SDSC — they need their own MLIR parameters; plain `s0 // 64` in sympy creates `floor(s0/64)` (wrong), so the emitter checks `Integer` before `Rational` |
| `torch_spyre/_inductor/decompositions.py` | Added `_get_dynamic_outer_dim_info(tensor)` utility; updated `spyre_gelu` to use `for_each_tile` when dim 0 is dynamic | The utility reads `ShapeEnv.bound_sympy()` directly, avoiding the need for `V.graph` during decomposition tracing |
| `torch_spyre/execution/async_compile.py` | Removed `NotImplementedError` guard for dimension symbols; updated docstring | kDimension dispatch is now wired end-to-end |
| `torch_spyre/execution/kernel_runner.py` | `__init__`: dimension symbols → `None` sentinel, `_has_dim_args` flag, `pool_offset` logic; `run()`: resolves `None` sentinels to `SymbolicArg(kDimension, value=actual_shape)` | Actual tensor sizes are only known at dispatch time; the sentinel defers the `SymbolicArg` construction |
| `torch_spyre/_monkey_patch.py` | Added `dynamic=` kwarg to `spyre_to()`; allocates at max size, copies data, calls `mark_dynamic` | Convenience API for users |
| `torch_spyre/csrc/job_plan.cpp` | `JobPlanStepHostCompute::construct`: added `kDimension` branch that pushes `static_cast<int64_t>(sym.value)` into the `flex::HostComputeArg` variant | `flex::HostComputeArg = std::variant<const CompositeAddress*, int64_t>`; the `int64_t` arm was unused before this change |

---

## Key Challenges and Fixes

### 1. `sympy.Integer` is a `Rational` (divisor bug)

**Symptom:** `_emit_symbolic_loop_bound` emitted `arith.constant 0 : index` for the divisor of `FloorDiv(s0, 64)`.

**Root Cause:** `sympy.Integer(64)` is a subclass of `sympy.Rational`. The original check:
```python
if isinstance(arg1, sympy.Rational):
    divisor = int(1 / arg1)  # int(1/64) = 0 !
```
matched `Integer(64)` and computed `int(1/64) = 0`.

**Fix:** Check `Integer` before `Rational`:
```python
if isinstance(arg1, (sympy.Integer, int)):
    divisor = int(arg1)       # 64
elif isinstance(arg1, sympy.Rational):
    divisor = int(1 / arg1)   # only for true fractions like Rational(1, 64)
```

### 2. `_extract_trip_count` rejected `torch.SymInt` bounds

**Symptom:** Dynamic `for_each_tile` loops returned `None` trip count, falling back to legacy unrolling.

**Root Cause:** The recorder's `constants` list holds a `torch.SymInt`, not a `sympy.Expr`. The existing `isinstance(bound, (int, sympy.Expr))` check returned False.

**Fix:** Extract `bound.node.expr` before the isinstance check:
```python
if isinstance(bound, torch.SymInt):
    bound = bound.node.expr
```

### 3. `tensor_id` unknown at `wrap_op_specs_in_loop` time

**Symptom:** `symbolic_dim_bounds` needed to reference the first input tensor, but `arg_index` is assigned later in `codegen_kernel()`.

**Fix:** Use sentinel `tensor_id=-1` in `wrap_op_specs_in_loop`, then post-process in `codegen_kernel()` after the arg_index assignment loop to replace `-1` with the first input tensor's `arg_index`.

### 4. `sympy.floor(s0/64)` vs. `FloorDiv(s0, 64)`

**Symptom:** Tests used `s0 // 64` (plain Python with sympy) which creates `floor(s0/64)` with `.args = (s0/64,)` — a single-argument expression. `_emit_symbolic_loop_bound` expected `.args = (s0, 64)`.

**Fix:** Tests were updated to use `torch.utils._sympy.functions.FloorDiv(s0, 64)`, matching what torch's compilation pipeline actually produces.

### 5. C++ brace mismatch in `job_plan.cpp`

**Symptom:** `expected '}' before 'else'` compile error after adding the `kDimension` branch.

**Root Cause:** The Edit tool's `old_string` captured up to the last statement of the for loop body, but the `new_string` did not include the for loop's closing `}`, leaving it to be closed by the outer `if` block's brace.

**Fix:** Added the missing for-loop closing brace before the `} else {` of the outer conditional.

### 6. `_get_dynamic_outer_dim_info` cannot use `V.graph`

**Symptom:** Decompositions run during Dynamo's `make_fx` / AOT tracing before `V.graph` is populated.

**Fix:** Read bounds directly from `tensor.shape[0].node.shape_env.bound_sympy(expr)`, bypassing `V.graph` entirely.

---

## Testing

New test file: `tests/inductor/test_symbolic_shapes.py` — 26 tests, no hardware required.

| Test Group | Tests | What It Covers |
|-----------|-------|----------------|
| `TestLoopSpecSymbolicDimBounds` | 3 | `LoopSpec.symbolic_dim_bounds` field stores correctly, defaults to `{}` |
| `TestSymbolKindDimension` | 4 | `SymbolKind.dimension()` factory with `tensor_id`/`dim_index`; `is_dimension` property |
| `TestLoopSpecBundleMlirGeneration` | 7 | `generate_bundle` with a `LoopSpec`: `arith.divsi`, divisor constant, `input_arg` param, `input_arg_extract`, SSA name in `divsi`, deduplication of same `pytorch_sym`, no `divsi` for static count |
| `TestKernelRunnerKDimensionDispatch` | 5 | `_has_dim_args`, `None` sentinel, `launch_jobplan` called with resolved `kDimension` arg at correct `shape[dim_index]` value, no-sym-args path |
| `TestGetDynamicOuterDimInfo` | 3 | Returns `None` for static/default-bound tensors; returns `(gran, max_val)` for user-set bounds |
| `TestExtractTripCountWithSymInt` | 3 | `sympify` of int; sympy Expr; SymInt `.node.expr` extraction logic |
| `TestMarkDynamicNoLongerExpectedToFail` | 1 | `mark_dynamic.py` has a success path (kDimension is now wired) |

**Results:**
```
26 passed in 6.08s
```

No regressions in existing tests:
- `test_symbolic_dim_bundle.py`: 21 passed
- `test_for_each_tile_lowering.py` + `test_for_each_tile.py`: 111 passed

---

## Limitations (Phase 1)

1. **Outer axis only.** Only `dim=0` is supported. Inner dynamic dimensions (sequence length, hidden size) are Phase 2.

2. **Map operations only.** Operations with a reduction along the dynamic axis (e.g., `mean(dim=0)`) are not supported. `for_each_tile` is not applicable to reductions that cross tile boundaries.

3. **Single dynamic dimension per kernel.** If two tensors in the same kernel have different dynamic batch sizes, only the first is used. Multiple simultaneous dynamic batch dims require Phase 2 SDSC support.

4. **Decomposition-by-decomposition opt-in.** Only `spyre_gelu` has the dynamic wrapper. Other operations (`softmax`, `softplus`, etc.) need the same `_get_dynamic_outer_dim_info` check added individually.

5. **`mark_dynamic.py` test is still expected to fail.** The test script `tests/dynamic_shapes/mark_dynamic.py` has a `try/except` guard from issue #2434. It now has a success path in the `else` branch, but the hardware path (actual Spyre dispatch) is not tested in this PR because the runtime is not available in CI.

6. **No automatic shape re-use.** The compiler currently does not re-use a compiled kernel for slightly different batch sizes in the same granularity bucket — each distinct size triggers a new trace unless Dynamo's guard machinery handles it.
