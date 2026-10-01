# One layer, end to end: how a symbolic dim becomes a device loop

A single worked example traced through every stage, with the actual IR at each step.

Stages 1 and 2 are proposed and not built. Everything from stage 4 onward is existing code with
our bundle changes on top.

## The example

```python
h = torch.relu(x @ W + b)      # x is (S, 1024), W is (1024, 512), b is (512,)
```

One linear layer. Small enough to follow, but it still has all three operand kinds and it produces
more than one SDSC, which is the thing people usually get wrong.

```python
x = x.to("spyre", dynamic={0: dict(min=64, max=512, granularity=64)})
```

So the tile is 64 rows and the trip count is `S // 64`, between 1 and 8.

## Stage 0: what Dynamo gives us

Three aten ops, and `x` carries a symbol.

```python
def forward(x, W, b):                                  # x: (s0, 1024)
    mm   = torch.ops.aten.mm.default(x, W)             # (s0, 512)
    add  = torch.ops.aten.add.Tensor(mm, b)            # (s0, 512)
    relu = torch.ops.aten.relu.default(add)            # (s0, 512)
    return relu
```

`s0` has range `[64, 512]` in the ShapeEnv. Nothing is specialised.

A detail that matters later: `decompose_addmm` in `CustomPostPasses` deliberately splits `addmm`
back into `add` plus `mm`, so the bias never folds into the matmul. That is why there are three ops
here and not two.

## Stage 1: Bridge 1 picks the region (proposed)

Runs as a pre-grad FX pass in `CustomPreGradPasses`.

It seeds from `x`, because `x.meta['val']` carries `s0`. Then it grows forward while the marked axis
passes through one to one:

| Node | Does `s0` pass through one to one? | In region? |
|---|---|---|
| `mm` | yes, `s0` is M. The contraction is over 1024, which is static | yes |
| `add` | yes, pointwise | yes |
| `relu` | yes, pointwise | yes |

Nothing trips the stop rule, so all three go in one region. Had there been a `sum(dim=0)` anywhere,
the region would have ended right before it.

The pass then rewrites the graph:

```python
def forward(x, W, b):
    out = torch.ops.spyre.tiled_region.default(
        [x, W, b], region_id=0, tiled_dim=0, tile_size=64
    )
    return out
```

and the three nodes move into a child module.

## What the lifted submodule actually is

It is a plain `torch.fx.GraphModule`, stored as a child of the main GraphModule, exactly like the
subgraph of any higher-order op.

```python
class tiled_region_0(torch.nn.Module):
    def forward(self, x_tile, W, b):                   # x_tile: (64, 1024)
        mm   = torch.ops.aten.mm.default(x_tile, W)    # (64, 512)
        add  = torch.ops.aten.add.Tensor(mm, b)        # (64, 512)
        relu = torch.ops.aten.relu.default(add)        # (64, 512)
        return relu
```

Two things to notice.

**Its placeholders are the tiles, not the full tensors.** The first placeholder is `(64, 1024)`, not
`(s0, 1024)`. That is the whole point. The body is written against one tile and knows nothing about
how many tiles there are. `W` and `b` are not tiled so they come through whole.

**The body contains no slicing.** `for_each_tile` hands the body tiles that are already carved, and
the body must not slice, because a data-dependent slice does not trace. Bridge 1 does the carving by
construction, simply by declaring those placeholder shapes.

The region body cannot be an argument to the op, since a custom op schema only takes tensors and
ints. So the op carries `region_id` and the body is looked up from a module-level registry. That is
the same approach `_MARKER_MAPS` already uses in `for_each_tile_lowering.py`, and the op itself is
the same idea as the existing `spyre::tile_dim_marker`, which `for_each_tile` lowering already
creates and then consumes.

`spyre.tiled_region` is a name invented for this document. There is no such op in the tree today.

## Stage 2: Bridge 2 builds the loop (proposed)

A decomposition, so it runs during AOT tracing.

```python
@register_spyre_decompositions([torch.ops.spyre.tiled_region.default])
def _decompose_tiled_region(operands, region_id, tiled_dim, tile_size):
    body_gm = _REGION_BODIES[region_id]

    _, out = for_each_tile(
        lambda carry, tiles: (None, body_gm(*tiles)),
        operands,
        dims=(tiled_dim, None, None),     # x SLICE on 0, W and b INVARIANT
        tile_size=tile_size,
        out_dim=tiled_dim,                # map mode
    )
    return out
```

`body_gm(*tiles)` is the interesting line. A `GraphModule` is callable, so calling it while AOT is
tracing **inlines its nodes into the enclosing trace**. The three aten ops become part of
`combine_fn`'s traced graph, which becomes the WhileLoop's body subgraph. We do not have to
reconstruct anything by hand.

`init` is None and `out_dim` is set, so this is map mode. Each step writes its own 64 rows and
nothing carries between steps.

`for_each_tile` calls `scan`, which decomposes to `ir.WhileLoop`.

## Stage 3: Inductor lowering

The WhileLoop's body lowers through Spyre's own lowerings. `aten.mm` has
`@register_spyre_lowering(torch.ops.aten.mm.default)`, so matmul becomes an ordinary Spyre buffer
rather than an extern kernel. That is why it can share a loop group with the pointwise ops.

## Stage 4: `splice_while_loops`, the first pre-scheduling pass

Three things happen, in order.

**Recover the trip count.** `try_prove_for_each_tile` reads it out of the cond subgraph as
`s0 // 64`. One of our two POC changes lives here: a concrete bound arrives as `constant` and a
symbolic one as `index_expr`, and the original recorder only looked at `constant`, so every symbolic
loop was silently declined.

**Inline the body.** `splice_while_loop` moves the body ops into the main graph. After this there is
no WhileLoop anywhere. The three ops are ordinary entries in `graph.operations`, each stamped with
`loop_info` carrying the same `loop_group_id`.

**Divide the ranges.** `coarse_tile_pre_stickify` rewrites every op's iteration space to the tile.
`x`'s dim 0 becomes 64. The matmul's M becomes 64.

```
before:  mm over (s0, 1024) x (1024, 512)
after:   mm over (64,  1024) x (1024, 512)      trip count s0 // 64 held separately
```

This is the pivot of the whole design. From here the symbol exists only in the loop count, and every
remaining pass sees a static tile.

## Stage 5: the rest of the pipeline, which needs no changes

Stickification, restickify, `_distribute_work`, scratchpad planning. All operate on a 64 x 1024
tile. Work division splits **that tile** across cores, which is worth stating plainly: tiles are not
spread across cores, the loop is sequential and the cores divide the work inside one trip.

The scratchpad solver decides whether the `(64, 512)` intermediate between `mm` and `relu` stays
LX-resident. It is only a candidate because both ops are in the same loop group.

## Stage 6: scheduler

Inductor builds SchedulerNodes and fuses what it can. `add` and `relu` are both pointwise on the
same tile, so they fuse into one node. The matmul stays its own node.

`_regroup_by_outer_loop_key` makes every node sharing `loop_group_id[0]` contiguous, then
`_build_loop_group` wraps them in one `CountedLoopSchedulerNode` with `count = s0 // 64`.

## Stage 7: one kernel, many op specs

```python
with kernel:
    self._codegen_into_kernel(nodes, kernel)       # each node appends its OpSpec
kernel.wrap_op_specs_in_loop(node.loop_count)      # wraps the WHOLE list
```

`wrap_op_specs_in_loop` is three lines and it is where the shape of the final bundle is decided:

```python
body = self.op_specs
self.op_specs = [LoopSpec(count=count, body=body)]
```

giving

```python
LoopSpec(
    count=FloorDiv(s0, 64),
    count_symbol_bounds={'s0': (512, 64)},     # max, granularity
    body=[
        OpSpec(op='matmul', ...),              # x_tile @ W
        OpSpec(op='relu',   ...),              # + b, relu, fused
    ],
)
```

**`LoopSpec` is a container, not an op.** Its `body` is a list, the scheduler puts the whole group
in it, and nesting is the same thing again, a `LoopSpec` holding another `LoopSpec`.

`count_symbol_bounds` is our second POC change. It is carried rather than looked up, because codegen
also runs in a reload phase where the ShapeEnv is gone.

## Stage 8: the bundle

```mlir
#map_0 = affine_map<(d0)[s0] -> (s0 + d0 * 131072)>

func.func @sdsc_bundle(
    %arg_0_base_addr: !sdscbundle.input_arg<index>,
    %arg_1_base_addr: !sdscbundle.input_arg<index>,
    %dim_s0_base: !sdscbundle.input_arg<index, granularity=64, max_value=512>) {

  %dim_s0 = sdscbundle.input_arg_extract value from %dim_s0_base
      : !sdscbundle.input_arg<index, granularity=64, max_value=512> -> index

  %tile_0 = arith.constant 64 : index
  %loop_bound_0 = arith.ceildivsi %dim_s0, %tile_0 : index

  scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
    %addr_0 = affine.apply #map_0(%i_0)[%arg_0]
    sdscbundle.sdsc_execute (%addr_0, ...) {sdsc_filename="sdsc_0.json", ...}   // matmul
    sdscbundle.sdsc_execute (...)          {sdsc_filename="sdsc_1.json", ...}   // relu
  }
  return
}
```

**Two SDSCs, one loop.** One `sdsc_execute` per OpSpec, each with its own `sdsc_N.json` describing a
static 64-row compute step.

So grouping does not merge SDSCs. The number of SDSCs is whatever Inductor fusion decides, exactly
as for a static kernel today. What grouping decides is the number of **loops**, and that is what lets
the intermediate stay in LX instead of going out to HBM and back on every trip.

## Who owns what

| Stage | Decides | Built? |
|---|---|---|
| Bridge 1, pre-grad | which ops share a loop | no |
| Bridge 2, decomposition | that the loop exists, and its tile size | no |
| `splice_while_loops` | the tile becomes static | yes |
| Inductor fusion | how many SDSCs | yes, unchanged |
| `wrap_op_specs_in_loop` | they all go in one LoopSpec | yes, unchanged |
| `generate_bundle` | `ceildivsi` and the `input_arg` | yes, our POC change |

## When the region is bigger

The two-layer version:

```python
h = torch.relu(x @ W1 + b1)
y = torch.softmax(h @ W2 + b2, dim=-1)
```

Nothing structural changes. Softmax reduces over the last axis, which is 10 and static, so `s0` still
passes through one to one and the region covers all of it. One loop, roughly four or five SDSCs
depending on fusion, and `h` at `(64, 512)` becomes a candidate to stay in LX across both matmuls.

Point that softmax at dim 0 instead and the region stops at it, because then it reduces along the
varying axis and the tiles stop being independent. That is the Phase 1 to Phase 2 line.
