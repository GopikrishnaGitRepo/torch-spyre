import torch
import torch_spyre
from torch_spyre._inductor.wsr.for_each_tile import Gather, for_each_tile

# =========================================================================
# CONFIGURATION & CONSTANTS
# =========================================================================
PAGE_POOL_MAX = 16        # Traced upper bound for pages pool
PAGE_BLOCKS_MAX = 4       # Traced upper bound for table blocks (max_count)
PAGE_SIZE = 32            # Rows per page
PAGE_HS = 64              # Head dimension (64 FP16 elements = 1 hardware stick)
PAGE_LQ = 32              # Query length
INT32_STICK = 32          # 32 int32 elements per stick (128 bytes)
PAGE_ORDER = (5, 2, 7, 0) # Page access schedule


def create_inputs(num_pages: int, num_blocks: int):
    """Build inputs for a specific pool size and table block count."""
    torch.manual_seed(0)
    pages = torch.empty(num_pages, PAGE_SIZE, PAGE_HS, dtype=torch.float16)
    torch.nn.init.xavier_uniform_(pages)

    q = torch.empty(PAGE_LQ, PAGE_HS, dtype=torch.float16)
    torch.nn.init.xavier_uniform_(q)

    table = torch.zeros(num_blocks, INT32_STICK, dtype=torch.int32)
    for i, page in enumerate(PAGE_ORDER[:num_blocks]):
        # Ensure page index is strictly within the current num_pages range
        table[i, 0] = page % num_pages
    return pages, table, q


def paged_gather_fn(pages: torch.Tensor, table: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Paged attention gather function with for_each_tile."""
    def body(acc, tiles):
        table_row, pages_all, q_whole = tiles
        page_idx = table_row[0, 0:1]
        page = pages_all.index_select(0, page_idx).squeeze(0)
        scores = q_whole @ page.transpose(0, 1)
        return acc + scores @ page, None

    acc0 = torch.zeros(PAGE_LQ, PAGE_HS, device=q.device, dtype=q.dtype)
    final, _ = for_each_tile(
        body,
        (table, pages, q),
        dims=(0, None, None),  # table is SLICED on dim 0; pages and q are INVARIANT
        tile_size=1,
        init=acc0,             # Accumulator carry of static shape [PAGE_LQ, PAGE_HS]
    )
    return final


def paged_scatter_fn(
    cache: torch.Tensor,
    table: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """Paged attention scatter function with for_each_tile.

    Writes each row of ``updates`` into the cache at the page address supplied
    by the matching entry in ``table``.  Mirrors ``paged_gather_fn`` but in the
    write direction:

    For step i:
      page_idx  = table[i, 0]          -- 1-element gather from the table row
      cache[page_idx] = updates[i]     -- in-place scatter into the pool

    ``cache``   : [PAGE_POOL_MAX, PAGE_SIZE, PAGE_HS] -- the mutable page pool
                  (invariant to the loop; written in-place each step)
    ``table``   : [num_blocks, INT32_STICK] -- sliced on dim 0, one row per step
    ``updates`` : [num_blocks, PAGE_SIZE, PAGE_HS] -- sliced on dim 0, one row
                  per step; each slice is written into the cache page pointed to
                  by the corresponding table entry

    NOTE: ``out_dim=Gather`` is not yet implemented in for_each_tile (see
    for_each_tile.py line ~307).  The scatter is therefore expressed as an
    in-place ``index_copy_`` inside the loop body with ``cache`` passed as an
    INVARIANT operand.  The function returns the mutated cache so the compiled
    graph's output captures the write.
    """
    def body(carry, tiles):
        # tiles arrives in operand order: (table_row, cache_all, update_row)
        table_row, cache_all, update_row = tiles
        # table_row : [1, INT32_STICK]  (one-row slice, rank-preserving)
        # update_row: [1, PAGE_SIZE, PAGE_HS] (one-row slice, rank-preserving)
        page_idx = table_row[0, 0:1]                   # shape [1], dtype int32
        cache_all.index_copy_(0, page_idx, update_row) # scatter into pool
        return carry, None

    acc0 = torch.zeros((), dtype=torch.int64, device=cache.device)
    for_each_tile(
        body,
        (table, cache, updates),
        dims=(0, None, 0),  # table SLICED on dim 0;
                            # cache INVARIANT (written in-place);
                            # updates SLICED on dim 0
        tile_size=1,
        init=acc0,          # reduction mode (no per-step output tile)
        out_dim=None,
    )
    return cache


def main():
    print("=" * 80)
    print("TEST: Symbolic Value Tensor (pages) + Concrete Trip Count on Spyre")
    print("=" * 80)

    # Only mark the VALUE tensor (pages pool) as dynamic on dim 0.
    # The TABLE tensor's dim 0 drives for_each_tile's trip count; symbolic
    # trip counts are not yet supported end-to-end (splice_while_loops requires
    # a concrete integer loop count). Only pages.shape[0] (pool capacity) is
    # the symbolic batch dim that GAP 1 in pass_utils.py targets.
    print("\n[Step 1] Compiling single dynamic kernel with mark_dynamic on pages (pool size only)...")
    dynamic_compiled = torch.compile(paged_gather_fn, backend="inductor", fullgraph=True)

    # num_blocks is FIXED at PAGE_BLOCKS_MAX so the trip count is concrete.
    # num_pages varies across calls to exercise the symbolic pool dimension.
    test_configs = [
        # (num_pages, num_blocks)
        (8, PAGE_BLOCKS_MAX),
        (12, PAGE_BLOCKS_MAX),
        (PAGE_POOL_MAX, PAGE_BLOCKS_MAX),
    ]

    walked = {}
    for num_pages, num_blocks in test_configs:
        pages, table, q = create_inputs(num_pages, num_blocks)
        # Reserve storage at the ceiling; mark only pages dim 0 as symbolic.
        pages_spyre = pages.to("spyre", max=PAGE_POOL_MAX)
        table_spyre = table.to("spyre")   # concrete trip count, no mark_dynamic
        q_spyre = q.to("spyre")

        torch._dynamo.mark_dynamic(pages_spyre, 0, min=4, max=PAGE_POOL_MAX)

        print(f"  -> Executing dynamic kernel for pages.shape={tuple(pages.shape)}, table.shape={tuple(table.shape)}...")
        out = dynamic_compiled(pages_spyre, table_spyre, q_spyre)
        walked[(num_pages, num_blocks)] = out.cpu().float()
        print(f"     [PASS] Execution successful for pool={num_pages}, blocks={num_blocks}")

    # 2. Verify numerical accuracy against static reference execution
    print("\n[Step 2] Verifying numerical accuracy against static reference compilation...")
    torch._dynamo.reset()
    static_compiled = torch.compile(
        paged_gather_fn, backend="inductor", fullgraph=True, dynamic=False
    )

    for num_pages, num_blocks in test_configs:
        pages, table, q = create_inputs(num_pages, num_blocks)
        expected = static_compiled(pages.to("spyre"), table.to("spyre"), q.to("spyre"))
        torch.testing.assert_close(
            walked[(num_pages, num_blocks)],
            expected.cpu().float(),
            atol=1e-2,
            rtol=1e-3,
        )
        print(f"  -> Numerics MATCH for pool={num_pages}, blocks={num_blocks}")

    # =========================================================================
    # SCATTER TEST: write updated pages back into the pool via for_each_tile
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST: Paged scatter via for_each_tile (in-place index_copy_ per step)")
    print("=" * 80)

    # The scatter kernel also uses table as the tiled (trip-count) operand,
    # so num_blocks must be fixed for the same reason as the gather test above.
    print("\n[Step 3] Compiling single dynamic scatter kernel...")
    torch._dynamo.reset()
    dynamic_scatter = torch.compile(paged_scatter_fn, backend="inductor", fullgraph=True)

    scatter_configs = [
        # (num_pages, num_blocks)  -- num_blocks FIXED so trip count is concrete
        (8, PAGE_BLOCKS_MAX),
        (12, PAGE_BLOCKS_MAX),
        (PAGE_POOL_MAX, PAGE_BLOCKS_MAX),
    ]

    for num_pages, num_blocks in scatter_configs:
        torch.manual_seed(42)
        # Build a zero-initialised cache and a set of update pages.
        cache_cpu = torch.zeros(num_pages, PAGE_SIZE, PAGE_HS, dtype=torch.float16)
        updates_cpu = torch.rand(num_blocks, PAGE_SIZE, PAGE_HS, dtype=torch.float16)
        table_cpu = torch.zeros(num_blocks, INT32_STICK, dtype=torch.int32)
        for i, page in enumerate(PAGE_ORDER[:num_blocks]):
            table_cpu[i, 0] = page % num_pages

        cache_spyre = cache_cpu.to("spyre", max=PAGE_POOL_MAX)
        table_spyre = table_cpu.to("spyre")
        updates_spyre = updates_cpu.to("spyre")

        # Only the cache (destination) dim 0 is symbolic — the pool capacity.
        # The table's dim 0 is the trip count and must stay concrete.
        torch._dynamo.mark_dynamic(cache_spyre, 0, min=4, max=PAGE_POOL_MAX)

        print(
            f"  -> Scatter for cache.shape={tuple(cache_cpu.shape)}, "
            f"table.shape={tuple(table_cpu.shape)}..."
        )
        result_spyre = dynamic_scatter(cache_spyre, table_spyre, updates_spyre)

        # CPU reference: apply the same scatter eagerly.
        cache_ref = cache_cpu.clone()
        for i in range(num_blocks):
            page_idx = table_cpu[i, 0].item()
            cache_ref[page_idx] = updates_cpu[i]

        torch.testing.assert_close(
            result_spyre.cpu().float(),
            cache_ref.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        print(
            f"     [PASS] Scatter numerics MATCH for pool={num_pages}, "
            f"blocks={num_blocks}"
        )

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED: Symbolic value tensor batch dimension successfully verified!")
    print("=" * 80)


if __name__ == "__main__":
    main()
