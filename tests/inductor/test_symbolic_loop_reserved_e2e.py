# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One compiled binary serving a range of sizes, on device.

The combined milestone that ``test_symbolic_loop_e2e.py`` defers to: a
symbolic ``for_each_tile`` loop (the front end) over a tensor whose HBM buffer
is reserved for the declared maximum by ``tensor.to("spyre", dynamic=...)``
(the runtime). Neither half proves this alone. Without the reservation every
size is a differently sized allocation, so the layout guard recompiles; without
the symbolic loop the kernel specialises to one size.

What is asserted:
  * the user function compiles ONCE for every multi-tile size in the range,
    whether each call brings a fresh reserved tensor or one reserved buffer is
    resized in place;
  * the numbers are right at every size, including sizes above the warm-up
    (the case hint-sized geometry gets wrong);
  * the launch payload carries a kDimension slot that resolves to the LOGICAL
    row count of the call, not the reserved maximum.

A single tile (rows == TILE) compiles separately; that is the known
``Ne(count, 1)`` split documented in ``for_each_tile._stacked_to_full``.
"""

import unittest
from unittest.mock import patch

import torch
from torch._inductor.compile_fx import compile_fx

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._C import (
    SymbolicArgKind,
    _resolve_symbolic_args,
    get_reserved_dims,
    get_spyre_tensor_layout,
)
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.execution import kernel_runner

COLS = 64
TILE = 64
MIN_ROWS = 64
MAX_ROWS = 512
WARMUP_ROWS = 256
SPEC = {0: {"min": MIN_ROWS, "max": MAX_ROWS, "granularity": TILE}}

ATOL = 1e-2
RTOL = 1e-2


def tiled_gelu(x):
    rows = x.shape[0]
    torch._check(rows >= MIN_ROWS)
    torch._check(rows <= MAX_ROWS)
    torch._check(rows % TILE == 0)

    def body(_carry, tiles):
        (tile,) = tiles
        return None, torch.nn.functional.gelu(tile)

    _carry, out = for_each_tile(body, (x,), dims=(0,), tile_size=TILE, out_dim=0)
    return out


class _CountingCompile:
    """``torch.compile(tiled_gelu)`` that counts only ITS OWN compiles.

    Dynamo's global counters also count the standalone per-op kernels eager
    Spyre ops compile (e.g. a d2d ``copy_``), which would make "one compile"
    unassertable for the in-place pattern.
    """

    def __init__(self):
        self.compiles = 0

        def backend(gm, example_inputs):
            self.compiles += 1
            return compile_fx(gm, example_inputs)

        self.fn = torch.compile(tiled_gelu, backend=backend, fullgraph=True)


def _reference(x):
    return torch.nn.functional.gelu(x.float())


class TestOneBinaryAcrossSizes(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_fresh_reserved_tensor_per_call(self):
        compiled = _CountingCompile()
        for rows in [WARMUP_ROWS, 128, MAX_ROWS, 192, 448]:
            x = torch.randn(rows, COLS, dtype=torch.float16)
            out = compiled.fn(x.to(DEVICE_NAME, dynamic=SPEC))
            self.assertEqual(tuple(out.shape), (rows, COLS))
            torch.testing.assert_close(
                out.cpu().float(), _reference(x), atol=ATOL, rtol=RTOL
            )
        self.assertEqual(
            compiled.compiles,
            1,
            "a reserved, marked-dynamic input recompiled within its declared range",
        )

    def test_one_buffer_resized_in_place(self):
        """The serving pattern: allocate at max once, resize per request."""
        compiled = _CountingCompile()
        buf = torch.zeros(MAX_ROWS, COLS, dtype=torch.float16).to(
            DEVICE_NAME, dynamic=SPEC
        )
        ptr = buf.untyped_storage().data_ptr()
        nbytes = buf.untyped_storage().nbytes()
        layout = str(get_spyre_tensor_layout(buf))

        for rows in [320, 128, MAX_ROWS, WARMUP_ROWS]:
            x = torch.randn(rows, COLS, dtype=torch.float16)
            buf.resize_(rows, COLS)
            buf.copy_(x.to(DEVICE_NAME))  # device-to-device
            out = compiled.fn(buf)
            torch.testing.assert_close(
                out.cpu().float(), _reference(x), atol=ATOL, rtol=RTOL
            )
            self.assertEqual(buf.untyped_storage().data_ptr(), ptr)
            self.assertEqual(buf.untyped_storage().nbytes(), nbytes)
            self.assertEqual(str(get_spyre_tensor_layout(buf)), layout)
        self.assertEqual(compiled.compiles, 1)

    def test_single_tile_is_correct(self):
        compiled = _CountingCompile()
        x = torch.randn(MIN_ROWS, COLS, dtype=torch.float16)
        out = compiled.fn(x.to(DEVICE_NAME, dynamic=SPEC))
        torch.testing.assert_close(
            out.cpu().float(), _reference(x), atol=ATOL, rtol=RTOL
        )


class TestLaunchPayloadBindsLogicalSize(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_dimension_slot_is_the_logical_row_count(self):
        compiled = _CountingCompile()
        launches = []
        original = kernel_runner.launch_jobplan

        def record(job_plan, args, symbolic_args=()):
            launches.append((list(args), list(symbolic_args)))
            if symbolic_args:
                return original(job_plan, args, symbolic_args)
            return original(job_plan, args)

        with patch.object(kernel_runner, "launch_jobplan", side_effect=record):
            for rows in [WARMUP_ROWS, 128, MAX_ROWS]:
                launches.clear()
                x = torch.randn(rows, COLS, dtype=torch.float16)
                dev = x.to(DEVICE_NAME, dynamic=SPEC)
                self.assertEqual(get_reserved_dims(dev)[0]["max"], MAX_ROWS)
                compiled.fn(dev)

                dims = [
                    (args, sa)
                    for args, payload in launches
                    for sa in payload
                    if sa.kind == SymbolicArgKind.kDimension
                ]
                self.assertTrue(
                    dims, "no kDimension slot was launched, so no loop is symbolic"
                )
                for args, sa in dims:
                    (value,) = _resolve_symbolic_args(args, [sa])
                    self.assertEqual(
                        value,
                        rows,
                        "the loop bound must be the call's logical size, not the "
                        f"reserved maximum ({MAX_ROWS})",
                    )
        self.assertEqual(compiled.compiles, 1)


if __name__ == "__main__":
    unittest.main()
