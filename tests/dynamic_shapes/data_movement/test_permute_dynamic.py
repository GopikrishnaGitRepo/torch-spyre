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

# permute inside each row, on a dynamic number of rows, one compiled binary for
# every size (docs/source/compiler/symbolic_shapes.md, sections 7.3 and 18.1).

import pytest
import torch
from torch._inductor.exc import InductorError

from torch_spyre._inductor.wsr import for_each_tile

DEVICE = torch.device("spyre")

# Dim 0 varies from 64 to 512 rows, in steps of 64.
DYNAMIC = {0: dict(min=64, max=512, granularity=64)}

# First call at 256 rows, then smaller and larger sizes.
SIZES = [256, 128, 512, 192]


def permute_fn(a):
    rows = a.shape[0]
    torch._check(rows >= 64)
    torch._check(rows <= 512)
    torch._check(rows % 64 == 0)
    _, out = for_each_tile(
        lambda _, tiles: (None, tiles[0].permute(0, 3, 1, 2).contiguous()),
        (a,),
        dims=(0,),
        tile_size=64,
        out_dim=0,
    )
    return out


def test_permute_dynamic():
    compiled_fn = torch.compile(permute_fn, dynamic=None, fullgraph=True)

    for rows in SIZES:
        x = torch.rand(rows, 8, 64, 128, dtype=torch.float16)
        cpu_result = x.permute(0, 3, 1, 2).contiguous()

        x_device = x.to(DEVICE, dynamic=DYNAMIC)
        compiled_result = compiled_fn(x_device).cpu()

        # Spyre stores fp16 with slightly less precision, so a value can round by
        # one step on the way to the card and back; hence the small tolerance.
        torch.testing.assert_close(compiled_result, cpu_result, atol=1e-2, rtol=1e-2)


def permute_view_fn(a):
    rows = a.shape[0]
    torch._check(rows >= 64)
    torch._check(rows <= 512)
    torch._check(rows % 64 == 0)
    _, out = for_each_tile(
        lambda _, tiles: (None, tiles[0].permute(0, 3, 1, 2)),
        (a,),
        dims=(0,),
        tile_size=64,
        out_dim=0,
    )
    return out


@pytest.mark.xfail(
    strict=True,
    raises=InductorError,
    reason="permute without .contiguous() inside a tiled region fails to compile "
    'with "Cannot convert symbols to int"; it compiles without dynamic shapes. '
    "TODO(dynamic-shapes): no issue filed yet",
)
def test_permute_view_dynamic():
    compiled_fn = torch.compile(permute_view_fn, dynamic=None, fullgraph=True)

    for rows in SIZES:
        x = torch.rand(rows, 8, 64, 128, dtype=torch.float16)
        cpu_result = x.permute(0, 3, 1, 2)

        x_device = x.to(DEVICE, dynamic=DYNAMIC)
        compiled_result = compiled_fn(x_device).cpu()

        # Spyre stores fp16 with slightly less precision, so a value can round by
        # one step on the way to the card and back; hence the small tolerance.
        torch.testing.assert_close(compiled_result, cpu_result, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
