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

# Sizes that cannot be served are refused, never padded or silently wrong
# (docs/source/compiler/symbolic_shapes.md, section 12).

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.exc import TorchRuntimeError

from torch_spyre._inductor.wsr import for_each_tile

DEVICE = torch.device("spyre")

# Dim 0 varies from 64 to 512 rows, in steps of 64.
DYNAMIC = {0: dict(min=64, max=512, granularity=64)}

# First call at 256 rows, then smaller and larger sizes.
SIZES = [256, 128, 512, 192]

# Above the maximum, below the minimum, and not a multiple of the granularity.
BAD_SIZES = [576, 32, 200]


def gelu_fn(a):
    rows = a.shape[0]
    torch._check(rows >= 64)
    torch._check(rows <= 512)
    torch._check(rows % 64 == 0)
    _, out = for_each_tile(
        lambda _, tiles: (None, F.gelu(tiles[0])),
        (a,),
        dims=(0,),
        tile_size=64,
        out_dim=0,
    )
    return out


def test_size_outside_range_is_refused():
    for rows in [576, 32]:
        x = torch.rand(rows, 1024, dtype=torch.float16)

        with pytest.raises(ValueError, match="is not within"):
            x.to(DEVICE, dynamic=DYNAMIC)


def test_size_off_granularity_is_refused():
    x = torch.rand(200, 1024, dtype=torch.float16)

    with pytest.raises(ValueError, match="not a multiple of granularity=64"):
        x.to(DEVICE, dynamic=DYNAMIC)


def test_bad_size_is_refused_by_the_compiled_function():
    compiled_fn = torch.compile(gelu_fn, dynamic=None, fullgraph=True)
    compiled_fn(torch.rand(256, 1024, dtype=torch.float16).to(DEVICE, dynamic=DYNAMIC))

    for rows in BAD_SIZES:
        x_device = torch.rand(rows, 1024, dtype=torch.float16).to(DEVICE)

        with pytest.raises(TorchRuntimeError):
            compiled_fn(x_device)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='a bad size is refused, but the error is "RuntimeError when making '
    'fake tensor call" and does not name the granularity or the sizes that '
    "would be accepted. Section 12. TODO(dynamic-shapes): no issue filed yet",
)
def test_refusal_names_the_granularity():
    compiled_fn = torch.compile(gelu_fn, dynamic=None, fullgraph=True)
    compiled_fn(torch.rand(256, 1024, dtype=torch.float16).to(DEVICE, dynamic=DYNAMIC))
    x_device = torch.rand(200, 1024, dtype=torch.float16).to(DEVICE)

    with pytest.raises(Exception) as refusal:
        compiled_fn(x_device)

    assert "granularity" in str(refusal.value)


def test_bad_sizes_do_not_break_later_calls():
    compiled_fn = torch.compile(gelu_fn, dynamic=None, fullgraph=True)
    compiled_fn(torch.rand(256, 1024, dtype=torch.float16).to(DEVICE, dynamic=DYNAMIC))

    for rows in BAD_SIZES:
        with pytest.raises(TorchRuntimeError):
            compiled_fn(torch.rand(rows, 1024, dtype=torch.float16).to(DEVICE))

    for rows in SIZES:
        x = torch.rand(rows, 1024, dtype=torch.float16)
        cpu_result = F.gelu(x.float())

        x_device = x.to(DEVICE, dynamic=DYNAMIC)
        compiled_result = compiled_fn(x_device).cpu()

        torch.testing.assert_close(
            compiled_result.float(), cpu_result, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
