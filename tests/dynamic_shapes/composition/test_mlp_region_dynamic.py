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

# A two-layer MLP, relu(x @ w1 + b1) @ w2 + b2, run as one region on a dynamic
# number of rows: one loop over the rows does the whole MLP, and the hidden
# result never leaves the tile. One compiled binary for every size
# (docs/source/compiler/symbolic_shapes.md, sections 7.3, 9.3, 9.5 and 18.1).

import torch

from torch_spyre._inductor.wsr import for_each_tile

DEVICE = torch.device("spyre")

# Dim 0 varies from 64 to 512 rows, in steps of 64.
DYNAMIC = {0: dict(min=64, max=512, granularity=64)}

# First call at 256 rows, then smaller and larger sizes.
SIZES = [256, 128, 512, 192]


def mlp(x, w1, b1, w2, b2):
    return torch.relu(x @ w1 + b1) @ w2 + b2


def mlp_region_fn(x, w1, b1, w2, b2):
    rows = x.shape[0]
    torch._check(rows >= 64)
    torch._check(rows <= 512)
    torch._check(rows % 64 == 0)
    # The weights and biases do not vary: every tile gets them whole.
    _, out = for_each_tile(
        lambda _, tiles: (None, mlp(*tiles)),
        (x, w1, b1, w2, b2),
        dims=(0, None, None, None, None),
        tile_size=64,
        out_dim=0,
    )
    return out


def test_mlp_region_dynamic():
    compiled_fn = torch.compile(mlp_region_fn, dynamic=None, fullgraph=True)
    w1 = torch.rand(1024, 256, dtype=torch.float16) / 32
    b1 = torch.rand(256, dtype=torch.float16)
    w2 = torch.rand(256, 1024, dtype=torch.float16) / 16
    b2 = torch.rand(1024, dtype=torch.float16)
    weights = [w.to(DEVICE) for w in (w1, b1, w2, b2)]

    for rows in SIZES:
        x = torch.rand(rows, 1024, dtype=torch.float16)
        cpu_result = mlp(x.float(), w1.float(), b1.float(), w2.float(), b2.float())

        x_device = x.to(DEVICE, dynamic=DYNAMIC)
        compiled_result = compiled_fn(x_device, *weights).cpu()

        torch.testing.assert_close(
            compiled_result.float(), cpu_result, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    test_mlp_region_dynamic()
    print("PASSED")
