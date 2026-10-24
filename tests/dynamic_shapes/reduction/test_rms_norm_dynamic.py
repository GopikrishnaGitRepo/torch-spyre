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

# rms_norm over the last dimension, on a dynamic number of rows, one compiled
# binary for every size (docs/source/compiler/symbolic_shapes.md, sections
# 7.3, 9.2 and 18.1).

import torch
import torch.nn.functional as F

from torch_spyre._inductor.wsr import for_each_tile

DEVICE = torch.device("spyre")

# Dim 0 varies from 64 to 512 rows, in steps of 64.
DYNAMIC = {0: dict(min=64, max=512, granularity=64)}

# First call at 256 rows, then smaller and larger sizes.
SIZES = [256, 128, 512, 192]


def rms_norm_fn(a, weight):
    rows = a.shape[0]
    torch._check(rows >= 64)
    torch._check(rows <= 512)
    torch._check(rows % 64 == 0)
    # The weight does not vary: every tile gets the whole weight.
    _, out = for_each_tile(
        lambda _, tiles: (None, F.rms_norm(tiles[0], (1024,), tiles[1])),
        (a, weight),
        dims=(0, None),
        tile_size=64,
        out_dim=0,
    )
    return out


def test_rms_norm_dynamic():
    compiled_fn = torch.compile(rms_norm_fn, dynamic=None, fullgraph=True)
    weight = torch.rand(1024, dtype=torch.float16)
    weight_device = weight.to(DEVICE)

    for rows in SIZES:
        x = torch.rand(rows, 1024, dtype=torch.float16)
        cpu_result = F.rms_norm(x.float(), (1024,), weight.float())

        x_device = x.to(DEVICE, dynamic=DYNAMIC)
        compiled_result = compiled_fn(x_device, weight_device).cpu()

        torch.testing.assert_close(
            compiled_result.float(), cpu_result, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    test_rms_norm_dynamic()
    print("PASSED")
