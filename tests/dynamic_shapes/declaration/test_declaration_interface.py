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

# The user-facing interface of docs/source/compiler/symbolic_shapes.md, section
# 7.3: declaring a dynamic dimension with tensor.to("spyre", dynamic=...) and
# compiling with torch.compile(model, dynamic=None).

import pytest
import torch
import torch.nn.functional as F

from torch_spyre._C import get_reserved_dims

DEVICE = torch.device("spyre")

# Dim 0 varies from 64 to 512 rows, in steps of 64.
DYNAMIC = {0: dict(min=64, max=512, granularity=64)}

# First call at 256 rows, then smaller and larger sizes.
SIZES = [256, 128, 512, 192]


def test_declaration_is_read_back():
    x = torch.rand(128, 1024, dtype=torch.float16)

    x_device = x.to(DEVICE, dynamic=DYNAMIC)

    assert get_reserved_dims(x_device) == {
        0: {"min": 64, "max": 512, "granularity": 64}
    }


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="an omitted granularity is rejected: it defaults to 1, which gives "
    "512 steps between 64 and 512, more than the cap of 32. Section 7.3 says it "
    "defaults to min. TODO(dynamic-shapes): no issue filed yet",
)
def test_granularity_defaults_to_min():
    x = torch.rand(128, 1024, dtype=torch.float16)

    x_device = x.to(DEVICE, dynamic={0: dict(min=64, max=512)})

    assert get_reserved_dims(x_device)[0]["granularity"] == 64


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the interface alone, with no range stated in the model and no tiled "
    "region, compiles once and returns wrong values above the first call's size. "
    "Section 7.3: the declared range does not reach the compiler yet. "
    "TODO(dynamic-shapes): no issue filed yet",
)
def test_interface_alone():
    compiled_fn = torch.compile(F.gelu, dynamic=None)

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
