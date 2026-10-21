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

"""Eager ops on a tensor.to("spyre", dynamic=...) tensor.

dynamic= marks the reserved dim dynamic for the user's own compiled region.
Eager Spyre ops are standalone ``dynamic=False`` compiles, one binary per
shape, and must not inherit that mark: inheriting it either raised
ConstraintViolationError (a d2d ``copy_`` ties the dim to a concrete size) or
compiled a symbolic kernel sized from the first call and silently wrong for
larger ones. Sizes here deliberately go above the first call's.

Also covers the max= shorthand's interaction with dynamic=.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401
from torch_spyre._C import get_reserved_dims, spyre_empty_reserved

DEVICE = torch.device("spyre")
SPEC = {0: {"min": 64, "max": 512, "granularity": 64}}
# Growing on purpose: a kernel built from the first size is wrong above it.
SIZES = [128, 256, 512]


class TestEagerOpsOnDynamicTensor(TestCase):
    def setUp(self) -> None:
        torch._dynamo.reset()

    def _check(self, op, ref_op, atol) -> None:
        for rows in SIZES:
            x = torch.randn(rows, 64, dtype=torch.float16)
            out = op(x.to(DEVICE, dynamic=SPEC)).cpu().float()
            torch.testing.assert_close(out, ref_op(x.float()), atol=atol, rtol=1e-2)

    def test_add_scalar(self) -> None:
        self._check(lambda d: d + 1, lambda x: x + 1, atol=1e-2)

    def test_exp(self) -> None:
        self._check(torch.exp, torch.exp, atol=0.5)

    def test_sum_over_reserved_dim(self) -> None:
        self._check(lambda d: d.sum(0), lambda x: x.sum(0), atol=0.5)

    def test_d2d_copy_into_reserved_tensor(self) -> None:
        dst = torch.zeros(512, 64, dtype=torch.float16).to(DEVICE, dynamic=SPEC)
        for rows in SIZES:
            src = torch.randn(rows, 64, dtype=torch.float16)
            dst.resize_(rows, 64)
            dst.copy_(src.to(DEVICE))
            torch.testing.assert_close(dst.cpu(), src)

    def test_marks_survive_the_eager_op(self) -> None:
        # The alias handed to the per-op compile is mark-free; the caller's
        # tensor must keep its marks for the user's own torch.compile.
        d = torch.randn(256, 64, dtype=torch.float16).to(DEVICE, dynamic=SPEC)
        _ = d + 1
        self.assertIn(0, d._dynamo_dynamic_indices)

    def test_inplace_op_returns_the_callers_tensor(self) -> None:
        d = torch.randn(256, 64, dtype=torch.float16).to(DEVICE, dynamic=SPEC)
        self.assertIs(d.add_(1), d)


class TestMaxShorthand(TestCase):
    def test_max_is_a_ceiling_only_reservation(self) -> None:
        x = torch.randn(56, 32, dtype=torch.float16)
        d = x.to(DEVICE, max=616)
        self.assertEqual(
            get_reserved_dims(d), {0: {"min": 0, "max": 616, "granularity": 1}}
        )
        self.assertFalse(hasattr(d, "_dynamo_dynamic_indices"))

    def test_five_arg_overload_matches_max(self) -> None:
        x = torch.randn(56, 32, dtype=torch.float16)
        d = spyre_empty_reserved(x.size(), x.stride(), x.dtype, 0, 616)
        self.assertEqual(
            get_reserved_dims(d), {0: {"min": 0, "max": 616, "granularity": 1}}
        )

    def test_max_and_dynamic_together_rejected(self) -> None:
        x = torch.randn(560, 64, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "not both"):
            x.to(DEVICE, dynamic=SPEC, max=512)

    def test_non_int_max_rejected(self) -> None:
        x = torch.randn(56, 32, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "must be an int"):
            x.to(DEVICE, max=616.0)


if __name__ == "__main__":
    run_tests()
