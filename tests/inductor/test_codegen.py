# Copyright 2025 The Torch-Spyre Authors.
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

import math
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import sympy
import torch
from torch.testing import FileCheck
from torch._inductor.exc import InductorError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    run_and_get_code,
)

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.codegen.compute_ops import (
    SymbolKind,
    _per_core_symbolic_dim_info,
)
from torch_spyre._inductor.codegen.superdsc import _resolve_sdsc_size, compile_op_spec
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.work_division import (
    _collect_symbol_metadata,
    _effective_size,
    _valid_divisor_basis,
    adjust_it_space_for_sticks,
    multi_dim_iteration_space_split,
    prioritize_dimensions,
)


class TestSpyreConfig(InductorTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_config_default(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")

        comp_fn = torch.compile(fn)
        out, source_codes = run_and_get_code(comp_fn, x)
        # print("test_config_default")
        # print(source_codes[0])
        FileCheck().check("sdsc_fused_abs").check(
            f"sympify('c0'): (sympify('256'), {config.sencores})"
        ).run(source_codes[0])

    @config.patch({"sencores": 64})
    def test_config_too_many_sencores(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")

        with self.assertRaisesRegex(
            InductorError,
            "Unsupported: Spyre backend does not support: invalid SENCORES value 64",
        ):
            comp_fn = torch.compile(fn)
            comp_fn(x)

    @config.patch({"sencores": 16})
    def test_sencores_16(self):
        fn = torch.abs
        x = torch.randn((256, 128, 512)).to("spyre")
        cfn = torch.compile(fn)
        out, source_codes = run_and_get_code(cfn, x)
        # print("test_sencores 16")
        # print(source_codes[0])
        FileCheck().check("sdsc_fused_abs").check(
            f"sympify('c0'): (sympify('256'), {config.sencores})"
        ).run(source_codes[0])

    @config.patch({"sencores": 32})
    def test_symbolic_batch_dim_pointwise_split(self):
        """Symbolic batch dim must split by ``granularity`` not ``max_size`` (#2287).

        ``[s, 128]`` fp16 with ``s in [64, 1024]`` (granularity = 64). The planner picks the largest
        divisor of granularity ≤ SENCORES = 32, so the batch dim absorbs all
        32 cores and the static stick dim gets split 1.
        """
        fn = torch.add
        x = torch.randn((1024, 128), dtype=torch.float16)
        y = torch.randn_like(x)
        torch._dynamo.mark_dynamic(x, 0, min=64, max=1024)
        torch._dynamo.mark_dynamic(y, 0, min=64, max=1024)
        # dynamic=True not needed: mark_dynamic already makes dim 0 symbolic.
        comp_fn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(comp_fn, x.to("spyre"), y.to("spyre"))
        # Iteration space embeds (size_expr, split). The symbolic batch dim's
        # split must equal SENCORES=32; the static stick dim's split must be 1.
        FileCheck().check("sdsc_fused_add").check(", 32)").check(", 1)").run(
            source_codes[0]
        )

    # ------------------------------------------------------------------
    # Compiled-path symbolic-batch tests for reduction ops (#3062/#3063).
    #
    # These require the built torch_spyre._C extension and a real Spyre
    # target (or whatever backend `torch.compile` falls back to for codegen
    # inspection) -- this sandbox has neither, so the exact FileCheck
    # literals below (kernel name, fused-op naming) are inferred from the
    # pointwise precedent above and from reading wrapper.py's naming
    # convention, not verified by running them. Run these first in isolation
    # (`-k test_symbolic_batch_dim_reduction`) and adjust the literals if the
    # real kernel names differ once compiled on hardware.
    # ------------------------------------------------------------------

    @config.patch({"sencores": 32})
    def test_symbolic_batch_dim_mean_reduction_split(self):
        """Symbolic batch dim through a real reduction op (#3062/#3063).

        Same shape/bounds as test_symbolic_batch_dim_pointwise_split, but
        ``torch.mean(x, dim=-1)`` instead of ``add`` -- the batch dim (output,
        symbolic) should still absorb all 32 cores via its granularity, while
        the reduction dim (concrete, 128) gets split 1.
        """
        def fn(x):
            return torch.mean(x, dim=-1)

        x = torch.randn((1024, 128), dtype=torch.float16)
        torch._dynamo.mark_dynamic(x, 0, min=64, max=1024)
        comp_fn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(comp_fn, x.to("spyre"))
        FileCheck().check("sdsc_fused_mean").check(", 32)").check(", 1)").run(
            source_codes[0]
        )

    @config.patch({"sencores": 32})
    def test_symbolic_batch_dim_rms_norm_split(self):
        """Symbolic batch dim through rms_norm's mean-of-squares reduction.

        rms_norm decomposes to ``torch.mean(input * input, dim=-1, keepdim=True)``
        (see decompositions.py:spyre_rms_norm) -- the batch dim stays symbolic
        and is the only output dim, so it should still absorb all 32 cores.
        """
        def fn(x):
            return torch.nn.functional.rms_norm(x, [128])

        x = torch.randn((1024, 128), dtype=torch.float16)
        torch._dynamo.mark_dynamic(x, 0, min=64, max=1024)
        comp_fn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(comp_fn, x.to("spyre"))
        FileCheck().check(", 32)").run(source_codes[0])

    @config.patch({"sencores": 32})
    def test_symbolic_batch_dim_layer_norm_split(self):
        """Symbolic batch dim through layer_norm's exx2/layernormscale/
        layernormnorm chain (decompositions.py:spyre_layer_norm) -- exx2 is
        the only true Reduction op in the chain; layernormscale/layernormnorm
        are Pointwise (already covered by #2287) but consume exx2's output,
        so this proves the symbolic batch dim survives the whole chain.
        """
        def fn(x, w, b):
            return torch.nn.functional.layer_norm(x, [128], w, b)

        x = torch.randn((1024, 128), dtype=torch.float16)
        w = torch.randn(128, dtype=torch.float16)
        b = torch.randn(128, dtype=torch.float16)
        torch._dynamo.mark_dynamic(x, 0, min=64, max=1024)
        comp_fn = torch.compile(fn, dynamic=False)
        _, source_codes = run_and_get_code(
            comp_fn, x.to("spyre"), w.to("spyre"), b.to("spyre")
        )
        FileCheck().check(", 32)").run(source_codes[0])

    # Need a test where changing dxp_lx_frac_avail changes the generated OpSpec
    # @config.patch({"dxp_lx_frac_avail": 0.01, "lx_planning": True})
    # def test_config_dxp_lx_frac_avail(self):
    #    fn = torch.abs
    #    x = torch.randn((256, 128, 512)).to("spyre")
    #
    #    comp_fn = torch.compile(fn)
    #    out, source_codes = run_and_get_code(comp_fn, x)
    #    #print("test_conf_dxp_lx_frac_avail")
    #    #print(source_codes[0])

    # Need a test where setting lx_planning to True generates a different OpSpec
    # @config.patch({'lx_planning': True})
    # def test_config_lx_planning(self):
    #    fn = torch.abs
    #    x = torch.randn((256, 128, 512)).to("spyre")
    #
    #    comp_fn = torch.compile(fn)
    #    out, source_codes = run_and_get_code(comp_fn, x)
    #    #print(source_codes[0])

    # ------------------------------------------------------------------
    # Unit tests for the symbolic-shape sidecar in work_division.py
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_v(lower=None, upper=None, size_hint=None):
        """
        Mock V whose ShapeEnv reports the given lower / upper bounds.
        """
        shape_env = SimpleNamespace(
            bound_sympy=lambda _e: SimpleNamespace(lower=lower, upper=upper)
        )
        sizevars = SimpleNamespace(shape_env=shape_env)
        if size_hint is not None:
            sizevars.size_hint = lambda _e: size_hint
        return SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))

    def test_collect_symbol_metadata_opt_in(self):
        """
        User-marked dynamic dim (finite max) enters the metadata dict.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch(
                "torch_spyre._inductor.pass_utils.V",
                self._mock_v(lower=sympy.Integer(2), upper=sympy.Integer(512)),
            ):
                result = _collect_symbol_metadata({s0: s0})
        # max comes straight from the ShapeEnv upper bound;
        # granularity is the smallest divisor of 512 with d >= 4 and
        # 512/d <= 32, which is 16.
        self.assertEqual(result, {s0: (512, 16)})

    def test_collect_symbol_metadata_auto_dynamic_skipped(self):
        """
        Dynamo-promoted symbols (no finite max) are skipped, not assigned.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        with patch(
            "torch_spyre._inductor.pass_utils.V",
            self._mock_v(lower=sympy.Integer(2), upper=sympy.oo, size_hint=1024),
        ):
            self.assertEqual(_collect_symbol_metadata({s0: s0}), {})

    def test_dispatch_helpers_symbolic_vs_concrete(self):
        """
        ``_effective_size`` and ``_valid_divisor_basis`` dispatch on ``v in meta``.
        """
        s0 = sympy.Symbol("s0")
        it_space = {s0: sympy.Integer(128)}
        meta = {s0: (512, 16)}
        # In meta: use the (max, granularity) tuple.
        self.assertEqual(_effective_size(s0, it_space, meta), 512)
        self.assertEqual(_valid_divisor_basis(s0, it_space, meta), 16)
        # Not in meta: fall through to concretize_expr(it_space[v]).
        self.assertEqual(_effective_size(s0, it_space, meta={}), 128)
        self.assertEqual(_valid_divisor_basis(s0, it_space, meta={}), 128)

    def test_symbolic_stick_dim_raises_unsupported(self):
        """
        A symbolic dim that lands on a tensor's stick coord is rejected.
        This is a follow up work.
        """
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        # Minimal TensorDep stand-in: the function only reads
        # device_coords[-1], dep.name, and layout.device_layout.elems_per_stick().
        fake_td = SimpleNamespace(
            dep=SimpleNamespace(name="fake_buf"),
            layout=SimpleNamespace(
                device_layout=SimpleNamespace(elems_per_stick=lambda: 64)
            ),
            device_coords=[s0],
        )
        with self.assertRaises(Unsupported) as cm:
            adjust_it_space_for_sticks(
                {s0: sympy.Integer(128)}, [fake_td], {s0: (512, 64)}
            )
        self.assertIn("symbolic stick dim", str(cm.exception))

    def test_inplace_op_run_call_deduplicates_args(self):
        """An inplace op (x *= 2) must not pass the same tensor twice to .run().

        With symbolic args, the MLIR bundle emits one input_arg param per unique
        tensor.  Passing arg0_1 twice would cause a "Number of inputs mismatches"
        error at launch time.  This test verifies the generated .run() call
        contains no duplicate tensor arguments.
        """

        def fn(x):
            x *= 2
            return x

        x = torch.randn((4, 128), dtype=torch.float16, device="spyre")
        cfn = torch.compile(fn)
        _, source_codes = run_and_get_code(cfn, x)
        code = source_codes[0]
        # Find the .run(...) call for the fused kernel
        run_lines = [ln.strip() for ln in code.splitlines() if ".run(" in ln]
        self.assertTrue(run_lines, "No .run(...) call found in generated code")
        for line in run_lines:
            # Extract the argument list between the outermost parentheses
            args_str = line[line.index("(") + 1 : line.rindex(")")]
            args = [a.strip() for a in args_str.split(",")]
            self.assertEqual(
                len(args),
                len(set(args)),
                f"Duplicate args in .run() call: {line}",
            )


class TestSymbolicBatchReductionWorkDivision(InductorTestCase):
    """Unit tests for #3062: symbolic batch dim through the reduction-op work
    division path (``prioritize_dimensions`` / ``multi_dim_iteration_space_split``).

    ``mean``, ``exx2``, and ``prod`` all reach these functions with the same
    op-agnostic iteration-space shape: one output/batch dim plus one reduction
    dim -- neither function ever inspects the op name or ``reduction_type``,
    so a single shape-level test covers all three. ``topkvalue``/``topkindex``
    never reach ``multi_dim_iteration_space_split`` at all: they hit the
    single-core short-circuit in ``enumerate_work_division_candidates``
    (``if op.data.reduction_type in TOPK_OPS: return [{v: 1 for v in it_space}]``),
    which returns the all-ones split unconditionally, independent of whether any
    dim in ``it_space`` is symbolic -- there is nothing to unit test there
    beyond reading that one line. The symbolic-stick-dim guard
    (``adjust_it_space_for_sticks`` raising ``Unsupported``, exercised by
    ``test_symbolic_stick_dim_raises_unsupported`` above) is likewise already
    reduction-agnostic: it inspects only the stick tensor dep, never the op.
    """

    def test_symbolic_batch_dim_is_output_dim_split_before_reduction_dim(self):
        """A symbolic batch dim must land in ``output_dims`` (not
        ``reduction_dims``) purely because it appears in the output's device
        coordinates -- and, when its granularity alone can absorb every core,
        the (concrete) reduction dim gets nothing left to split. This is the
        priority-interaction policy #3062 asks to confirm and document."""
        batch, reduce_dim = sympy.Symbol("c0"), sympy.Symbol("c1")
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        it_space = {batch: s0, reduce_dim: sympy.Integer(8)}
        # (max_size, granularity) for the symbolic batch dim, keyed by the
        # loop variable (matches _collect_symbol_metadata's own keying).
        symbol_meta = {batch: (1024, 32)}
        # Only `batch` appears in the output's (non-stick) device coordinates;
        # `reduce_dim` is collapsed away, exactly like prioritize_dimensions'
        # own coord_vars computation.
        output = SimpleNamespace(device_coords=[batch, sympy.Integer(0)])

        output_dims, reduction_dims = prioritize_dimensions(
            output, it_space, symbol_meta
        )
        self.assertEqual(output_dims, [batch])
        self.assertEqual(reduction_dims, [reduce_dim])

        splits = multi_dim_iteration_space_split(
            it_space,
            max_cores=32,
            output_dims=output_dims,
            reduction_dims=reduction_dims,
            symbol_meta=symbol_meta,
        )
        # granularity=32 fully absorbs the 32-core budget in Pass 2, so Pass 3
        # never gets a chance to touch the (concrete) reduction dim.
        self.assertEqual(splits[batch], 32)
        self.assertEqual(splits[reduce_dim], 1)

    def test_reduction_dim_still_splits_when_cores_remain(self):
        """A symbolic batch dim always goes first, but doesn't starve the
        reduction dim of cores it can actually use -- if the batch dim's
        granularity doesn't consume the whole budget, Pass 3 still splits the
        (concrete) reduction dim with whatever is left."""
        batch, reduce_dim = sympy.Symbol("c0"), sympy.Symbol("c1")
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        it_space = {batch: s0, reduce_dim: sympy.Integer(8)}
        symbol_meta = {batch: (256, 4)}  # granularity=4, doesn't fill 32 cores

        output_dims, reduction_dims = prioritize_dimensions(
            SimpleNamespace(device_coords=[batch, sympy.Integer(0)]),
            it_space,
            symbol_meta,
        )
        splits = multi_dim_iteration_space_split(
            it_space,
            max_cores=32,
            output_dims=output_dims,
            reduction_dims=reduction_dims,
            symbol_meta=symbol_meta,
        )
        self.assertEqual(splits[batch], 4)
        self.assertEqual(splits[reduce_dim], 8)
        self.assertEqual(math.prod(splits.values()), 32)


class TestResolveSdscSize(InductorTestCase):
    """Unit tests for superdsc._resolve_sdsc_size."""

    def test_concrete_sympy_integer(self):
        self.assertEqual(_resolve_sdsc_size(sympy.Integer(256), {}), 256)

    def test_concrete_python_int(self):
        self.assertEqual(_resolve_sdsc_size(128, {}), 128)

    def test_symbolic_in_bounds_returns_max(self):
        # bounds carries (max, granularity); index [0] is max.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        self.assertEqual(_resolve_sdsc_size(s0, {"s0": (1024, 64)}), 1024)

    def test_symbolic_not_in_bounds_falls_back_to_size_hint(self):
        # Symbol absent from bounds → _concretize_for_sdsc → size_hint.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        sizevars = SimpleNamespace(size_hint=lambda _: 128)
        mock_v = SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))
        with patch("torch_spyre._inductor.codegen.superdsc.V", mock_v):
            self.assertEqual(_resolve_sdsc_size(s0, {}), 128)


class TestSymbolKindDimension(InductorTestCase):
    """Unit tests for the dimension variant added to compute_ops.SymbolKind."""

    def test_factory_sets_all_fields(self):
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertEqual(sk.kind, "dimension")
        self.assertEqual(sk.granularity, 64)
        self.assertEqual(sk.max_value, 1024)
        self.assertEqual(sk.pytorch_sym, "s0")

    def test_is_dimension_true(self):
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertTrue(sk.is_dimension)

    def test_address_fields_are_sentinels(self):
        # Address-specific fields must not be set by the dimension factory so
        # they don't collide with kernel/pool symbol-table entries.
        sk = SymbolKind.dimension(granularity=64, max_value=1024, pytorch_sym="s0")
        self.assertEqual(sk.arg_index, -1)
        self.assertEqual(sk.base_sym_idx, -1)
        self.assertEqual(sk.offset, 0)

    def test_kernel_is_not_dimension(self):
        self.assertFalse(SymbolKind.kernel(arg_index=0).is_dimension)

    def test_pool_is_not_dimension(self):
        self.assertFalse(SymbolKind.pool().is_dimension)


class TestPerCoreSymbolicDimInfo(InductorTestCase):
    """Unit tests for compute_ops._per_core_symbolic_dim_info."""

    def test_no_symbolic_dims_returns_empty(self):
        self.assertEqual(_per_core_symbolic_dim_info({}, {}), {})

    def test_single_dim_no_split(self):
        # work_slices == 1 means undivided: maxSize_/granularity_ pass through.
        symbolic_dims = {"c0": ("s0", 64, 1024)}
        work_slices = {sympy.Symbol("c0"): 1}
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {"c0": {"maxSize_": 1024, "granularity_": 64}},
        )

    def test_single_dim_split_across_cores(self):
        symbolic_dims = {"c0": ("s0", 64, 1024)}
        work_slices = {sympy.Symbol("c0"): 4}
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {"c0": {"maxSize_": 256, "granularity_": 16}},
        )

    def test_granularity_floors_at_one(self):
        # granularity // wk_slices would floor to 0; result must clamp to 1
        # so the runtime never sees a zero batch-size granularity.
        symbolic_dims = {"c0": ("s0", 1, 1024)}
        work_slices = {sympy.Symbol("c0"): 4}
        result = _per_core_symbolic_dim_info(symbolic_dims, work_slices)
        self.assertEqual(result["c0"], {"maxSize_": 256, "granularity_": 1})

    def test_multiple_symbolic_dims_independent(self):
        symbolic_dims = {
            "c0": ("s0", 64, 1024),
            "c1": ("s1", 32, 512),
        }
        work_slices = {
            sympy.Symbol("c0"): 4,
            sympy.Symbol("c1"): 2,
        }
        self.assertEqual(
            _per_core_symbolic_dim_info(symbolic_dims, work_slices),
            {
                "c0": {"maxSize_": 256, "granularity_": 16},
                "c1": {"maxSize_": 256, "granularity_": 16},
            },
        )


class TestSdscJsonSymbolicDimSmoke(InductorTestCase):
    """Smoke test: a symbolic iteration-space dim survives end-to-end through
    compile_op_spec (parse_op_spec + generate_sdsc) into the emitted SDSC
    JSON's dimToSymbolMapping_ / symbolicDimInfo_ fields.

    Fixture uses a [512, 256] fp16 stick-layout tensor with the row dim
    made symbolic. Because _resolve_sdsc_size resolves a symbolic dim to
    its declared max (512), every downstream computation (padding,
    stick-dim detection, core slicing) runs identically to the equivalent
    concrete case -- only the symbolic_dims side-channel asserted on here
    differs.
    """

    _DEVICE_SIZE = [4, 512, 64]
    _HBM_BASE = 0x400000000

    def _make_symbolic_op_spec(self) -> OpSpec:
        c_row, c_col = sympy.Symbol("c_row"), sympy.Symbol("c_col")
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        coords = [c_col // 64, c_row, sympy.Mod(c_col, 64)]

        def _tensor_arg(is_input, arg_index, hbm_base):
            return TensorArg(
                is_input=is_input,
                arg_index=arg_index,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=list(self._DEVICE_SIZE),
                device_coordinates=coords,
                allocation={"hbm": hbm_base},
            )

        return OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={
                c_row: (s0, 1),
                c_col: (sympy.Integer(256), 1),
            },
            args=[
                _tensor_arg(True, 0, self._HBM_BASE),
                _tensor_arg(True, 1, self._HBM_BASE + 0x1000),
                _tensor_arg(False, 2, self._HBM_BASE + 0x100000000),
            ],
            op_info={},
            symbolic_dim_bounds={"s0": (512, 64)},  # (max, granularity)
        )

    def test_symbolic_dim_fields_in_sdsc_json(self):
        op_spec = self._make_symbolic_op_spec()
        sdsc_json, _, _, _ = compile_op_spec(idx=0, op_spec=op_spec, symbols=[])

        top = next(iter(sdsc_json.values()))
        dsc = next(iter(top["dscs_"][0].values()))

        # "s0" is registered as dim-symbol id -1 and bound to the SDSC "mb"
        # dim (c_row maps to the first non-output dim label for a 2-dim op).
        self.assertEqual(dsc["dimToSymbolMapping_"], {"mb": [-1]})

        for stage in ("ss_", "el_"):
            sym_info = dsc["dataStageParam_"]["0"][stage]["symbolicDimInfo_"]
            self.assertEqual(sym_info, {"mb": {"maxSize_": 512, "granularity_": 64}})


class TestSdscReductionSymbolicBatch(InductorTestCase):
    """Unit tests for #3063: a reduction op's SDSC JSON must correctly mark the
    reduced dim AND carry the symbolic batch dim's side-channel info on the
    same op, and the bundle-global symbol pool must number a reduction op's
    symbolic-dim symbol relative to whatever a preceding op in the bundle
    already registered.

    Fixture models ``mean(x, dim=-1, keepdim=True)`` (the same reduction
    ``rms_norm``/``layer_norm`` build on, see decompositions.py:spyre_rms_norm):
    input ``[s0, 256]`` fp16 (``s0`` symbolic, max=512, granularity=64,
    matching TestSdscJsonSymbolicDimSmoke's fixture) reduced over its stick
    (last) dimension; output ``[s0, 1]`` -- the reduced dim collapses to a
    placeholder coordinate, exactly as ``_create_sdsc_tensors`` already
    handles for any concrete reduction (mirrors the shape convention in
    test_coarse_tiling.py's TestSharedWeightUnitBmmLayout fixtures, which use
    ``Integer(0)`` placeholder coordinates for a collapsed dim).
    """

    _HBM_BASE = 0x400000000

    def _make_reduction_op_spec(self, *, symbolic: bool) -> OpSpec:
        c_row, c_col = sympy.Symbol("c_row"), sympy.Symbol("c_col")
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        batch_size = s0 if symbolic else sympy.Integer(512)

        input_arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[4, 512, 64],
            device_coordinates=[c_col // 64, c_row, sympy.Mod(c_col, 64)],
            allocation={"hbm": self._HBM_BASE},
        )
        output_arg = TensorArg(
            is_input=False,
            arg_index=1,
            device_dtype=DataFormats.SEN169_FP16,
            # The reduced (stick) dim collapses to a single placeholder slot:
            # device_size keeps the same 3-slot shape as the input (tile-count,
            # batch, stick-elem) with the tile-count position pinned to 1.
            device_size=[1, 512, 64],
            device_coordinates=[sympy.Integer(0), c_row, sympy.Integer(0)],
            allocation={"hbm": self._HBM_BASE + 0x100000000},
        )
        return OpSpec(
            op="mean",
            is_reduction=True,
            iteration_space={
                c_row: (batch_size, 1),
                c_col: (sympy.Integer(256), 1),
            },
            args=[input_arg, output_arg],
            op_info={},
            symbolic_dim_bounds={"s0": (512, 64)} if symbolic else {},
        )

    def test_reduced_stick_dim_and_symbolic_batch_coexist(self):
        """Concern #1: the scale vector must mark the reduced dim correctly
        while a different dim on the same tensor is symbolic."""
        op_spec = self._make_reduction_op_spec(symbolic=True)
        sdsc_json, _, _, _ = compile_op_spec(idx=0, op_spec=op_spec, symbols=[])

        top = next(iter(sdsc_json.values()))
        dsc = next(iter(top["dscs_"][0].values()))

        # The symbolic batch dim ("mb") registers exactly as it does for a
        # pointwise op (#2673) -- reduction doesn't change dim-symbol
        # registration.
        self.assertEqual(dsc["dimToSymbolMapping_"], {"mb": [-1]})
        for stage in ("ss_", "el_"):
            sym_info = dsc["dataStageParam_"]["0"][stage]["symbolicDimInfo_"]
            self.assertEqual(sym_info, {"mb": {"maxSize_": 512, "granularity_": 64}})

        # The reduced dim ("out" -- also the stick dim here, since this models
        # a reduction over the last/stick axis like rms_norm/layer_norm) is
        # marked -2 (reduced AND stick) on the output tensor, while the
        # symbolic batch dim keeps its ordinary scale of 1 on that SAME
        # tensor -- the two markers coexist without interfering.
        output_ds = dsc["labeledDs_"][-1]
        dim_order = dsc["primaryDsInfo_"][output_ds["dsType_"]]["layoutDimOrder_"]
        scale_by_dim = dict(zip(dim_order, output_ds["scale_"]))
        self.assertEqual(scale_by_dim["mb"], 1)
        self.assertEqual(scale_by_dim["out"], -2)

        # opfunc stays "mean" (not "meannonstick"): _get_op_func only appends
        # "nonstick" when no output dim is marked -2, i.e. when the reduction
        # does NOT touch the stick dim -- confirming the fixture models a
        # stick-dim reduction as intended.
        self.assertEqual(next(iter(top["dscs_"][0])), "mean")

    def test_bundle_symbol_pool_shared_across_ops(self):
        """Concern #3: bundle.mlir's symbol IDs are drawn from the same
        bundle-global pool -- a reduction op's ``dimension``-kind symbol must
        be numbered relative to whatever a preceding op in the bundle already
        registered (mirrors how bundle.py._compile_specs threads `symbols`/
        `symbol_id_offset` across every OpSpec in a bundle), not restart at -1
        as it would in isolation (TestSdscJsonSymbolicDimSmoke only exercises
        the symbol_id_offset=0 case).
        """
        concrete_op_spec = self._make_reduction_op_spec(symbolic=False)
        symbolic_op_spec = self._make_reduction_op_spec(symbolic=True)

        symbols: list[int] = []
        _, sym_values_a, _, _ = compile_op_spec(
            idx=0,
            op_spec=concrete_op_spec,
            symbols=symbols,
            symbol_id_offset=0,
            use_symbols=True,
        )
        offset = len(sym_values_a)
        sdsc_json_b, sym_values_b, _, _ = compile_op_spec(
            idx=1,
            op_spec=symbolic_op_spec,
            symbols=symbols,
            symbol_id_offset=offset,
            use_symbols=True,
        )

        top_b = next(iter(sdsc_json_b.values()))
        dsc_b = next(iter(top_b["dscs_"][0].values()))
        self.assertEqual(dsc_b["dimToSymbolMapping_"], {"mb": [-(offset + 1)]})

        # No symbol ID collisions between the two ops' entries in the shared
        # pool: op A's ids occupy -1..-len(sym_values_a); op B's ids continue
        # from -(offset+1) onward.
        ids_a = {-(i + 1) for i in range(len(sym_values_a))}
        ids_b = {-(offset + i + 1) for i in range(len(sym_values_b))}
        self.assertEqual(ids_a & ids_b, set())
        self.assertEqual(len(symbols), len(sym_values_a) + len(sym_values_b))
