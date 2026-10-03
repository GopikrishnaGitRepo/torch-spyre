# Copyright 2025-2026 The Torch-Spyre Authors.
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

import torch
from torch_spyre._C import (
    SymbolicArg,
    SymbolicArgKind,
    launch_jobplan,
    prepare_kernel,
    register_kernel_provenance,
)
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.kernel_provenance import KernelProvenanceDescriptor
from torch_spyre._inductor.profiler_event import (
    format_kernel_provenance_event_name,
)
from torch_spyre.profiler._ffdc import (
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    with_ffdc,
)

logger = get_inductor_logger("kernel_runner")


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    @with_ffdc(CATEGORY_UNIMPLEMENTED, logger, code_dir_attr=None)
    def run(self, *args, **kwargs):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains"
            f" unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    """Kernel runner for a compiled SDSC bundle.

    The jobplan handle is initialised lazily on first call to :meth:`run`.
    This avoids calling ``prepare_kernel`` (which requires a live C++
    RuntimeContext) in the compiling process; the context is only guaranteed
    to be available on the process that actually launches the kernel.
    """

    def __init__(
        self,
        name: str,
        code_dir: str,
        kernel_provenance: KernelProvenanceDescriptor | None = None,
        symbol_kinds: list[SymbolKind] | None = None,
    ):
        self.kernel_name = name
        self.code_dir = code_dir
        self.kernel_provenance = kernel_provenance
        # Canonical symbol order returned by generate_bundle()
        self.symbol_kinds: list[SymbolKind] = (
            symbol_kinds if symbol_kinds is not None else []
        )
        self.profiler_event_name: str | None
        self._jobplan = None  # initialised lazily, not pickled

        if kernel_provenance is None:
            self.profiler_event_name = None
        else:
            self.profiler_event_name = format_kernel_provenance_event_name(
                kernel_provenance
            )
            # Rejection is intentionally fail-open: C++ warns and counts
            # conflicts while the key-bearing name remains the compatibility
            # join.
            register_kernel_provenance(
                self.profiler_event_name,
                list(kernel_provenance.debug_handle_ids),
            )

        # Build the SymbolicArg payload from the canonical symbol order
        # that generate_bundle() returned and stored on this runner.
        # symbol_kinds matches the MLIR input_arg slot order: pool first
        # (when frontend_pool_allocation is active), then kernel tensor
        # args in arg_index order, then dimension args.
        # Dimension symbols get a None sentinel; they are resolved to concrete
        # SymbolicArg(kDimension, ...) at each run() call.
        if self.symbol_kinds:
            pool_offset = 1 if self.symbol_kinds[0].is_pool else 0
            args_template: list = []
            for sk in self.symbol_kinds:
                if sk.is_pool:
                    args_template.append(
                        SymbolicArg(kind=SymbolicArgKind.kAddress, tensor_id=0)
                    )
                elif sk.is_dimension:
                    args_template.append(None)  # resolved at run() time
                else:
                    args_template.append(
                        SymbolicArg(
                            kind=SymbolicArgKind.kAddress,
                            tensor_id=sk.arg_index + pool_offset,
                        )
                    )
            self._symbolic_args: list | None = args_template
            self._has_dim_args: bool = any(a is None for a in args_template)
        else:
            self._symbolic_args = None
            self._has_dim_args = False

    @property
    def jobplan(self):
        if self._jobplan is None:
            logger.debug(
                "Initialising jobplan for %s from %s", self.kernel_name, self.code_dir
            )
            # _lazy_init() ensures the C++ RuntimeContext is initialised before
            # prepare_kernel(), which calls into JobPlanBuilder/getDefaultStream().
            torch.spyre._impl._lazy_init()
            spyrecode_dir = self.code_dir + "/spyreCodeDir"
            if self.profiler_event_name is None:
                self._jobplan = prepare_kernel(spyrecode_dir)
            else:
                with torch.profiler.record_function(
                    f"prepare_kernel:{self.kernel_name}"
                ):
                    self._jobplan = prepare_kernel(
                        spyrecode_dir,
                        profiler_name=self.profiler_event_name,
                    )
        return self._jobplan

    @with_ffdc(CATEGORY_RUNTIME_LAUNCH, logger)
    def run(self, *args):
        logger.info("RUN: %s %s", self.kernel_name, self.code_dir)
        with torch.profiler.record_function(f"launch_jobplan:{self.kernel_name}"):
            if self._symbolic_args is not None:
                if self._has_dim_args:
                    # Resolve dimension symbols from actual tensor shapes.
                    pool_offset = (
                        1
                        if self.symbol_kinds and self.symbol_kinds[0].is_pool
                        else 0
                    )
                    resolved = []
                    for sk, sa in zip(self.symbol_kinds, self._symbolic_args):
                        if sa is None:  # is_dimension sentinel
                            tid = sk.arg_index + pool_offset
                            actual_size = int(args[tid].shape[sk.dim_index])
                            resolved.append(
                                SymbolicArg(
                                    kind=SymbolicArgKind.kDimension,
                                    tensor_id=tid,
                                    dim_index=sk.dim_index,
                                    value=actual_size,
                                )
                            )
                        else:
                            resolved.append(sa)
                    launch_jobplan(self.jobplan, args, resolved)
                else:
                    launch_jobplan(self.jobplan, args, self._symbolic_args)
            else:
                launch_jobplan(self.jobplan, args)
