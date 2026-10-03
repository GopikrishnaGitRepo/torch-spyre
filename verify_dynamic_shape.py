# Simple verification script for Phase 1 symbolic/dynamic shape support.
# Compile once with a dynamic batch dimension, run at three different sizes,
# confirm all results match CPU gelu within fp16 tolerance.

import torch
import torch.nn.functional as F

DEVICE = torch.device("spyre")
GRANULARITY = 64   # tile size / minimum batch
MAX_BATCH = 576    # upper bound registered with mark_dynamic

torch.manual_seed(42)


def gelu_fn(x):
    return F.gelu(x)


# Allocate at max size and mark dim 0 as dynamic
x_max = torch.rand(MAX_BATCH, 1024, dtype=torch.float16)
x_device = x_max.to(DEVICE)
torch._dynamo.mark_dynamic(x_device, 0, min=GRANULARITY, max=MAX_BATCH)

compiled_fn = torch.compile(gelu_fn)

print("Compiling (first call triggers compile)...")
for batch in [64, 128, 512]:
    x_slice = x_device[:batch]
    out = compiled_fn(x_slice).cpu()
    ref = gelu_fn(x_max[:batch])
    delta = (out - ref).abs().max().item()
    status = "PASS" if delta < 1e-2 else "FAIL"
    print(f"  batch={batch:4d}  max_delta={delta:.6f}  [{status}]")
