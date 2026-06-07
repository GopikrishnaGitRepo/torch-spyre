import torch
import torch.fx

def insert_dtype_conversions(graph: torch.fx.Graph):
    for node in list(graph.nodes):
        # Target binary ops
        if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
            
            args = node.args
            if len(args) != 2:
                continue

            # Ensure both are nodes
            if not all(isinstance(arg, torch.fx.Node) for arg in args):
                continue

            # Get dtype metadata
            val0 = args[0].meta.get("val", None)
            val1 = args[1].meta.get("val", None)

            if val0 is None or val1 is None:
                continue

            dtype0 = getattr(val0, "dtype", None)
            dtype1 = getattr(val1, "dtype", None)

            if dtype0 is None or dtype1 is None:
                continue

            if dtype0 == dtype1:
                continue

            # Decide higher precision
            dtype_priority = {
                torch.float32: 3,
                torch.bfloat16: 2,
                torch.float16: 1,
            }

            p0 = dtype_priority.get(dtype0, 0)
            p1 = dtype_priority.get(dtype1, 0)

            if p0 == p1:
                continue

            with graph.inserting_before(node):
                if p0 > p1:
                    # convert arg1 → dtype0
                    converted = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        (args[1], dtype0),
                    )
                    converted.meta["val"] = val1.to(dtype0)
                    node.update_arg(1, converted)
                else:
                    # convert arg0 → dtype1
                    converted = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        (args[0], dtype1),
                    )
                    converted.meta["val"] = val0.to(dtype1)
                    node.update_arg(0, converted)
