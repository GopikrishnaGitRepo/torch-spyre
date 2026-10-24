# Indirect access

Dynamic-shape tests for indirect-access operations (gather and scatter, for
example `torch.embedding` and `torch.index_select`) belong in this directory.

Work on these tests is in progress. The directory exists now because indirect
access is one of the operation categories covered by
`tests/dynamic_shapes`, alongside pointwise, reduction, linear algebra,
comparison and logical, data movement and composition.

Indirect access under a symbolic loop count is tracked separately from the
other categories (`docs/source/compiler/symbolic_shapes.md`, section 14.3).
Today the index tensor of these operations is one-dimensional, so its only
dimension is the innermost one, and `tensor.to("spyre", dynamic=...)` does not
yet support declaring the innermost dimension dynamic.
