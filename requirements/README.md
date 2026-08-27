# Consolidated Python dependencies

The `.in` files consolidate dependencies found across the archived stage
packages. They use bounded version ranges so a resolver can select a mutually
compatible binary stack.

Generate Linux/Python 3.11 lock files with:

```bash
uv pip compile requirements/analysis.in --python-version 3.11 \
  --python-platform x86_64-manylinux_2_28 --torch-backend cpu \
  -o requirements/analysis-lock.txt
uv pip compile requirements/qa.in --python-version 3.11 \
  --python-platform x86_64-manylinux_2_28 \
  -o requirements/qa-lock.txt
uv pip compile requirements/full.in --python-version 3.11 \
  --python-platform x86_64-manylinux_2_28 --torch-backend cpu \
  -o requirements/full-lock.txt
```

The committed `*-lock.txt` files were resolved on 2026-08-27 with uv 0.11.19
for CPython 3.11 on Linux x86_64 / manylinux_2_28. The analysis and full locks
use the PyTorch CPU wheel; the QA lock intentionally omits PyTorch.

For CUDA runs, generate and retain a separate CUDA-specific lock from the
corresponding `.in` file using the official PyTorch wheel index for the target
driver/CUDA stack. Do not edit or mix the committed CPU lock in place: CUDA
wheels are not portable across compute hosts.
