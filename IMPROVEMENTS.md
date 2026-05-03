# Improvements in this fork

This branch (`fix/marlin-w4a16-pad-sub-tile`) carries a single cherry-picked
bug fix on top of [`vllm-project/vllm`](https://github.com/vllm-project/vllm).

---

## 1. Marlin W4A16: pad sub-tile output dims on load

**Cherry-picked from**: [PR #41440 by @wasifbasharat](https://github.com/vllm-project/vllm/pull/41440)
(upstream still under review at the time of fork).

**Files**: Marlin kernel layer files.

Fixes a Marlin W4A16 quantisation path that produced incorrect output for
weights whose output dimension wasn't a multiple of the kernel's sub-tile
size. The fix pads the sub-tile output dims at weight-load time so the
kernel sees a properly aligned tensor.

We ship this branch as a temporary unblock for serving W4A16-quantised
models (e.g. specific HF checkpoints) until the upstream PR lands.

---

## How to install this fork

```bash
git clone https://github.com/svilendotorg/vllm
cd vllm
pip install -e .
```

Default branch is `fix/marlin-w4a16-pad-sub-tile`, so cloning lands on the
patched code automatically.

To sync with upstream `main`, click **Sync fork** on the GitHub page or:

```bash
git remote add upstream https://github.com/vllm-project/vllm.git
git fetch upstream
git rebase upstream/main   # or merge if you prefer
git push --force-with-lease
```

If the upstream PR merges, drop this branch — the patch will already be
in upstream `main`.
