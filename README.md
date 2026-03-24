# LSR-Patch
面向 PatchTST 的频率引导自适应粒度 Patch 预测框架（Frequency-Guided Adaptive Patching for PatchTST）

## Device switch
- CUDA: `--use_gpu True`
- Apple Silicon MPS: `--use_mps True --use_gpu False` (or just `--use_mps 1`)
- CPU: `--use_gpu False --use_mps 0`
- Shell scripts: set `use_mps=1` in `scripts/*.sh` or `scripts/univariate/*.sh`
