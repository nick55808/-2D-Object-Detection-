# CUDA timing note

The supplied `DeepDataMiningLearning.ngdet.latency` implementation already
uses the required CUDA synchronization protocol:

```python
if device.startswith("cuda") and torch.cuda.is_available():
    torch.cuda.synchronize()
```

It synchronizes after warm-up and after each timed prediction. This prevents
asynchronous CUDA work from making the reported wall-clock latency look too
small. Do not remove those calls when updating the tutorial repository.

The RTX bundle runs the benchmark with `--accel fp32`, batch size 1, ten warm-up
iterations, and fifty measured iterations. Keep that protocol fixed across all
three models and record the actual GPU, driver, CUDA runtime, and package
versions in `results/environment.json`.

