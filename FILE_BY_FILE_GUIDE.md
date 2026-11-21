# RTX 3060 code-bundle guide

This bundle is the CUDA/NVIDIA version of the Track A experiment. The original
Apple-M2 bundle is separate; do not mix its `mps` configuration or installation
instructions with this directory.

## Files to use

| File | Purpose |
|---|---|
| `README.md` | Full installation, dataset, experiment, troubleshooting, and submission instructions. |
| `RTX3060_QUICKSTART.md` | Short Linux and Windows setup path for an RTX 3060. |
| `DATASET_SETUP.md` | COCO/KITTI directory layouts, download notes, and path checks. |
| `config.example.json` | Models, datasets, paths, thresholds, image counts, and CUDA settings. |
| `requirements-cuda126.txt` | Pinned PyTorch 2.8.0/torchvision 0.23.0 CUDA 12.6 wheel source. |
| `requirements.txt` | Non-PyTorch Python dependencies. CUDA torch/torchvision are installed separately from the official PyTorch index. |
| `run_experiment.py` | Orchestrates environment capture, evaluation, latency, analysis, failure audit, and report generation. |
| `scripts/verify_cuda.py` | Fails fast if the NVIDIA driver, PyTorch CUDA build, or GPU test is unavailable. |
| `scripts/collect_environment.py` | Records GPU, VRAM, driver output, CUDA runtime, Python, and package versions. |
| `scripts/analyze_results.py` | Computes summaries, COCO-to-KITTI gaps, plots, and speed/accuracy data. |
| `scripts/failure_analysis.py` | Creates candidate images for manual visual failure inspection. |
| `scripts/generate_report.py` | Inserts measured environment and result values into the report template. |
| `scripts/verify_datasets.py` | Checks tutorial, COCO, KITTI, annotation, and latency-image paths before evaluation. |
| `report/technical_report_template.md` | Six-to-ten-page report structure customized for NVIDIA/CUDA measurements. |
| `patches/CUDA_TUTORIAL_NOTE.md` | Documents the CUDA synchronization already present in the tutorial latency helper. |
| `make_pdf.py` | Converts the generated Markdown report into a PDF. |

## Model and device settings

The default configuration is:

```json
{
  "models": [
    "torchvision:fasterrcnn_resnet50_fpn_v2",
    "yolo:yolo11s.pt",
    "hf_detr:PekingU/rtdetr_r50vd"
  ],
  "device": "cuda",
  "batch_size": 1,
  "precision": "fp32"
}
```

This gives three configurations from three detector families: two-stage CNN,
one-stage CNN, and query-based transformer. The experiment intentionally uses
FP32 for the primary comparison. If FP16 is added, it must be reported as a
separate configuration with its own latency and accuracy evidence.

## Installation order

1. Confirm `nvidia-smi` sees the RTX 3060.
2. Create and activate `.venv`.
3. Install the matched CUDA-enabled torch/torchvision wheel, for example:

   ```bash
   python -m pip install torch==2.8.0 torchvision==0.23.0 \
     --index-url https://download.pytorch.org/whl/cu126
   ```

4. Install `requirements.txt`.
5. Run `python scripts/verify_cuda.py`.
6. Configure the tutorial and dataset paths.
7. Run the dry-run, smoke, and final phases in that order.

Do not use `python3` or bare `pip` after activation unless it resolves to the
same `.venv` interpreter. Use `python -m pip` so package installation and
experiment execution use the same environment.

## Report evidence

The final report should include the actual values from:

- `results/environment.json`: RTX 3060 name, VRAM, driver, CUDA runtime, and package versions;
- `results/final/metrics.json`: mAP, AP50, AP75, scale AP, AR, and per-class AP;
- `results/final/latency.json`: synchronized end-to-end mean, P50, P90, and FPS;
- `results/final/domain_gap.csv`: COCO-to-KITTI transfer gap;
- `results/final/failure_candidates.csv` plus at least five manually confirmed visual cases.

