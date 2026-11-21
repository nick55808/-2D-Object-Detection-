# RTX 3060 Quickstart

This is the shortest reliable path from a fresh NVIDIA machine to a smoke run.
Run every Python command as `python` after activating `.venv`; do not mix the
virtual-environment interpreter with a system `python3` or user-level `pip`.

## Linux

```bash
cd /path/to/track_a_object_detection_rtx3060
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cuda126.txt
python -m pip install -r requirements.txt
python scripts/verify_cuda.py
```

## Windows PowerShell

```powershell
cd C:\path\to\track_a_object_detection_rtx3060
nvidia-smi
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cuda126.txt
python -m pip install -r requirements.txt
python scripts/verify_cuda.py
```

If the driver supports CUDA 12.8 and the course environment requires it, use
the cu128 index instead of cu126 for the PyTorch install. Do not install both
CUDA wheel variants in one environment.

## Configure and run

```bash
cp config.example.json config.json
# Edit config.json: tutorial_root, dataset_roots.coco, dataset_roots.kitti,
# coco_ann, and latency_image.
python scripts/verify_datasets.py --config config.json
python run_experiment.py --config config.json --phase all --dry-run
python run_experiment.py --config config.json --phase smoke
python run_experiment.py --config config.json --phase final
python make_pdf.py 
  --markdown results/final/final_report.md 
  --output results/final/2D_Object_Detection_Track_A_Report.pdf
```

Expected smoke artifacts include `results/smoke/metrics.json`. The final
submission must use measured values from `results/final/metrics.json` and
`results/final/latency.json`, not the template placeholders.

