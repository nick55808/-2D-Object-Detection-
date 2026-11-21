# Track A: 2D Object Detection Evaluation on NVIDIA RTX 3060

This bundle evaluates three pretrained 2D object detectors without fine-tuning:

- Faster R-CNN ResNet-50-FPN v2
- YOLO11s
- RT-DETR R50

The experiment compares COCO val2017 with KITTI using the shared
vehicle / person / cyclist taxonomy. It produces detection metrics,
cross-dataset analysis, end-to-end CUDA latency, annotated videos, failure
candidates, and a report draft.

This version is configured for an NVIDIA GeForce RTX 3060 using PyTorch CUDA.
The default reproducible build is PyTorch 2.8.0, torchvision 0.23.0, and the
CUDA 12.6 wheel. The official PyTorch version table lists matched
torch/torchvision CUDA 12.6 and CUDA 12.8 commands for this release. See:
https://pytorch.org/get-started/previous-versions/

## 1. Bundle layout

~~~text
track_a_object_detection_rtx3060/
├── RTX3060_QUICKSTART.md
├── DATASET_SETUP.md
├── config.example.json
├── requirements.txt
├── requirements-cuda126.txt
├── run_experiment.py
├── scripts/
│   ├── analyze_results.py
│   ├── collect_environment.py
│   ├── failure_analysis.py
│   ├── generate_report.py
│   ├── verify_datasets.py
│   └── verify_cuda.py
├── report/
│   └── technical_report_template.md
├── patches/
│   └── CUDA_TUTORIAL_NOTE.md
└── make_pdf.py
~~~

The DeepDataMiningLearning repository is the tutorial dependency. Clone it
beside this folder, or pass its location with --tutorial-root.

## 2. Check the NVIDIA driver

Run:

~~~bash
nvidia-smi
~~~

Confirm that the output lists an NVIDIA GeForce RTX 3060. Record the displayed
Driver Version and CUDA Version. The CUDA Version shown by nvidia-smi is the
maximum CUDA runtime level supported by the installed driver.

If nvidia-smi is not found, install the official NVIDIA driver for your
operating system before installing PyTorch. Do not continue until the GPU is
visible.

## 3. Create the Python environment

### Linux

~~~bash
cd /path/to/track_a_object_detection_rtx3060
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
~~~

### Windows PowerShell

~~~powershell
cd C:\path\to\track_a_object_detection_rtx3060
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
~~~

Use Python 3.11 or newer when possible. The current PyTorch installer recommends
Python 3.11 or later for the latest stable release.

## 4. Install the CUDA-enabled PyTorch build

For a driver that supports CUDA 12.6 or newer, run:

~~~bash
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
~~~

The same installation is captured in `requirements-cuda126.txt`, so it can
also be installed with `python -m pip install -r requirements-cuda126.txt`.

If your course or machine specifically uses CUDA 12.8 and nvidia-smi reports
support for it, use:

~~~bash
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
~~~

Do not run both commands. Do not install the CPU-only PyTorch build afterward.
For this inference experiment, the pip wheel supplies the CUDA runtime; the
NVIDIA driver is the required system-level prerequisite.

Install the remaining dependencies:

~~~bash
python -m pip install -r requirements.txt
~~~

## 5. Verify CUDA before downloading model checkpoints

~~~bash
python scripts/verify_cuda.py
~~~

The verification should report:

~~~text
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
CUDA runtime: ...
GPU test passed
~~~

If it reports False, check that the virtual environment is active and that
nvidia-smi works before reinstalling PyTorch.

## 6. Prepare the tutorial repository

Clone the tutorial beside the bundle:

~~~bash
cd /path/to
git clone https://github.com/lkk688/DeepDataMiningLearning.git
~~~

The expected path is:

~~~text
/path/to/DeepDataMiningLearning/DeepDataMiningLearning
~~~

## 7. Configure datasets and experiment

~~~bash
cd /path/to/track_a_object_detection_rtx3060
cp config.example.json config.json
~~~

Edit these paths in config.json:

- tutorial_root
- dataset_roots.coco
- dataset_roots.kitti
- coco_ann
- latency_image

Before the smoke run, validate both the paths and the annotation contents:

~~~bash
python scripts/verify_datasets.py --config config.json
~~~

The CUDA-specific settings are already configured:

~~~json
{
  "device": "cuda",
  "batch_size": 1,
  "precision": "fp32"
}
~~~

The smoke run uses 50 images per dataset. The final run uses 500 images per
dataset. Use the same image subset for all models within each dataset.

## 8. Run the experiment

Print commands without running models:

~~~bash
python run_experiment.py --config config.json --phase all --dry-run
~~~

Run the smoke test:

~~~bash
python run_experiment.py --config config.json --phase smoke
~~~

Check that this file exists and contains model/dataset entries:

~~~text
results/smoke/metrics.json
~~~

Run the final evaluation:

~~~bash
python run_experiment.py --config config.json --phase final
~~~

Generate the PDF:

~~~bash
python make_pdf.py \
  --markdown results/final/final_report.md \
  --output results/final/2D_Object_Detection_Track_A_Report.pdf
~~~

## 9. CUDA troubleshooting

Always use python inside the active virtual environment. Check:

~~~bash
which python
python -c "import sys; print(sys.executable)"
python -m pip show torch torchvision ultralytics
~~~

The paths should point into .venv. Avoid using a separate system python3 or
user-level pip installation.

If CUDA is unavailable:

~~~bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
~~~

If the GPU runs out of memory, keep batch size 1, disable videos during the
smoke test, reduce max_images_smoke, and test models individually. The RTX
3060 should normally handle these inference-only configurations, but available
VRAM varies by model and system.

## 10. Report requirements

State the exact:

- GPU model and VRAM
- NVIDIA driver and reported CUDA version
- Python, PyTorch, and torchvision versions
- checkpoint names
- dataset paths/splits and image counts
- confidence threshold
- input resolution
- batch size and precision
- CUDA device string
- end-to-end latency protocol with CUDA synchronization

Use this setup statement as a starting point:

> The experiment was performed on an NVIDIA GeForce RTX 3060 using a
> CUDA-enabled PyTorch build. All models were evaluated in FP32 with batch size
> 1. GPU timing included warm-up iterations and CUDA synchronization. GPU name,
> VRAM, driver, CUDA runtime, and package versions were recorded automatically.

## 11. Required final evidence

The final results directory should contain:

- environment.json
- commands.log
- metrics.json
- latency.json
- summary.csv
- domain_gap.csv
- heatmap_mAP.png
- domain_gap.png
- speed_accuracy.png
- annotated videos or a documented subset
- failure_candidates.csv
- final_report.md

Do not submit until the report contains measured values from the RTX 3060 run.
