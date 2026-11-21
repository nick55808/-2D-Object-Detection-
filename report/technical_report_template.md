---
title: "{{TITLE}}"
author: "{{STUDENT}}"
date: "November 2026"
geometry: margin=0.72in
fontsize: 10.5pt
colorlinks: true
linkcolor: blue
urlcolor: blue
---

# Abstract

This report evaluates three modern 2D object detectors without fine-tuning: Faster R-CNN ResNet-50-FPN v2, YOLO11s, and RT-DETR R50. The experiment compares a two-stage region-proposal detector, a dense one-stage detector, and a transformer-based set-prediction detector. All models are evaluated using the same unified driving taxonomy on COCO val2017 and KITTI 2D detection data.

The central question is how detector architecture and domain shift affect accuracy, object-scale performance, and inference speed. The evaluation reports COCO-style mAP, AP50, AP75, AP by object size, average recall, per-class AP, and end-to-end latency. Because the experiment is zero-shot, the difference between COCO and KITTI performance is interpreted as a combination of visual domain shift, label-taxonomy mismatch, and model design.

**{{STATUS}}**

# 1. Models and datasets

## 1.1 Models

The selected models satisfy the assignment requirement of at least three configurations from at least two detector families.

| Model | Detector family | Main architectural idea | Why it was selected |
|---|---|---|---|
| Faster R-CNN ResNet-50-FPN v2 | Two-stage CNN | Region Proposal Network followed by RoI classification/regression | Provides an established accuracy-oriented baseline |
| YOLO11s | One-stage dense CNN | Multi-scale dense box and class prediction | Provides a practical real-time baseline on the RTX 3060 |
| RT-DETR R50 | Query-based transformer | Hybrid multi-scale encoder and object-query decoder | Tests an end-to-end detector that avoids traditional NMS |

Faster R-CNN uses an RPN to propose candidate regions and a second-stage RoI head to classify and refine them. YOLO produces many dense candidate predictions and generally relies on confidence filtering and NMS. DETR-style models instead formulate detection as set prediction with one-to-one matching; RT-DETR adapts that formulation for real-time use.

## 1.2 Datasets and taxonomy

The in-domain reference is COCO val2017. KITTI is used as the cross-domain autonomous-driving dataset. The experiment does not fine-tune on either dataset.

The evaluator maps native dataset/model labels into a common three-class taxonomy:

| Unified class | COCO labels | KITTI labels | Caveat |
|---|---|---|---|
| vehicle | car, truck, bus, van, motorcycle | Car, Van, Truck, Tram | Coarse grouping hides vehicle-type differences |
| person | person | Pedestrian | Mostly comparable, although annotation policy can differ |
| cyclist | bicycle | Cyclist | Not equivalent: COCO separates rider and bicycle while KITTI often annotates one combined box |

The cyclist category is therefore reported, but its results are interpreted cautiously rather than treated as a perfectly comparable class.

# 2. Experimental setup and hardware

## 2.1 Hardware and software

The intended experiment uses an NVIDIA GeForce RTX 3060 through PyTorch's CUDA backend. The exact GPU name, VRAM, NVIDIA driver, CUDA runtime, Python version, and package versions are recorded automatically by `collect_environment.py` and reproduced below.

{{ENVIRONMENT_TABLE}}

## 2.2 Controlled variables

- Zero-shot inference; no training or fine-tuning.
- Same mapped taxonomy for all models.
- Same dataset image subset for every model within each dataset.
- Same confidence threshold: 0.30.
- Same evaluation metrics and maximum detections per image.
- Batch size: 1 for latency measurement.
- Input resolution: 640 pixels where supported by the model adapter.
- Latency includes preprocessing, model inference, decoding, and post-processing.
- CUDA timing uses warm-up iterations followed by `torch.cuda.synchronize()` before each timed result.
- Primary inference uses FP32 to make the three model measurements directly comparable; the RTX 3060 CUDA path can be extended with a separately reported FP16 study.

The smoke test uses 50 images per dataset. The final run should use at least 500 images per dataset or the full available evaluation split.

# 3. Commands and configurations

The implementation is based on the `DeepDataMiningLearning.ngdet` tutorial harness. The wrapper script records every command in `commands.log`.

## 3.1 Smoke test

```bash
python run_experiment.py --config config.json --phase smoke
```

The smoke test checks dataset paths, checkpoint loading, taxonomy mapping, COCO evaluation, video writing, and result serialization.

## 3.2 Final accuracy evaluation

The underlying command is equivalent to:

```bash
python -m DeepDataMiningLearning.ngdet.run_eval \
  --models torchvision:fasterrcnn_resnet50_fpn_v2 \
           yolo:yolo11s.pt \
           hf_detr:PekingU/rtdetr_r50vd \
  --datasets coco kitti \
  --taxonomy driving3 \
  --max-images 500 \
  --score-thr 0.3 \
  --device cuda \
  --coco-ann /path/to/instances_val2017.json \
  --roots coco=/path/to/coco2017/val2017 kitti=/path/to/Kitti \
  --video --video-max 40
```

## 3.3 Latency evaluation

```bash
python -m DeepDataMiningLearning.ngdet.latency \
  --models torchvision:fasterrcnn_resnet50_fpn_v2 \
           yolo:yolo11s.pt \
           hf_detr:PekingU/rtdetr_r50vd \
  --accel fp32 \
  --image /path/to/Kitti/training/image_2/000008.png \
  --warmup 10 --iters 50 \
  --device cuda \
  --taxonomy driving3
```

# 4. Quantitative results

## 4.1 Detection accuracy

The main metric is mAP@[0.50:0.95]. AP50 measures performance at a looser overlap requirement, while AP75 emphasizes stricter localization. AP small, medium, and large show whether the detector is sensitive to object scale.

{{RESULTS_TABLE}}

## 4.2 Speed and latency

The latency results are end-to-end measurements rather than forward-pass-only timings. This includes image preparation and post-processing, which makes the comparison more relevant to deployment.

{{LATENCY_TABLE}}

## 4.3 Cross-dataset generalization

The domain gap is defined as:

\[
G_m = \mathrm{mAP}_{m,COCO} - \mathrm{mAP}_{m,KITTI}
\]

Higher values indicate a larger loss when transferring from the COCO visual domain to KITTI.

{{DOMAIN_GAP_TABLE}}

{{FIGURES}}

# 5. Technical analysis

## 5.1 Cross-dataset generalization

{{INTERPRETATION}}

The expected sources of transfer degradation are changes in camera viewpoint, object scale, weather and illumination, road-scene composition, occlusion, and annotation policy. A model can have strong COCO mAP while still losing substantial accuracy on KITTI because COCO and KITTI contain different distributions of object sizes and scene layouts.

The per-class and per-scale results should be read together. If vehicle AP remains relatively high while cyclist or small-object AP collapses, the result may indicate both domain shift and taxonomy mismatch. The cyclist category is particularly sensitive because COCO commonly provides separate person and bicycle boxes, whereas KITTI often represents a cyclist as a combined object.

## 5.2 Speed-accuracy tradeoff

Faster R-CNN performs proposal generation and RoI refinement, which can improve localization but introduces additional stages and post-processing. YOLO predicts densely in one pass and is expected to provide the strongest latency baseline. RT-DETR removes traditional NMS through query-based set prediction, but its transformer decoder and multi-scale feature processing may alter the speed-accuracy balance.

The appropriate conclusion is not simply that the fastest model is best. The result should be described as a Pareto comparison: a model is attractive when it provides higher accuracy at comparable latency, or substantially lower latency for a tolerable accuracy reduction.

## 5.3 Implementation-level interpretation

The code path in the tutorial is:

```text
image -> dataset adapter -> detector adapter -> unified labels -> COCO evaluator
      -> metrics.json + per-image counts + annotated video
```

The evaluation adapter maps class names rather than assuming that detector class IDs have the same meaning. This is necessary because COCO, KITTI, Torchvision, Hugging Face, and Ultralytics use different category vocabularies and ID orderings.

# 6. Failure analysis

The experiment writes `failure_candidates.csv` using per-image prediction and ground-truth counts. These rows are triage candidates, not proof of a semantic failure. At least five candidates should be visually confirmed in the annotated videos before submission.

{{FAILURE_TEXT}}

The manual audit should classify each confirmed example as one or more of:

| Failure type | What to inspect |
|---|---|
| Missed object | Ground-truth object has no corresponding prediction |
| Duplicate detection | Multiple predictions describe the same object |
| Localization error | Prediction overlaps the object but has poor IoU |
| Small-object failure | Distant or low-resolution object is missed |
| Occlusion failure | Partially visible object is missed or mislocalized |
| Taxonomy mismatch | COCO and KITTI definitions disagree |
| False positive | Prediction has no corresponding ground-truth object |

The report should include several annotated examples with a short explanation of why each example matters. Count-based triage must not be described as a substitute for visual verification.

# 7. Limitations and reproducibility

This is a zero-shot evaluation, so it measures transfer and deployment behavior rather than the maximum achievable accuracy after adaptation. The COCO-to-KITTI comparison is also affected by the coarse taxonomy and the cyclist annotation mismatch. A fixed score threshold is used to keep comparisons consistent, although thresholds can affect recall and the number of predictions retained before evaluation. KITTI evaluation is performed on a labeled local split and should not be described as an official hidden-test benchmark.

The submission should include the final PDF, the experiment code, `config.json`, `commands.log`, `environment.json`, `metrics.json`, `latency.json`, generated plots, and the failure-audit table.

# 8. Conclusions

{{CONCLUSIONS}}

# References

1. Ren, S., He, K., Girshick, R., and Sun, J. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” NeurIPS 2015. https://arxiv.org/abs/1506.01497
2. Tian, Z., Shen, C., Chen, H., and He, T. “FCOS: Fully Convolutional One-Stage Object Detection.” ICCV 2019. https://arxiv.org/abs/1904.01355
3. Carion, N. et al. “End-to-End Object Detection with Transformers.” ECCV 2020. https://arxiv.org/abs/2005.12872
4. Zhao, Y. et al. “DETRs Beat YOLOs on Real-time Object Detection.” CVPR 2024. https://arxiv.org/abs/2304.08069
5. PyTorch Torchvision. “Faster R-CNN.” https://docs.pytorch.org/vision/master/models/faster_rcnn.html
6. PyTorch Torchvision source code. https://github.com/pytorch/vision/tree/main/torchvision/models/detection
7. Ultralytics. “YOLO Documentation and Repository.” https://docs.ultralytics.com/ and https://github.com/ultralytics/ultralytics
8. lyuwenyu et al. “Official RT-DETR Repository.” https://github.com/lyuwenyu/RT-DETR
9. Feng, C. et al. “TOOD: Task-aligned One-stage Object Detection.” ICCV 2021. https://arxiv.org/abs/2108.07755
10. Li, X. et al. “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection.” NeurIPS 2020. https://arxiv.org/abs/2006.04388
11. DeepDataMiningLearning. “ngdet Tutorial - Zero-shot 2D Detection Across Datasets.” https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md

# Appendix A. Implementation and evidence map

The experiment is not based only on screenshots or published benchmark tables. The following files are the evidence chain used to connect the command line to the reported results:

| Evidence | File or artifact | Role in the experiment |
|---|---|---|
| Model and dataset command | `commands.log` | Records the exact command and working directory |
| Environment | `environment.json` | Records Python, package versions, GPU, driver, and platform |
| Detection metrics | `metrics.json` | Stores mAP, AP50, AP75, AP by scale, AR, and per-class AP |
| Latency metrics | `latency.json` | Stores mean, P50, P90, FPS, and timing mode |
| Cross-domain analysis | `domain_gap.csv` | Computes COCO-to-KITTI mAP loss for each model |
| Human-audit candidates | `failure_candidates.csv` | Prioritizes images for visual failure verification |
| Qualitative evidence | `video_*__*.mp4` | Shows predictions and ground truth together |

The tutorial implementation also provides a clear pipeline boundary: dataset adapters create samples in a common format, detector adapters return predictions in the same label space, and the evaluator computes COCO-style metrics. This separation makes it possible to compare model families without silently changing the evaluation logic between models.

# Appendix B. Submission checklist

- [ ] The final PDF is between 6 and 10 pages.
- [ ] Three model configurations from at least two detector families are evaluated.
- [ ] At least two datasets or domains are evaluated.
- [ ] The final table contains measured values rather than placeholders.
- [ ] Hardware, software versions, input resolution, threshold, and timing protocol are stated.
- [ ] The exact command and configuration are included.
- [ ] Cross-dataset generalization analysis is included.
- [ ] Speed-accuracy analysis is included.
- [ ] At least five failure candidates are visually confirmed.
- [ ] The code bundle and final result artifacts are submitted with the PDF.

# Appendix C. Artifact manifest

The following files are generated by the supplied code and form the reproducibility package for this report:

| File | Description | Submission purpose |
|---|---|---|
| `config.json` | Model, dataset, threshold, and timing configuration | Defines the experiment inputs |
| `commands.log` | Timestamped commands and console output | Shows what was actually run |
| `environment.json` | Hardware and package inventory | Documents the runtime |
| `metrics.json` | COCO-style accuracy metrics | Supports the quantitative tables |
| `latency.json` | Mean, percentile, and FPS timings | Supports the speed analysis |
| `summary.csv` | Flattened accuracy table | Enables independent checking |
| `domain_gap.csv` | COCO-to-KITTI gap calculations | Supports the transfer analysis |
| `failure_candidates.csv` | Count-based visual-audit candidates | Supports failure analysis |
| `video_*__*.mp4` | Prediction/ground-truth visualizations | Supports manual qualitative review |

The PDF should be generated only after the final run, so that the report, tables, plots, and artifact manifest all describe the same experiment version.
