# Dataset setup for the RTX 3060 experiment

The code expects two datasets: COCO val2017 and the KITTI object-detection
training split. The models are evaluated without fine-tuning, so the dataset
directories and annotation files must exist before the smoke run.

## COCO val2017

Download these official files:

- Images: <https://images.cocodataset.org/zips/val2017.zip>
- Annotations: <https://images.cocodataset.org/annotations/annotations_trainval2017.zip>

After extraction, the required layout is:

```text
/data/coco2017/
├── val2017/
│   ├── 000000000139.jpg
│   └── ...
└── annotations/
    └── instances_val2017.json
```

The annotation archive also contains train annotations; only
`instances_val2017.json` is required for this assignment.

If `curl` reports a certificate or proxy hostname error, do not use `curl -k`
for a graded dataset download. Download the same official URLs in a browser or
fix the machine's certificate/proxy configuration, then verify the ZIP before
extracting it. A valid ZIP should be much larger than a few KB and should open
with:

```bash
unzip -t /data/coco2017/val2017.zip
unzip -t /data/coco2017/annotations_trainval2017.zip
```

## KITTI object detection

Download the KITTI 2D object-detection files from the official benchmark page:

<https://www.cvlibs.net/datasets/kitti/eval_object.php>

The evaluation harness needs the image and label portions of the object
detection training data. The required layout is:

```text
/data/Kitti/
└── training/
    ├── image_2/
    │   ├── 000000.png
    │   └── ...
    └── label_2/
        ├── 000000.txt
        └── ...
```

Use the exact capitalization and directory names expected by the tutorial. A
KITTI directory containing only `image_2` but no `label_2` cannot produce
supervised metrics.

## Configure paths

From the bundle directory:

```bash
cp config.example.json config.json
```

Edit these fields in `config.json`:

```json
{
  "dataset_roots": {
    "coco": "/data/coco2017/val2017",
    "kitti": "/data/Kitti"
  },
  "coco_ann": "/data/coco2017/annotations/instances_val2017.json",
  "latency_image": "/data/Kitti/training/image_2/000008.png"
}
```

The `latency_image` must be an existing readable image. The `tutorial_root`
must point to the directory that contains the cloned `DeepDataMiningLearning`
package, for example:

```text
/work/DeepDataMiningLearning/
└── DeepDataMiningLearning/
```

## Preflight path checks

Run these commands before the smoke phase:

```bash
test -d /data/coco2017/val2017
test -f /data/coco2017/annotations/instances_val2017.json
test -d /data/Kitti/training/image_2
test -d /data/Kitti/training/label_2
test -f /data/Kitti/training/image_2/000008.png
python -m json.tool config.json >/dev/null
```

On Windows PowerShell, replace `test -d`/`test -f` with `Test-Path` and use
Windows paths in `config.json`, such as `C:/data/coco2017/val2017`.

