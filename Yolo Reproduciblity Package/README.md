# Weld Defect Segmentation: YOLO11 vs. YOLO26 Replication Package

This supplementary repository contains the group-assignment manifests, training configurations, and core evaluation scripts supporting the paper:
**"Instance Segmentation of Weld Surface Defects: A Controlled Comparison of YOLO11 and YOLO26"** (IRIS 2027 / Springer LNEE).

---

## 1. Repository Structure

```text
├── manifests/
│   ├── group_assignments.csv   # Full 17,063 image-to-group mapping across 1,241 groups
│   ├── train_images.txt        # 13,551 training image filenames (951 groups)
│   ├── val_images.txt          # 1,756 validation image filenames (158 groups)
│   ├── test_images.txt         # 1,756 test image filenames (132 groups)
│   └── robust_subset_254.txt   # 254 fixed seed-0 robustness audit filenames
├── configs/
│   ├── data.yaml               # Ultralytics dataset configuration (Crack, Porosity, Spatters, Welding line)
│   └── train_hyperparams.yaml  # 60 epochs, batch 16, lr 0.01, MuSGD optimizer, close_mosaic 10
├── scripts/
│   ├── audit_leakage.py        # Replicates Table 1 leakage audit (base ID, parent ID, pHash <= 5)
│   ├── union_find_grouping.py  # Disjoint-set grouping algorithm ensuring zero cross-split leakage
│   ├── group_bootstrap_eval.py # 1,000 paired group-bootstrap resampling over 132 test groups
│   └── sharpness_gate.py       # Pre-inference Laplacian variance filter (tau = 50.0)
└── results_ledger.csv          # Comprehensive empirical ledger of all runs, seeds, and conditions
```

---

## 2. Dataset Access and Setup

1. **Source Dataset**: Download the public **Roboflow Welding Project (version 3)** released under [CC BY 4.0](https://universe.roboflow.com/welding-vnkkh/welding-project-iaruw).
2. **Directory Structure**:
   Place the uncompressed images into a directory matching `configs/data.yaml`:
   ```text
   dataset_v2/
   ├── train/images/
   ├── valid/images/
   └── test/images/
   ```
3. **Partitioning**: Use `manifests/train_images.txt`, `manifests/val_images.txt`, and `manifests/test_images.txt` to partition the images according to our 1,241-group leakage-free split.

---

## 3. Training & Evaluation Commands

### Environment
```bash
pip install ultralytics==8.4.131 torch opencv-python pandas numpy
```

### Training (3 Seeds)
```bash
# YOLO11n-seg
yolo segment train data=configs/data.yaml model=yolo11n-seg.pt epochs=60 batch=16 imgsz=640 seed=0
yolo segment train data=configs/data.yaml model=yolo11n-seg.pt epochs=60 batch=16 imgsz=640 seed=1
yolo segment train data=configs/data.yaml model=yolo11n-seg.pt epochs=60 batch=16 imgsz=640 seed=2

# YOLO26n-seg
yolo segment train data=configs/data.yaml model=yolo26n-seg.pt epochs=60 batch=16 imgsz=640 seed=0
yolo segment train data=configs/data.yaml model=yolo26n-seg.pt epochs=60 batch=16 imgsz=640 seed=1
yolo segment train data=configs/data.yaml model=yolo26n-seg.pt epochs=60 batch=16 imgsz=640 seed=2
```

### Validation Modes
- **YOLO11n-seg**: Standard validation with NMS.
- **YOLO26n-seg (NMS)**: Evaluated using the one-to-many head (`nms=True`).
- **YOLO26n-seg (NMS-free / End-to-End)**: Evaluated using native one-to-one decoding (`end2end=True`).

---

## 4. Citation and Attribution

- Public dataset: Roboflow Universe Welding Project v3 (CC BY 4.0).
- Models: Ultralytics YOLO11 and YOLO26.
