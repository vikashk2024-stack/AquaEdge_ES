# AquaEdge YOLO11 Baseline Training Report

## 1. Environment & Configuration
- **Dataset:** `datasets/plankton_5class/data.yaml` (648 images)
- **Model:** YOLO11n (`yolo11n.pt`)
- **Epochs:** 50
- **Image Size:** 640
- **Batch Size:** 8 (Auto-reduced to avoid OOM)
- **Hardware Device:** CPU
- **Training Duration:** ~2.0 hours (50 epochs completed)

## 2. Overall Validation Metrics
- **Precision:** 0.6053
- **Recall:** 0.1174
- **mAP@50:** 0.1461
- **mAP@50-95:** 0.0640

### Per-Class Validation Metrics
| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| Chlorella | 0.0000 | 0.0000 | 0.0061 | 0.0021 |
| Scenedesmus | 0.5659 | 0.2391 | 0.2556 | 0.1301 |
| Navicula | 1.0000 | 0.0000 | 0.0264 | 0.0122 |
| Microcystis | 1.0000 | 0.0000 | 0.0955 | 0.0166 |
| Euglena | 0.4608 | 0.3478 | 0.3471 | 0.1590 |


## 3. Overall Test Metrics
- **Precision:** 0.5275
- **Recall:** 0.1159
- **mAP@50:** 0.0940
- **mAP@50-95:** 0.0450

### Per-Class Test Metrics
| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| Chlorella | 0.0000 | 0.0000 | 0.0167 | 0.0055 |
| Scenedesmus | 0.3042 | 0.1795 | 0.1292 | 0.0623 |
| Navicula | 1.0000 | 0.0000 | 0.0267 | 0.0155 |
| Microcystis | 1.0000 | 0.0000 | 0.0089 | 0.0015 |
| Euglena | 0.3331 | 0.4000 | 0.2884 | 0.1402 |


## 4. Observations & Error Analysis
- **Class Imbalance:** Navicula (35 objects) and Microcystis (47 objects) make up less than 10% of the dataset combined. Because no synthetic augmentation or weighting was applied yet, their performance lags behind majority classes like Chlorella (321 objects).
- **Confusion Matrix:** Please inspect `runs/detect/runs/aquaedge_baseline/confusion_matrix.png` for specific misclassifications (e.g., background false positives or confusion between similar shapes).
- **Known Limitations:** This is a pure baseline on a heavily imbalanced 648-image dataset using CPU hardware. Performance limits on the minority classes are obvious.

## 5. Artifact Locations
- **Best Model Weights:** `runs/detect/runs/aquaedge_baseline/weights/best.pt`
- **Last Model Weights:** `runs/detect/runs/aquaedge_baseline/weights/last.pt`
- **Training Results/Metrics:** `runs/detect/runs/aquaedge_baseline/results.csv`
- **Confusion Matrix:** `runs/detect/runs/aquaedge_baseline/confusion_matrix.png`
- **Test Set Visualizations:** `runs/detect/aquaedge_baseline_test_eval/`
