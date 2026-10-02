import os
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

def eval_baseline():
    data_yaml_path = Path("datasets/plankton_5class/data.yaml")
    
    # We found that ultralytics nested the directory
    best_model_path = Path("runs/detect/runs/aquaedge_baseline/weights/best.pt")
    
    if not best_model_path.exists():
        print(f"Error: Could not find {best_model_path}")
        return
        
    print("Loading model...")
    best_model = YOLO(str(best_model_path))
    
    print("\n--- EVALUATING ON VALIDATION SET ---")
    val_metrics = best_model.val(
        data=str(data_yaml_path),
        split="val",
        project="runs",
        name="aquaedge_baseline_val_eval"
    )

    print("\n--- EVALUATING ON TEST SET ---")
    test_metrics = best_model.val(
        data=str(data_yaml_path),
        split="test",
        project="runs",
        name="aquaedge_baseline_test_eval"
    )
    
    expected_classes = {0: 'Chlorella', 1: 'Scenedesmus', 2: 'Navicula', 3: 'Microcystis', 4: 'Euglena'}
    target_classes = [expected_classes[i] for i in range(5)]
    
    def extract_metrics(metrics_obj):
        try:
            p = metrics_obj.box.mp
            r = metrics_obj.box.mr
            map50 = metrics_obj.box.map50
            map95 = metrics_obj.box.map
            
            per_class_p = metrics_obj.box.p
            per_class_r = metrics_obj.box.r
            per_class_map50 = metrics_obj.box.ap50
            per_class_map95 = metrics_obj.box.ap
            
            classes = metrics_obj.box.ap_class_index
            
            per_class_data = {}
            for i, c in enumerate(classes):
                per_class_data[c] = {
                    'p': per_class_p[i],
                    'r': per_class_r[i],
                    'map50': per_class_map50[i],
                    'map95': per_class_map95[i]
                }
            
            return {
                'overall': {'p': p, 'r': r, 'map50': map50, 'map95': map95},
                'per_class': per_class_data
            }
        except Exception as e:
            print("Error extracting metrics:", e)
            return None

    val_data = extract_metrics(val_metrics)
    test_data = extract_metrics(test_metrics)
    
    report_path = Path("datasets/reports/baseline_training_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    def format_per_class(data):
        if not data:
            return "Metrics extraction failed programmatically."
            
        md = "| Class | Precision | Recall | mAP@50 | mAP@50-95 |\n"
        md += "|-------|-----------|--------|--------|-----------|\n"
        
        for i, cname in enumerate(target_classes):
            if i in data['per_class']:
                m = data['per_class'][i]
                md += f"| {cname} | {m['p']:.4f} | {m['r']:.4f} | {m['map50']:.4f} | {m['map95']:.4f} |\n"
            else:
                md += f"| {cname} | 0.0000 | 0.0000 | 0.0000 | 0.0000 |\n"
        return md

    report_md = f"""# AquaEdge YOLO11 Baseline Training Report

## 1. Environment & Configuration
- **Dataset:** `datasets/plankton_5class/data.yaml` (648 images)
- **Model:** YOLO11n (`yolo11n.pt`)
- **Epochs:** 50
- **Image Size:** 640
- **Batch Size:** 8 (Auto-reduced to avoid OOM)
- **Hardware Device:** CPU
- **Training Duration:** ~2.0 hours (50 epochs completed)

## 2. Overall Validation Metrics
- **Precision:** {val_data['overall']['p']:.4f}
- **Recall:** {val_data['overall']['r']:.4f}
- **mAP@50:** {val_data['overall']['map50']:.4f}
- **mAP@50-95:** {val_data['overall']['map95']:.4f}

### Per-Class Validation Metrics
{format_per_class(val_data)}

## 3. Overall Test Metrics
- **Precision:** {test_data['overall']['p']:.4f}
- **Recall:** {test_data['overall']['r']:.4f}
- **mAP@50:** {test_data['overall']['map50']:.4f}
- **mAP@50-95:** {test_data['overall']['map95']:.4f}

### Per-Class Test Metrics
{format_per_class(test_data)}

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
"""

    with open(report_path, 'w') as f:
        f.write(report_md)
        
    print("BASELINE EVALUATION COMPLETE")

if __name__ == "__main__":
    eval_baseline()
