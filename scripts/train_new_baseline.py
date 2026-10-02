import os
import yaml
import time
import pandas as pd
from pathlib import Path
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    import subprocess
    import sys
    print("Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "torch", "torchvision"])
    from ultralytics import YOLO
    import torch

def train_new_baseline():
    data_yaml_path = Path("datasets/plankton_5class/data.yaml")
    
    if not data_yaml_path.exists():
        print(f"Error: {data_yaml_path} not found.")
        return
        
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    names = data.get('names', {})
    print("Loaded data.yaml classes:", names)
    
    expected_classes = {0: 'Chlorella', 1: 'Scenedesmus', 2: 'Navicula', 3: 'Microcystis', 4: 'Euglena'}
    
    # Ensure strict matching (dictionary or list conversion)
    actual_classes = {}
    if isinstance(names, dict):
        actual_classes = names
    elif isinstance(names, list):
        actual_classes = {i: n for i, n in enumerate(names)}
        
    for k, v in expected_classes.items():
        if actual_classes.get(k) != v:
            print(f"Error: Class mapping mismatch. Expected {k}: {v}, got {k}: {actual_classes.get(k)}")
            return
            
    print("Classes verified successfully.")

    project_dir = "runs"
    name = "aquaedge_baseline"
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    start_time = time.time()
    
    print("\n--- STARTING TRAINING ---")
    model = YOLO("yolo11n.pt")
    
    batch_size = 8
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=50,
            imgsz=640,
            batch=batch_size,
            workers=0,
            project=project_dir,
            name=name,
            device=device,
            exist_ok=True
        )
    except Exception as e:
        if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
            print("Memory issue with batch 16, retrying with batch 8...")
            batch_size = 8
            model = YOLO("yolo11n.pt")
            results = model.train(
                data=str(data_yaml_path),
                epochs=50,
                imgsz=640,
                batch=batch_size,
                project=project_dir,
                name=name + "_b8",
                device=device,
                exist_ok=False
            )
            name = name + "_b8"
        else:
            raise e

    duration_seconds = time.time() - start_time
    duration_str = f"{int(duration_seconds // 3600)}h {int((duration_seconds % 3600) // 60)}m {int(duration_seconds % 60)}s"

    print("\n--- TRAINING COMPLETE ---")
    
    best_model_path = Path(project_dir) / name / "weights" / "best.pt"
    
    print("\n--- EVALUATING ON VALIDATION SET ---")
    # Actually results object from train() contains validation metrics of the best epoch.
    # We will explicitly run val() again to ensure standard output structures are generated if needed,
    # or just use results. But let's run model.val() on validation explicitly to get clean metrics.
    best_model = YOLO(str(best_model_path))
    val_metrics = best_model.val(
        data=str(data_yaml_path),
        split="val",
        project=project_dir,
        name=name + "_val_eval"
    )

    print("\n--- EVALUATING ON TEST SET ---")
    test_metrics = best_model.val(
        data=str(data_yaml_path),
        split="test",
        project=project_dir,
        name=name + "_test_eval"
    )
    
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
        except:
            return None

    val_data = extract_metrics(val_metrics)
    test_data = extract_metrics(test_metrics)
    
    report_path = Path("datasets/reports/baseline_training_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    target_classes = [expected_classes[i] for i in range(5)]
    
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
- **Batch Size:** {batch_size}
- **Hardware Device:** {device}
- **Training Duration:** {duration_str}

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
- **Class Imbalance:** Navicula (35 objects) and Microcystis (47 objects) make up less than 10% of the dataset combined. Because no synthetic augmentation or weighting was applied yet, their performance is expected to lag significantly behind majority classes like Chlorella (321 objects).
- **Confusion Matrix:** Please inspect `runs/{name}/confusion_matrix.png` for specific misclassifications (e.g. background false positives or confusion between similar shapes).
- **Known Limitations:** This is a pure baseline on a heavily imbalanced 648-image dataset.

## 5. Artifact Locations
- **Best Model Weights:** `runs/{name}/weights/best.pt`
- **Last Model Weights:** `runs/{name}/weights/last.pt`
- **Training Results/Metrics:** `runs/{name}/results.csv`
- **Confusion Matrix:** `runs/{name}/confusion_matrix.png`
- **Test Set Visualizations:** `runs/{name}_test_eval/`
"""

    with open(report_path, 'w') as f:
        f.write(report_md)
        
    print("BASELINE TRAINING COMPLETE")

if __name__ == "__main__":
    train_new_baseline()
