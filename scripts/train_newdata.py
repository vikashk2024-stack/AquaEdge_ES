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

def train_new_data():
    data_yaml_path = Path("datasets/plankton/new-data-set/data.yaml")
    
    if not data_yaml_path.exists():
        print(f"Error: {data_yaml_path} not found.")
        return
        
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    names = data.get('names', {})
    print("Loaded data.yaml classes:", names)
    
    if isinstance(names, dict):
        target_classes = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        target_classes = names
    else:
        print("Error: 'names' in data.yaml is not a list or dictionary.")
        return
        
    print("Classes verified successfully. Number of classes:", len(target_classes))

    project_dir = "runs"
    name = "aquaedge_newdata"
    
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
            print("Memory issue with batch 8, retrying with batch 4...")
            batch_size = 4
            model = YOLO("yolo11n.pt")
            results = model.train(
                data=str(data_yaml_path),
                epochs=50,
                imgsz=640,
                batch=batch_size,
                project=project_dir,
                name=name + "_b4",
                device=device,
                exist_ok=False
            )
            name = name + "_b4"
        else:
            raise e

    duration_seconds = time.time() - start_time
    duration_str = f"{int(duration_seconds // 3600)}h {int((duration_seconds % 3600) // 60)}m {int(duration_seconds % 60)}s"

    print("\n--- TRAINING COMPLETE ---")
    
    best_model_path = Path(project_dir) / name / "weights" / "best.pt"
    
    print("\n--- EVALUATING ON VALIDATION SET ---")
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
    
    report_path = Path("datasets/reports/newdata_training_report.md")
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

    val_p = f"{val_data['overall']['p']:.4f}" if val_data else "N/A"
    val_r = f"{val_data['overall']['r']:.4f}" if val_data else "N/A"
    val_map50 = f"{val_data['overall']['map50']:.4f}" if val_data else "N/A"
    val_map95 = f"{val_data['overall']['map95']:.4f}" if val_data else "N/A"

    test_p = f"{test_data['overall']['p']:.4f}" if test_data else "N/A"
    test_r = f"{test_data['overall']['r']:.4f}" if test_data else "N/A"
    test_map50 = f"{test_data['overall']['map50']:.4f}" if test_data else "N/A"
    test_map95 = f"{test_data['overall']['map95']:.4f}" if test_data else "N/A"

    report_md = f"""# AquaEdge YOLO11 New Dataset Training Report

## 1. Environment & Configuration
- **Dataset:** `datasets/plankton/new-data-set/data.yaml`
- **Model:** YOLO11n (`yolo11n.pt`)
- **Epochs:** 50
- **Image Size:** 640
- **Batch Size:** {batch_size}
- **Hardware Device:** {device}
- **Training Duration:** {duration_str}

## 2. Overall Validation Metrics
- **Precision:** {val_p}
- **Recall:** {val_r}
- **mAP@50:** {val_map50}
- **mAP@50-95:** {val_map95}

### Per-Class Validation Metrics
{format_per_class(val_data)}

## 3. Overall Test Metrics
- **Precision:** {test_p}
- **Recall:** {test_r}
- **mAP@50:** {test_map50}
- **mAP@50-95:** {test_map95}

### Per-Class Test Metrics
{format_per_class(test_data)}

## 4. Artifact Locations
- **Best Model Weights:** `runs/{name}/weights/best.pt`
- **Last Model Weights:** `runs/{name}/weights/last.pt`
- **Training Results/Metrics:** `runs/{name}/results.csv`
- **Confusion Matrix:** `runs/{name}/confusion_matrix.png`
- **Test Set Visualizations:** `runs/{name}_test_eval/`
"""

    with open(report_path, 'w') as f:
        f.write(report_md)
        
    print(f"NEW DATA TRAINING COMPLETE. Report saved to {report_path}")

if __name__ == "__main__":
    train_new_data()
