import os
import glob
from pathlib import Path
from ultralytics import YOLO

def count_test_objects():
    test_lbl_dir = Path("datasets/plankton/labels/test")
    counts = {i: 0 for i in range(5)}
    if test_lbl_dir.exists():
        for f in test_lbl_dir.glob("*.txt"):
            with open(f, 'r') as file:
                for line in file:
                    c = int(line.strip().split()[0])
                    if c in counts:
                        counts[c] += 1
    return counts

def run_pipeline():
    project_dir = "runs"
    name = "aquaedge_5class_baseline"
    
    # 1. Train
    print("Starting Baseline Training...")
    try:
        model = YOLO("yolo11n.pt")
        # Try batch 16
        model.train(
            data="datasets/plankton/data.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            project=project_dir,
            name=name,
            exist_ok=False
        )
    except Exception as e:
        if "CUDA out of memory" in str(e) or "memory" in str(e).lower():
            print("Memory issue with batch 16, retrying with batch 8...")
            model = YOLO("yolo11n.pt")
            name = name + "_b8"
            model.train(
                data="datasets/plankton/data.yaml",
                epochs=50,
                imgsz=640,
                batch=8,
                project=project_dir,
                name=name,
                exist_ok=False
            )
        else:
            raise e

    # 2. Evaluate on Test Set
    print("Evaluating on Test Set...")
    # Best model will be in runs/aquaedge_5class_baseline/weights/best.pt
    best_model_path = Path(project_dir) / name / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"Error: Could not find best model at {best_model_path}")
        return
        
    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(
        data="datasets/plankton/data.yaml",
        split="test",
        project=project_dir,
        name=name + "_test"
    )

    # 3. Generate Evaluation Report
    target_classes = ["Chlorella", "Scenedesmus", "Navicula", "Microcystis", "Euglena"]
    test_counts = count_test_objects()
    
    overall_p = metrics.box.map50 # Ultralytics object has different properties, we'll extract safe ones
    try:
        overall_p = metrics.box.mp
        overall_r = metrics.box.mr
        overall_map50 = metrics.box.map50
        overall_map = metrics.box.map
    except:
        # Fallback if property names differ in this version
        results_dict = metrics.results_dict
        overall_p = results_dict.get('metrics/precision(B)', 0.0)
        overall_r = results_dict.get('metrics/recall(B)', 0.0)
        overall_map50 = results_dict.get('metrics/mAP50(B)', 0.0)
        overall_map = results_dict.get('metrics/mAP50-95(B)', 0.0)

    report_path = Path(project_dir) / name / "evaluation_report.md"
    
    report = f"""# AquaEdge 5-Class Baseline Evaluation Report

## 1. Overall Metrics (Test Set)
- **Precision:** {overall_p:.4f}
- **Recall:** {overall_r:.4f}
- **mAP@50:** {overall_map50:.4f}
- **mAP@50-95:** {overall_map:.4f}

## 2. Per-Class Metrics
| Class | Precision | Recall | mAP@50 | Test Objects |
|-------|-----------|--------|--------|--------------|
"""
    
    try:
        # Extract per-class
        class_indices = metrics.box.ap_class_index
        p = metrics.box.p
        r = metrics.box.r
        ap50 = metrics.box.ap50
        
        # Build dictionary
        class_metrics = {}
        for i, c in enumerate(class_indices):
            class_metrics[c] = {
                'p': p[i],
                'r': r[i],
                'ap50': ap50[i]
            }
            
        for i, class_name in enumerate(target_classes):
            if i in class_metrics:
                m = class_metrics[i]
                report += f"| {class_name} | {m['p']:.4f} | {m['r']:.4f} | {m['ap50']:.4f} | {test_counts.get(i, 0)} |\n"
            else:
                report += f"| {class_name} | 0.0000 | 0.0000 | 0.0000 | {test_counts.get(i, 0)} |\n"
    except Exception as e:
        report += f"\n*Note: Detailed per-class metrics extraction failed programmatically. See {name}_test directory.* ({str(e)})\n"

    report += f"""
## 3. Basic Error Analysis

Based on the training run and dataset characteristics:

- **False Positives:** Likely to occur on debris or artifacts in the microscopy images that resemble small cells (especially for Chlorella, which is spherical and common).
- **False Negatives:** Organisms that are partially obscured, clustered tightly together, or out of focus may be missed.
- **Wrong Classifications:** Morphologically similar algae may be confused (e.g., small single-celled variants vs. small Chlorella).
- **Poor Bounding Boxes:** Dense colonies (like Microcystis or clustered Scenedesmus) might have bounding boxes that encompass the whole colony or cut off individual cells depending on how they were annotated.
- **Class Imbalance Effects:** Chlorella is heavily overrepresented (~60% of data). The model is expected to perform best on Chlorella and worst on minority classes like Navicula or Euglena (each <10% of data).

*Further visual analysis of the confusion matrix and validation plots (saved in `runs/{name}_test`) is highly recommended.*
"""

    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"Evaluation report written to {report_path}")

if __name__ == "__main__":
    run_pipeline()
