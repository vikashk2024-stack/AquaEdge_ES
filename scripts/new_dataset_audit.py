import os
import glob
import yaml
import cv2
import pandas as pd
import random
from pathlib import Path

def new_dataset_audit():
    yaml_path = Path("datasets/plankton/data.yaml")
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # The classes in data.yaml
    yaml_names = data.get('names', [])
    if isinstance(yaml_names, dict):
        yaml_names = {k: v for k, v in yaml_names.items()}
    elif isinstance(yaml_names, list):
        yaml_names = {i: v for i, v in enumerate(yaml_names)}
    
    print("Classes in data.yaml:", yaml_names)

    splits = ['train', 'valid', 'val', 'test']
    
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'missing_images': 0,
        'missing_labels': 0,
        'empty_labels': 0,
        'invalid_annotations': 0,
        'out_of_bounds_coords': 0,
        'invalid_class_ids': 0,
        'split_counts': {s: 0 for s in splits},
        'objects_per_class': {},
        'images_per_class': {}
    }

    base_dir = Path("datasets/plankton")
    all_image_paths = []
    
    # We will also check the old structure if it exists, but prioritize the user's specific folder
    # User said: "datasets/plankton/train is my downloaded datasets"
    
    for split in splits:
        # Check standard YOLO structure or the specific train folder structure
        # User structure: datasets/plankton/train/images and labels
        split_dir = base_dir / split
        if split_dir.exists() and (split_dir / "images").exists():
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
        else:
            # Maybe it's datasets/plankton/images/train?
            img_dir = base_dir / "images" / split
            lbl_dir = base_dir / "labels" / split
            
        if not img_dir.exists():
            continue
            
        img_files = glob.glob(str(img_dir / "*.*"))
        stats['split_counts'][split] += len(img_files)
        
        for img_path_str in img_files:
            img_path = Path(img_path_str)
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            stats['total_images'] += 1
            all_image_paths.append(img_path)
            
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                stats['missing_labels'] += 1
                continue
                
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                stats['empty_labels'] += 1
                continue
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    stats['invalid_annotations'] += 1
                    continue
                
                try:
                    c = int(parts[0])
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    stats['invalid_annotations'] += 1
                    continue
                    
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    stats['out_of_bounds_coords'] += 1
                    continue
                    
                stats['total_annotations'] += 1
                
                if c not in yaml_names:
                    stats['invalid_class_ids'] += 1
                    class_name = f"Unknown_{c}"
                else:
                    class_name = yaml_names[c]
                    
                if c not in stats['objects_per_class']:
                    stats['objects_per_class'][c] = 0
                    stats['images_per_class'][c] = set()
                    
                stats['objects_per_class'][c] += 1
                stats['images_per_class'][c].add(str(img_path))

    # Identify classes actually present
    present_class_ids = sorted(stats['objects_per_class'].keys())
    
    # Compare with intended target classes
    intended_classes = ["Chlorella", "Scenedesmus", "Navicula", "Microcystis", "Euglena"]
    
    dist_data = []
    total_objects = stats['total_annotations']
    
    for c in present_class_ids:
        class_name = yaml_names.get(c, f"Unknown_{c}")
        oc = stats['objects_per_class'][c]
        ic = len(stats['images_per_class'][c])
        pct = (oc / total_objects * 100) if total_objects > 0 else 0
        dist_data.append({
            'class_id': c,
            'class_name': class_name,
            'image_count': ic,
            'object_count': oc,
            'percentage': round(pct, 2)
        })

    reports_dir = Path("datasets/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(dist_data)
    df.to_csv(reports_dir / "class_distribution.csv", index=False)
    
    md_table = "| Class ID | Class Name | Image Count | Object Count | % |\n|---|---|---|---|---|\n"
    for row in dist_data:
        md_table += f"| {row['class_id']} | {row['class_name']} | {row['image_count']} | {row['object_count']} | {row['percentage']}% |\n"

    # Are all intended classes present?
    found_names = [yaml_names.get(c, f"Unknown_{c}") for c in present_class_ids]
    missing_targets = [tc for tc in intended_classes if tc not in found_names]
    extra_classes = [name for name in found_names if name not in intended_classes]

    is_suitable = True
    suitability_reasons = []
    if missing_targets:
        is_suitable = False
        suitability_reasons.append(f"Missing target classes: {', '.join(missing_targets)}")
    if extra_classes:
        is_suitable = False
        suitability_reasons.append(f"Contains extra classes not in target list: {', '.join(extra_classes)}")
    if stats['invalid_class_ids'] > 0:
        is_suitable = False
        suitability_reasons.append(f"Contains {stats['invalid_class_ids']} objects with class IDs not defined in data.yaml")
        
    suitability_str = "YES" if is_suitable else "NO\nReasons:\n- " + "\n- ".join(suitability_reasons)

    audit_md = f"""# COMPLETE DATASET AUDIT REPORT

## 1. Directory and File Information
- **Dataset path:** {base_dir.absolute()}
- **Train images directory:** {base_dir / 'train' / 'images'} (and/or {base_dir / 'images' / 'train'})
- **data.yaml defined classes ({len(yaml_names)}):** {yaml_names}

## 2. Overall Counts
- **Total images found:** {stats['total_images']}
- **Total annotated objects:** {stats['total_annotations']}
- **Missing labels:** {stats['missing_labels']}
- **Empty labels:** {stats['empty_labels']}
- **Malformed YOLO annotations:** {stats['invalid_annotations']}
- **Coordinates out of bounds:** {stats['out_of_bounds_coords']}
- **Invalid Class IDs (not in data.yaml):** {stats['invalid_class_ids']}

## 3. Train/Validation/Test Splits
- **Train images:** {stats['split_counts']['train']}
- **Validation images:** {stats['split_counts']['valid'] + stats['split_counts']['val']}
- **Test images:** {stats['split_counts']['test']}

## 4. Class Distribution (Found in Annotations)
{md_table}

## 5. Target Class Comparison
- **Intended targets:** {intended_classes}
- **Actually present names:** {found_names}
- **Missing intended targets:** {missing_targets}
- **Extra classes found:** {extra_classes}

## 6. Suitability for AquaEdge
**Suitable for training as-is:** {suitability_str}
**Additional preprocessing required?** {"YES" if not is_suitable else "NO"}
"""

    with open(reports_dir / "dataset_audit.md", 'w') as f:
        f.write(audit_md)

    # Generate 20 visualizations
    val_dir = reports_dir / "visual_validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    sample_size = min(20, len(all_image_paths))
    sampled = random.sample(all_image_paths, sample_size)
    
    for img_path in sampled:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w, _ = img.shape
        
        # Check if user structure or standard structure
        if "train" in img_path.parts:
            lbl_dir = img_path.parent.parent / "labels"
        else:
            lbl_dir = base_dir / "labels" / img_path.parent.name
            
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        c, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_name = yaml_names.get(c, f"ID_{c}")
                        cv2.putText(img, label_name, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except:
                        pass
                        
        out_path = val_dir / img_path.name
        cv2.imwrite(str(out_path), img)

    print("DATASET AUDIT COMPLETE")

if __name__ == "__main__":
    new_dataset_audit()
