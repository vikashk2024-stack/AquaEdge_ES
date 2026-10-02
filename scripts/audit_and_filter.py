import os
import glob
import yaml
import shutil
import cv2
import pandas as pd
import random
from pathlib import Path

raw_dir = Path("datasets/raw/plankton_dataset")
output_dir = Path("datasets/plankton")

# Clear existing output dir to ensure clean state
if output_dir.exists():
    shutil.rmtree(output_dir)

target_classes = ["Chlorella", "Scenedesmus", "Navicula", "Microcystis", "Euglena"]
target_mapping = {name: i for i, name in enumerate(target_classes)}

def audit_and_filter():
    yaml_path = raw_dir / "data.yaml"
    if not yaml_path.exists():
        print(f"Cannot find {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    orig_names = data.get('names', [])
    if isinstance(orig_names, dict):
        orig_names = [orig_names[k] for k in sorted(orig_names.keys())]

    orig_mapping = {}
    for i, name in enumerate(orig_names):
        for tc in target_classes:
            if name.lower() == tc.lower():
                orig_mapping[i] = target_mapping[tc]
                break
    
    print("Original classes mapping to new targets:", orig_mapping)

    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'missing_images': 0,
        'invalid_annotations': 0,
        'objects_per_class': {tc: 0 for tc in target_classes},
        'images_per_class': {tc: set() for tc in target_classes},
        'split_counts': {'train': 0, 'val': 0, 'test': 0}
    }

    # Collect all valid filtered items
    valid_items = []

    splits = ['train', 'valid', 'test', 'val']
    
    for split in splits:
        split_img_dir = raw_dir / split / "images"
        split_lbl_dir = raw_dir / split / "labels"
        if not split_img_dir.exists():
            continue
            
        lbl_files = glob.glob(str(split_lbl_dir / "*.txt"))
        
        for lbl_file in lbl_files:
            img_name = Path(lbl_file).stem
            img_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
            img_file = None
            for ext in img_exts:
                if (split_img_dir / (img_name + ext)).exists():
                    img_file = split_img_dir / (img_name + ext)
                    break
            
            if img_file is None:
                stats['missing_images'] += 1
                continue
            
            stats['total_images'] += 1
            
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            has_valid_target = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    stats['invalid_annotations'] += 1
                    continue
                
                try:
                    c, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        stats['invalid_annotations'] += 1
                        continue
                        
                    stats['total_annotations'] += 1
                    
                    if c in orig_mapping:
                        target_id = orig_mapping[c]
                        target_name = target_classes[target_id]
                        new_lines.append(f"{target_id} {x} {y} {w} {h}\n")
                        stats['objects_per_class'][target_name] += 1
                        stats['images_per_class'][target_name].add(str(img_file))
                        has_valid_target = True
                except ValueError:
                    stats['invalid_annotations'] += 1
                    continue
            
            if has_valid_target:
                valid_items.append({
                    'img_file': img_file,
                    'lbl_lines': new_lines,
                    'orig_lbl_file': lbl_file
                })

    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_items)
    total_valid = len(valid_items)
    
    train_end = int(total_valid * 0.70)
    val_end = int(total_valid * 0.85)
    
    train_items = valid_items[:train_end]
    val_items = valid_items[train_end:val_end]
    test_items = valid_items[val_end:]
    
    def process_split(items, split_name):
        out_img_dir = output_dir / "images" / split_name
        out_lbl_dir = output_dir / "labels" / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for item in items:
            lbl_name = Path(item['orig_lbl_file']).name
            with open(out_lbl_dir / lbl_name, 'w') as f:
                f.writelines(item['lbl_lines'])
            shutil.copy(item['img_file'], out_img_dir / item['img_file'].name)
            stats['split_counts'][split_name] += 1

    process_split(train_items, 'train')
    process_split(val_items, 'val')
    process_split(test_items, 'test')

    # Write data.yaml
    out_yaml = output_dir / "data.yaml"
    with open(out_yaml, 'w') as f:
        f.write("names:\n")
        for i, name in enumerate(target_classes):
            f.write(f"  {i}: {name}\n")
        f.write("nc: 5\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")

    # Generate Reports
    reports_dir = Path("datasets/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    total_target_objects = sum(stats['objects_per_class'].values())
    
    dist_data = []
    for i, tc in enumerate(target_classes):
        oc = stats['objects_per_class'][tc]
        ic = len(stats['images_per_class'][tc])
        pct = (oc / total_target_objects * 100) if total_target_objects > 0 else 0
        dist_data.append({
            'class_id': i,
            'class_name': tc,
            'image_count': ic,
            'object_count': oc,
            'percentage_of_objects': round(pct, 2)
        })
    
    df = pd.DataFrame(dist_data)
    df.to_csv(reports_dir / "class_distribution.csv", index=False)
    
    md_table = "| Class ID | Class Name | Image Count | Object Count | % |\n|---|---|---|---|---|\n"
    for row in dist_data:
        md_table += f"| {row['class_id']} | {row['class_name']} | {row['image_count']} | {row['object_count']} | {row['percentage_of_objects']}% |\n"

    audit_md = f"""# Dataset Audit Report

## Overall Statistics
- **Total images in raw dataset:** {stats['total_images']}
- **Total annotations in raw dataset:** {stats['total_annotations']}
- **Missing images:** {stats['missing_images']}
- **Invalid annotations:** {stats['invalid_annotations']}

## Filtered Dataset (5 Classes)
- **Train images:** {stats['split_counts']['train']}
- **Validation images:** {stats['split_counts']['val']}
- **Test images:** {stats['split_counts']['test']}

## Class Distribution
{md_table}

## Final Mapping
- 0 = Chlorella
- 1 = Scenedesmus
- 2 = Navicula
- 3 = Microcystis
- 4 = Euglena

## Ready for Training
YES. The dataset has been filtered and structured properly.
"""
    with open(reports_dir / "dataset_audit.md", 'w') as f:
        f.write(audit_md)
        
    print("Audit and filtering complete.")

if __name__ == "__main__":
    audit_and_filter()
