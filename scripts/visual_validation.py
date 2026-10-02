import os
import glob
import cv2
import random
from pathlib import Path

output_dir = Path("datasets/plankton")
val_dir = Path("datasets/reports/visual_validation")
val_dir.mkdir(parents=True, exist_ok=True)

target_classes = ["Chlorella", "Scenedesmus", "Navicula", "Microcystis", "Euglena"]

def generate_visual_validation():
    # Gather all final images
    all_imgs = []
    for split in ['train', 'val', 'test']:
        img_dir = output_dir / "images" / split
        if img_dir.exists():
            all_imgs.extend(glob.glob(str(img_dir / "*.*")))
            
    if not all_imgs:
        print("No images found for visual validation.")
        return
        
    sample_size = min(20, len(all_imgs))
    sampled = random.sample(all_imgs, sample_size)
    
    for img_path in sampled:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w, _ = img.shape
        
        # Find label
        split = img_path.parent.name
        lbl_path = output_dir / "labels" / split / (img_path.stem + ".txt")
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_name = target_classes[c]
                    cv2.putText(img, label_name, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        out_path = val_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        
    print(f"Saved {sample_size} validation images to {val_dir}")

if __name__ == "__main__":
    generate_visual_validation()
