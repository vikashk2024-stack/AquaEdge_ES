# FILTERED DATASET AUDIT REPORT

## 1. Overall Statistics
- **Original image count:** 7270
- **Cleaned image count:** 648
- **Images removed:** 6622
- **Original object count:** 14665
- **Cleaned object count:** 851
- **Objects removed:** 13817

## 2. Train/Validation/Test Splits
- **Train images:** 362
- **Validation images:** 143
- **Test images:** 143

## 3. Errors and Anomalies
- **Malformed annotation files found:** 3
- **Empty annotation files found:** 41

## 4. Class Distribution (Cleaned Dataset)
| Class ID | Class Name | Image Count | Object Count | % |
|---|---|---|---|---|
| 0 | Chlorella | 192 | 321 | 37.72% |
| 1 | Scenedesmus | 205 | 216 | 25.38% |
| 2 | Navicula | 34 | 35 | 4.11% |
| 3 | Microcystis | 37 | 47 | 5.52% |
| 4 | Euglena | 228 | 232 | 27.26% |


## 5. Ready for YOLO Training
YES. The cleaned dataset passes all structural validation rules and contains only the 5 target classes.
