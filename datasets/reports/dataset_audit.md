# COMPLETE DATASET AUDIT REPORT

## 1. Directory and File Information
- **Dataset path:** C:\Users\vikas\Desktop\Embedded_project\AquaEdge\datasets\plankton
- **Train images directory:** datasets\plankton\train\images (and/or datasets\plankton\images\train)
- **data.yaml defined classes (5):** {0: 'Chlorella', 1: 'Scenedesmus', 2: 'Navicula', 3: 'Microcystis', 4: 'Euglena'}

## 2. Overall Counts
- **Total images found:** 7270
- **Total annotated objects:** 14665
- **Missing labels:** 0
- **Empty labels:** 41
- **Malformed YOLO annotations:** 3
- **Coordinates out of bounds:** 0
- **Invalid Class IDs (not in data.yaml):** 13814

## 3. Train/Validation/Test Splits
- **Train images:** 6984
- **Validation images:** 143
- **Test images:** 143

## 4. Class Distribution (Found in Annotations)
| Class ID | Class Name | Image Count | Object Count | % |
|---|---|---|---|---|
| 0 | Chlorella | 192 | 321 | 2.19% |
| 1 | Scenedesmus | 205 | 216 | 1.47% |
| 2 | Navicula | 34 | 35 | 0.24% |
| 3 | Microcystis | 37 | 47 | 0.32% |
| 4 | Euglena | 228 | 232 | 1.58% |
| 5 | Unknown_5 | 33 | 33 | 0.23% |
| 6 | Unknown_6 | 130 | 133 | 0.91% |
| 7 | Unknown_7 | 101 | 101 | 0.69% |
| 8 | Unknown_8 | 162 | 165 | 1.13% |
| 9 | Unknown_9 | 24 | 27 | 0.18% |
| 10 | Unknown_10 | 1525 | 2874 | 19.6% |
| 11 | Unknown_11 | 262 | 285 | 1.94% |
| 12 | Unknown_12 | 21 | 21 | 0.14% |
| 13 | Unknown_13 | 36 | 36 | 0.25% |
| 14 | Unknown_14 | 302 | 330 | 2.25% |
| 15 | Unknown_15 | 236 | 242 | 1.65% |
| 16 | Unknown_16 | 506 | 559 | 3.81% |
| 17 | Unknown_17 | 33 | 36 | 0.25% |
| 18 | Unknown_18 | 345 | 390 | 2.66% |
| 19 | Unknown_19 | 81 | 81 | 0.55% |
| 20 | Unknown_20 | 240 | 355 | 2.42% |
| 21 | Unknown_21 | 830 | 1835 | 12.51% |
| 22 | Unknown_22 | 348 | 359 | 2.45% |
| 23 | Unknown_23 | 27 | 30 | 0.2% |
| 24 | Unknown_24 | 175 | 196 | 1.34% |
| 25 | Unknown_25 | 67 | 76 | 0.52% |
| 26 | Unknown_26 | 53 | 56 | 0.38% |
| 27 | Unknown_27 | 204 | 211 | 1.44% |
| 28 | Unknown_28 | 24 | 24 | 0.16% |
| 29 | Unknown_29 | 3 | 3 | 0.02% |
| 30 | Unknown_30 | 330 | 408 | 2.78% |
| 31 | Unknown_31 | 65 | 68 | 0.46% |
| 32 | Unknown_32 | 33 | 33 | 0.23% |
| 33 | Unknown_33 | 291 | 315 | 2.15% |
| 34 | Unknown_34 | 93 | 117 | 0.8% |
| 35 | Unknown_35 | 416 | 450 | 3.07% |
| 36 | Unknown_36 | 460 | 543 | 3.7% |
| 37 | Unknown_37 | 52 | 52 | 0.35% |
| 38 | Unknown_38 | 141 | 144 | 0.98% |
| 39 | Unknown_39 | 128 | 134 | 0.91% |
| 40 | Unknown_40 | 801 | 883 | 6.02% |
| 41 | Unknown_41 | 12 | 12 | 0.08% |
| 42 | Unknown_42 | 6 | 6 | 0.04% |
| 43 | Unknown_43 | 341 | 449 | 3.06% |
| 44 | Unknown_44 | 12 | 18 | 0.12% |
| 45 | Unknown_45 | 36 | 39 | 0.27% |
| 46 | Unknown_46 | 42 | 42 | 0.29% |
| 47 | Unknown_47 | 156 | 156 | 1.06% |
| 48 | Unknown_48 | 74 | 89 | 0.61% |
| 49 | Unknown_49 | 999 | 1386 | 9.45% |
| 50 | Unknown_50 | 9 | 9 | 0.06% |
| 51 | Unknown_51 | 3 | 3 | 0.02% |


## 5. Target Class Comparison
- **Intended targets:** ['Chlorella', 'Scenedesmus', 'Navicula', 'Microcystis', 'Euglena']
- **Actually present names:** ['Chlorella', 'Scenedesmus', 'Navicula', 'Microcystis', 'Euglena', 'Unknown_5', 'Unknown_6', 'Unknown_7', 'Unknown_8', 'Unknown_9', 'Unknown_10', 'Unknown_11', 'Unknown_12', 'Unknown_13', 'Unknown_14', 'Unknown_15', 'Unknown_16', 'Unknown_17', 'Unknown_18', 'Unknown_19', 'Unknown_20', 'Unknown_21', 'Unknown_22', 'Unknown_23', 'Unknown_24', 'Unknown_25', 'Unknown_26', 'Unknown_27', 'Unknown_28', 'Unknown_29', 'Unknown_30', 'Unknown_31', 'Unknown_32', 'Unknown_33', 'Unknown_34', 'Unknown_35', 'Unknown_36', 'Unknown_37', 'Unknown_38', 'Unknown_39', 'Unknown_40', 'Unknown_41', 'Unknown_42', 'Unknown_43', 'Unknown_44', 'Unknown_45', 'Unknown_46', 'Unknown_47', 'Unknown_48', 'Unknown_49', 'Unknown_50', 'Unknown_51']
- **Missing intended targets:** []
- **Extra classes found:** ['Unknown_5', 'Unknown_6', 'Unknown_7', 'Unknown_8', 'Unknown_9', 'Unknown_10', 'Unknown_11', 'Unknown_12', 'Unknown_13', 'Unknown_14', 'Unknown_15', 'Unknown_16', 'Unknown_17', 'Unknown_18', 'Unknown_19', 'Unknown_20', 'Unknown_21', 'Unknown_22', 'Unknown_23', 'Unknown_24', 'Unknown_25', 'Unknown_26', 'Unknown_27', 'Unknown_28', 'Unknown_29', 'Unknown_30', 'Unknown_31', 'Unknown_32', 'Unknown_33', 'Unknown_34', 'Unknown_35', 'Unknown_36', 'Unknown_37', 'Unknown_38', 'Unknown_39', 'Unknown_40', 'Unknown_41', 'Unknown_42', 'Unknown_43', 'Unknown_44', 'Unknown_45', 'Unknown_46', 'Unknown_47', 'Unknown_48', 'Unknown_49', 'Unknown_50', 'Unknown_51']

## 6. Suitability for AquaEdge
**Suitable for training as-is:** NO
Reasons:
- Contains extra classes not in target list: Unknown_5, Unknown_6, Unknown_7, Unknown_8, Unknown_9, Unknown_10, Unknown_11, Unknown_12, Unknown_13, Unknown_14, Unknown_15, Unknown_16, Unknown_17, Unknown_18, Unknown_19, Unknown_20, Unknown_21, Unknown_22, Unknown_23, Unknown_24, Unknown_25, Unknown_26, Unknown_27, Unknown_28, Unknown_29, Unknown_30, Unknown_31, Unknown_32, Unknown_33, Unknown_34, Unknown_35, Unknown_36, Unknown_37, Unknown_38, Unknown_39, Unknown_40, Unknown_41, Unknown_42, Unknown_43, Unknown_44, Unknown_45, Unknown_46, Unknown_47, Unknown_48, Unknown_49, Unknown_50, Unknown_51
- Contains 13814 objects with class IDs not defined in data.yaml
**Additional preprocessing required?** YES
