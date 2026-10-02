# Dataset Selection Report

## Candidate Dataset 1

* **Dataset Name:** Plankton Analysis
* **URL:** [https://universe.roboflow.com/aqua-epmvm/plankton-analysis-iqktk](https://universe.roboflow.com/aqua-epmvm/plankton-analysis-iqktk)
* **Number of Images:** 2,332
* **Number of Classes:** 52
* **Annotation Format:** Bounding Box / YOLO compatible
* **Image Resolution:** Various (Microscopy standard)
* **Microscopy Information:** Yes, microscopy images of aquatic samples.
* **Freshwater/Marine Context:** Mixed aquatic environments.
* **License:** CC BY 4.0
* **Dataset Size:** Estimated < 500 MB (well under the 5 GB limit).

### Target Class Availability
* **Chlorella:** Present
* **Scenedesmus:** Present
* **Navicula:** Present
* **Microcystis:** Present
* **Euglena:** Present

*Note: Exact object counts per class are pending download as the detailed class distribution requires API access/download to view. However, search results confirm all 5 target classes are explicitly annotated.*

## Recommendation

I recommend using the **Plankton Analysis** dataset as our foundational dataset. It contains object-level bounding box annotations for all five of our target microorganism classes. Because it has 2,332 images overall, it should provide a manageable but sufficient starting point for our 5-class subset without exceeding our size limits. We will filter out the other 47 classes during the data processing phase.
