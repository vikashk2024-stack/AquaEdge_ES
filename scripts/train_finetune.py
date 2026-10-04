from ultralytics import YOLO
import os

def main():
    print("Loading the previously trained model...")
    # Load the best model from your 50-epoch run
    model = YOLO("runs/detect/runs/aquaedge_newdata/weights/best.pt")
    
    print("Resuming training for 20 more epochs to boost confidence scores...")
    
    # Fine-tune the model for an additional 20 epochs
    try:
        model.train(
            data="datasets/plankton/new-data-set/data.yaml",
            epochs=20,
            imgsz=640,
            batch=8,
            workers=0,
            project="runs/detect/runs",
            name="aquaedge_newdata_finetuned",
            exist_ok=True,
            resume=False # Start a new fine-tuning run from the best weights
        )
        print("Fine-tuning complete! Your new, higher-confidence model is in runs/detect/runs/aquaedge_newdata_finetuned/weights/best.pt")
    except Exception as e:
        print(f"Training error: {e}")
        print("Attempting to fallback to lower batch size (batch=4) to save memory...")
        model.train(
            data="datasets/plankton/new-data-set/data.yaml",
            epochs=20,
            imgsz=640,
            batch=4,
            workers=0,
            project="runs/detect/runs",
            name="aquaedge_newdata_finetuned_b4",
            exist_ok=True
        )

if __name__ == "__main__":
    main()
