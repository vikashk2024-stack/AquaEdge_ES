import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("ROBOFLOW_API_KEY not found in .env")
    exit(1)

print("Authenticating with Roboflow...")
rf = Roboflow(api_key=api_key)
project = rf.workspace("aqua-epmvm").project("plankton-analysis-iqktk")

# Finding the latest version to download
version = project.version(1) # Try version 1 first
print("Downloading dataset...")
dataset = version.download("yolov8", location="datasets/raw/plankton_dataset")
print("Download complete!")
