# FSD Capstone Project

Fully self driving car prototype that combines three models to simulate autonomous driving on dashcam video.

## What it does

Takes a driving video as input and shows three windows side by side:
- Original frame
- Segmented frame with lane detection and car detection overlays
- A steering wheel that rotates based on predicted angle

## Models used

- **Steering angle prediction**: NVIDIA PilotNet architecture (2016 paper), a CNN that takes a road image and regresses the steering angle
- **Lane segmentation**: YOLOv11m trained on Roboflow L-S-1 dataset
- **Object detection**: YOLOv11s pretrained on COCO for detecting cars, pedestrians etc

## Dataset

Steering model is trained on the Sully Chen driving dataset (45,406 images with steering angles).
Download: https://github.com/SullyChen/driving-datasets

Place it at `data/driving_dataset/` with `data.txt` and all the jpg images inside.

## Setup

```bash
conda create -n fsd python=3.9
conda activate fsd
pip install -r requirements.txt
```

## Folder structure

```
FSD Capstone Project/
  data/
    driving_dataset/          # 45k images + data.txt (not in git)
    steering_wheel_image.jpg
  model_training/
    train_lane_detection/     # yolo training notebook
    train_steering_angle/     # nvidia pilotnet training
      driving_data.py
      model.py
      train.py
  saved_models/
    regression_model/         # trained steering model checkpoint
    lane_segmentation_model/  # trained yolo lane model
    object_detection_model/   # pretrained yolo coco model
  src/
    inference/
      run_fsd_inference.py    # main demo script
      run_steering_angle_prediction.py
      run_segmentation_obj_det.py
    models/
      model.py                # nvidia architecture
```

## Training

Training is done on Google Colab with T4 GPU. See the training notebook in `model_training/`.

## Running the demo

```bash
python src/inference/run_fsd_inference.py
```
