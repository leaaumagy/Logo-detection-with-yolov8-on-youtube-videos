# YOLOv8 Logo Detection

This project utilizes YOLOv8 for logo detection on YouTube videos. Follow the steps below to effectively use the project.

## Project Structure

- **Yolov8/**
  - **Dataset/**
    - Contains training data, split into train, test, and valid folders.
    - An associated YAML file for the images.
  - **runs/**
    - The folder containing the trained model: `/train/weights/best.pt`.
    - Results from model evaluation.

- **yolov8_Clothes.ipynb**
  - A Jupyter notebook demonstrating how the model is constructed.
  - Applies logo detection on YouTube videos.
  - Results are displayed in a dataframe.

- **yolo_model.py**
  - A Python script to customize the code, such as modifying brands or model parameters.

- **Example_detection.py**
  - A Python script with an example application of detection.

- **yolov8_Clothes_detection.py**
  - A Python script to apply detection on user-selected YouTube videos.

## Model training results
![Model training results](https://github.com/leaaumagy/Logo-detection-with-yolov8-on-youtube-videos/blob/main/Yolov8/runs/detect/train/results.png)
## Important Notes

- A GPU is required to generate the base YOLOv8 model.
- A Kaggle account with a username and API key is necessary.
