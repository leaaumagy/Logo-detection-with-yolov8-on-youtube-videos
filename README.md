# YOLOv8 Logo Detection 🚀

This project uses YOLOv8 for clothing brand logo detection on YouTube videos. 🛍️👚👗👠

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

- **yolov8_model.py**
  - A Python script to customize the code, such as modifying brands or model parameters.

- **yolov8_example_detection.py**
  - A Python script with an example application of detection.

- **yolov8_Clothes_detection.py**
  - A Python script to apply detection on user-selected YouTube videos.

## Model training results
![Model training results](https://github.com/leaaumagy/Logo-detection-with-yolov8-on-youtube-videos/blob/main/Yolov8/runs/detect/train/results.png)

→ **Precision:** The model correctly predicts nearly 92% of objects on average.

→ **Recall:** The model detects on average nearly 92% of the real objects present in the images.

→ **mAP50:** The model has an accuracy of 94.18% on average, which is excellent, on predictions with an IoU threshold of 0.5.

→ **mAP50-95:** The model has an accuracy of 79.83% on average, which indicates good performance, on predictions with a wider interval of IoU thresholds (between 0.5 and 0.95).


The results of model validation on the /yolov8_Clothes.ipynb file indicate performance on a set of validation images.

- Total number of validation images: 648
- Total number of object instances detected: 801
- Performance of all classes:
  - Accuracy: 97.2%
  - Recall: 95.4%
  - mAP50: 97.7%
  - mAP50-95: 89.9%

## Important Notes

- A GPU is required to generate the base YOLOv8 model.
- A Kaggle account with a username and API key is necessary, it is free.


## Source

- YOLOv8 Model: [Ultralytics YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
- YOLOv8 Documentation: [YOLOv8 Documentation](https://docs.ultralytics.com/)
- Training Data for Logo Detection: [LogoDet3K Kaggle Dataset](https://www.kaggle.com/datasets/lyly99/logodet3k)

