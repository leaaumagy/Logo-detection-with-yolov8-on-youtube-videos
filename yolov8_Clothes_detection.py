## Logo detection 

# Trained yolov8 model
!git clone https://github.com/leaaumagy/Logo-detection-with-yolov8-on-youtube-videos.git

# Copy files to root
!cp -r /content/Logo-detection-with-yolov8-on-youtube-videos/Yolov8/* /content

!pip3 install pytube ultralytics

# Packages
import pandas as pd
import cv2
import os
import subprocess
from ultralytics import YOLO
from IPython.display import clear_output
from pytube import YouTube

# Clear the output in Jupyter Notebook
clear_output()

# Votre commande YOLO
!yolo checks

# List of selected folders
selected_folders = ['Adidas_SB', 'American_Eagle', 'Armani', 'Balenciaga',
                    'Bulgari', 'Ck_Calvin_Klein_1',
                    'Ck_Calvin_Klein_2', 'Canada Goose', 'Carhartt', 'Casio', 'Celine',
                    'Champion', 'chanel', 'Chloe', 'Columbia', 'Converse',
                    'Ellesse', 'Everlast', 'Gap', 'Giorgio', 'Guess',
                    'Hugo_Boss', 'Karl_Lagerfield', 'Kenzo', 'lacoste', 'levis',
                    'louis_vuitton_1', 'louis_vuitton_2', 'napapijri',
                    'new_balance_1', 'new_balance_2', 'obey', 'oxxford', 'patagonia', 'pepe_jeans',
                    'polo_ralph_lauren', 'prada', 'rolex', 'sergio_tacchini',
                    'the_timberland', 'the_north_face', 'timberland', 'tommy_hilfiger', 'tom_ford',
                    'uniqlo', 'versace', 'zara']

# Function to convert processed videos to mp4 format
def convert_avi_to_mp4(input_video_path, output_video_path):
    input_video = cv2.VideoCapture(input_video_path)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = input_video.read()
        if not ret:
            break
        output_video.write(frame)

    input_video.release()
    output_video.release()

# Function to detect folders starting with 'predict' where processed videos are saved
def process_detect_folders(detect_path):
    for folder_name in os.listdir(detect_path):
        folder_path = os.path.join(detect_path, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith("predict"):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if os.path.isfile(file_path) and file_name.lower().endswith(".avi"):
                    # Create an output path with the same filename but with the .mp4 extension
                    mp4_path = os.path.join(folder_path, file_name.lower().replace(".avi", ".mp4"))

                    # Convert AVI video to MP4
                    convert_avi_to_mp4(file_path, mp4_path)

                    # Remove the AVI file after conversion
                    os.remove(file_path)

# Create List of your videos
video_urls = []

# Keep asking the user for video URLs until they provide an empty input
while True:
    video_url = input("Enter a YouTube video URL (or press Enter to finish): ").strip()
    
    if not video_url:
        if not video_urls:
            print("You haven't entered any video URLs. Exiting the code.")
            break
        else:
            break

    # Check if the URL is valid
    try:
        yt = YouTube(video_url)
        print(f"Video with URL '{video_url}' exists.")
        video_urls.append(video_url)
    except Exception as e:
        print(f"Error: {str(e)} Please enter a valid YouTube video URL.")


# Folder where original videos are saved
save_path = '/content/video_youtube'

# Folder to save videos generated by YOLO
yolo_save_path = '/content/runs/detect/predict'

# Specify the path to the /content/runs/detect folder where folders starting with 'predict' are located
detect_folder_path = "/content/runs/detect"

# Extract video IDs from URLs
video_ids = [url.split("/")[-1].split("?")[0] for url in video_urls]

# Create a DataFrame
df = pd.DataFrame(index=selected_folders, columns=video_ids)

# Loop for each video
for video_url in video_urls:
    # Download the video
    yt = YouTube(video_url)
    video_stream = yt.streams.get_highest_resolution()
    video_path = video_stream.download(save_path)

    # Video ID associated with the video
    video_id = video_url.split("/")[-1].split("?")[0]

    # Rename the video file with the ID
    new_video_name = f'{video_id}.mp4'
    new_video_path = os.path.join(save_path, new_video_name)
    os.rename(video_path, new_video_path)

    # YOLO command
    yolo_command = f"yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt source={new_video_path}"
    result_yolo = subprocess.run(yolo_command, shell=True, capture_output=True, text=True)

    # Call the function that looks for and converts the processed video to mp4 format
    process_detect_folders(detect_folder_path)

    # Analyze the content of result_yolo.stdout for each element of each row in the DataFrame
    for folder in selected_folders:
        if folder in result_yolo.stdout:
            df.loc[folder, video_id] = 'Yes'
        else:
            df.loc[folder, video_id] = 'No'

# Save df
df.to_csv('/content/DataFrame_Logo_detection.csv', index=False)
df
