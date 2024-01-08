import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from IPython import display
from ultralytics import YOLO
import opendatasets as od
import xml.etree.ElementTree as ET
from pytube import YouTube
import cv2
import requests
import subprocess
from glob import glob
import random
import yaml

## Download training images
# you must have a kaggle account and generate your api in the settings
od.download("https://www.kaggle.com/datasets/lyly99/logodet3k")

## Copy the 'Clothes' folder to a new 'Dataset' folder
def copy_and_rename(folder_path, destination_path):
    # Copy the "Clothes" folder to the new "Dataset" folder
    shutil.copytree(folder_path, os.path.join(destination_path))

# Path of the 'Clothes' folder
folder_path = '/content/logodet3k/LogoDet-3K/Clothes'

# Path of the destination folder
destination_path = '/content/Dataset/Clothes'

# Call the function to copy
copy_and_rename(folder_path, destination_path)

## Rename brand folder names
def rename_folders(directory_path):
    # List all items in the directory
    items = os.listdir(directory_path)

    for item in items:
        item_path = os.path.join(directory_path, item)

        # Check if the item is a folder
        if os.path.isdir(item_path):
            # Rename the folder by replacing spaces with underscores
            new_name = item.replace(" ", "_")
            new_name = new_name.replace("76", "A_76")
            new_name = new_name.replace("-", "_")
            new_name = new_name.replace("'", "_")
            new_path = os.path.join(directory_path, new_name)
            os.rename(item_path, new_path)

# Directory to be processed
directory_to_process = '/content/Dataset/Clothes'
rename_folders(directory_to_process)


## Filter on selected brands

# Path to the "Clothes" folder
clothes_path = "/content/Dataset/Clothes"

# List of selected folders
# you can adapt this list according to your needs
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

# Get the list of all files and folders in the "Clothes" folder
all_items = os.listdir(clothes_path)

# Filter folders using the selected list
filtered_folders = [folder for folder in all_items if os.path.isdir(os.path.join(clothes_path, folder)) and folder.lower() in [sf.lower() for sf in selected_folders]]

# Display filtered folders
print("Filtered folders:")
print(filtered_folders)

# Iterate through all folders in the "Clothes" folder
for folder in all_items:
    folder_path = os.path.join(clothes_path, folder)

    # Check if the folder is not in the list of selected folders
    if folder.lower() not in [sf.lower() for sf in selected_folders] and os.path.isdir(folder_path):
        # Remove the non-selected folder and its contents
        shutil.rmtree(folder_path)

## Rename images and xml file names to avoid duplicates
def rename_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        # Exclude non-numeric subdirectories
        dirs = [d for d in dirs if d.isdigit()]

        for file in files:
            if file.endswith(".jpg") or file.endswith(".xml"):
                # Get the parent folder number
                parent_folder_name = os.path.basename(root)

                # Build the new file name with the parent folder number
                new_name = f"{os.path.splitext(file)[0]}_{parent_folder_name}{os.path.splitext(file)[1]}"

                # Current file path
                current_path = os.path.join(root, file)

                # Build the new file path
                new_path = os.path.join(root, new_name)

                # Rename the file
                os.rename(current_path, new_path)

# Path of the 'Clothes' folder
folder_path = '/content/Dataset/Clothes'

# Call the function to rename files
rename_files(folder_path)


## Balancing the number of instances between brands
def count_files(directory, extension):
    files = glob(os.path.join(directory, f'*.{extension}'))
    number_of_files = len(files)
    return number_of_files

def duplicate_files(folder_path, target_count_per_folder):
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        if os.path.isdir(subfolder_path):
            # Calculate the number of images to duplicate to reach at least target_count_per_folder
            images_to_duplicate = max(0, target_count_per_folder - count_files(subfolder_path, 'jpg'))

            # Duplicate files if necessary
            for i in range(images_to_duplicate):
                # New names for duplicated files
                new_image_number = images_to_duplicate + i + 1
                new_image_name = f"{new_image_number}_{subfolder}.jpg"
                new_xml_name = f"{new_image_number}_{subfolder}.xml"
                new_image_path = os.path.join(subfolder_path, new_image_name)
                new_xml_path = os.path.join(subfolder_path, new_xml_name)

                # Use a random image and XML file for duplication
                source_image_path = random.choice(glob(os.path.join(subfolder_path, '*.jpg')))
                source_xml_path = source_image_path.replace('.jpg', '.xml')

                # Check if the source file is different from the destination file
                if source_image_path != new_image_path:
                    shutil.copy(source_image_path, new_image_path)
                if source_xml_path != new_xml_path:
                    shutil.copy(source_xml_path, new_xml_path)

folder_path = "/content/Dataset/Clothes"
target_count_per_folder = 150
duplicate_files(folder_path, target_count_per_folder)


## Creation of the Dataset (train, test, valid) and the YALM file
# Path to the main folder
main_folder = '/content/Dataset/Clothes'

# Create the folder structure for the Dataset
dataset_folder = '/content/Dataset'
folders = ['train', 'test', 'valid']
subfolders = ['images', 'labels']

# Create the folder structure
for folder in folders:
    for subfolder in subfolders:
        path = os.path.join(dataset_folder, folder, subfolder)
        os.makedirs(path, exist_ok=True)

# Get the list of all brand folders in your Clothes folder
brand_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
num_classes = len(brand_folders)

# Initialize variables outside the loop
all_train_files = []
all_test_files = []
all_valid_files = []

# Split data into train, test, and validation sets for each brand
for brand_folder in brand_folders:
    brand_path = os.path.join(main_folder, brand_folder)
    image_files = [f for f in os.listdir(brand_path) if f.endswith('.jpg')]

    # Ensure there are enough samples for the split
    if len(image_files) < 5:
        print(f"Assigning all samples from {brand_folder} to the training set due to insufficient samples.")
        train_files = image_files
        test_files = []
        valid_files = []
    else:
        # Split into train and test_valid_files
        train_files, test_valid_files = train_test_split(image_files, test_size=0.2, random_state=43)

        # Check if test_valid_files has enough samples
        if len(test_valid_files) < 2:
            train_files.extend(test_valid_files)
            test_files = []
            valid_files = []
        else:
            # Split test_valid_files into test and validation
            test_files, valid_files = train_test_split(test_valid_files, test_size=0.5, random_state=43)

    # Add files from each brand to the overall list
    all_train_files.extend(train_files)
    all_test_files.extend(test_files)
    all_valid_files.extend(valid_files)

    # Move files to the corresponding folders
    def move_files(source_folder, destination_folder, file_list):
        for file in file_list:
            image_path = os.path.join(brand_path, file)
            xml_path = os.path.join(brand_path, os.path.splitext(file)[0] + '.xml')

            dest_image_path = os.path.join(dataset_folder, destination_folder, 'images', file)
            dest_xml_path = os.path.join(dataset_folder, destination_folder, 'labels', os.path.splitext(file)[0] + '.xml')

            shutil.copy(image_path, dest_image_path)
            shutil.copy(xml_path, dest_xml_path)

    # Move files to the corresponding folders
    move_files(brand_folder, 'train', train_files)
    move_files(brand_folder, 'test', test_files)
    move_files(brand_folder, 'valid', valid_files)

# Create a YAML file for YOLO
yolo_yaml_content = "names:\n"

for idx, brand_name in enumerate(brand_folders):
    yolo_yaml_content += f"  {idx}: {brand_name}\n"

yolo_yaml_content += f"""
train: {os.path.join(dataset_folder, 'train/images')}
val: {os.path.join(dataset_folder, 'valid/images')}
test: {os.path.join(dataset_folder, 'test/images')}

nc: {num_classes}
"""

with open('/content/Dataset/data.yaml', 'w') as yaml_file:
    yaml_file.write(yolo_yaml_content)


## Converting XML files to TXT
# Load the YAML file with names and associated IDs
with open('/content/Dataset/data.yaml', 'r') as yaml_file:
    yaml_content = yaml.load(yaml_file, Loader=yaml.FullLoader)

class_ids = {v: k for k, v in yaml_content['names'].items()}

def convert_xml_to_yolov8(xml_path, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find(".//size/width").text)
    image_height = int(root.find(".//size/height").text)

    with open(output_path, 'w') as output_file:
        for obj in root.findall(".//object"):
            raw_class_name = obj.find("name").text
            # Replace spaces with underscores
            class_name = raw_class_name.replace(" ", "_")
            class_name = class_name.replace("-", "_")
            class_name = class_name.replace("'", "_")
            class_name = class_name.replace("76", "A_76")
            class_name = class_name.replace("snow_peak_2", "timbuk2_1")
            class_name = class_name.replace("Alcatel_2", "alpinestars_2")
            class_name = class_name.replace("maybach_1", "meyba_1")
            class_name = class_name.replace("INOHERB_2", "Isabel_Maran_2")
            class_name = class_name.replace("Harvest_Moon", "Heat_1")
            class_name = class_name.replace("Alcatel_2", "alpinestars_2")

            # Use the get method with a default value of -1
            class_id = class_ids.get(class_name, -1)

            # Check if the class was found
            if class_id == -1:
                print(f"Class not found for {class_name} in the file {xml_path}")
                continue

            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            center_x = (xmin + xmax) / (2.0 * image_width)
            center_y = (ymin + ymax) / (2.0 * image_height)
            bbox_width = (xmax - xmin) / image_width
            bbox_height = (ymax - ymin) / image_height

            output_line = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n"
            output_file.write(output_line)

def process_xml_folder(xml_folder):
    yolov8_output_folder = xml_folder  # Same output folder as the input folder

    # Ensure the output folder exists
    os.makedirs(yolov8_output_folder, exist_ok=True)

    # Conversion for each XML file
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            yolov8_output_path = os.path.join(yolov8_output_folder, xml_file.replace(".xml", ".txt"))
            convert_xml_to_yolov8(xml_path, yolov8_output_path)

            # Remove the XML file after conversion
            os.remove(xml_path)

# Folders to process
input_folders = [
    '/content/Dataset/train/labels',
    '/content/Dataset/valid/labels',
    '/content/Dataset/test/labels'
]

# Process each XML folder
for xml_folder in input_folders:
    process_xml_folder(xml_folder)


## Loading the YOLO version 8 model
# Clear the output in Jupyter Notebook
display.clear_output()

# Your YOLO command
!yolo checks

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Training the yolov8 model with our images
# You can adapt the number of epoch, but it is necessary to analyse the metrics associate
!yolo task=detect mode=train model=/content/yolov8m.pt data=/content/Dataset/data.yaml epochs = 30 imgsz = 640

# Model validation
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data=/content/Dataset/data.yaml

# Now you need to obtain all the necessary files to detect clothing brand logos on images, videos, etc.
# It's your turn to play
