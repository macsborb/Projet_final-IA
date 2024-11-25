import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

def clean_and_verify_images(image_dir):
    """Verify images and clean metadata if needed."""
    verified_images = []
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        try:
            with Image.open(filepath) as img:
                img.verify()  # Check image integrity
            verified_images.append(filepath)
        except (IOError, SyntaxError):
            print(f"Corrupted image detected and skipped: {filepath}")
    return verified_images

def resize_images(image_paths, output_dir, target_size=(224, 224)):
    """Resize images to uniform dimensions."""
    os.makedirs(output_dir, exist_ok=True)
    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            resized_img = cv2.resize(img, target_size)
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, resized_img)
        except Exception as e:
            print(f"Error resizing image {image_path}: {e}")

def split_data(image_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split images into training, validation, and test sets."""
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    train_val, test = train_test_split(image_paths, test_size=1 - train_ratio - val_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

    for split, split_name in zip([train, val, test], ["train", "val", "test"]):
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for file in split:
            shutil.copy(file, split_dir)

if __name__ == "__main__":
    # Paths and parameters
    raw_image_dir = "./n02111889-Samoyed"
    clean_image_dir = "clean_images"
    resized_image_dir = "resized_images"
    dataset_dir = "dataset"
    log_file = "preprocessing_log.txt"


    # Step 1: Clean and verify images
    print("Step 1: Clean and verify images.")
    verified_images = clean_and_verify_images(raw_image_dir)

    # Step 2: Resize images
    print("Step 2: Resize images to {224}x{224}.")
    resize_images(verified_images, resized_image_dir)

    # Step 3: Split data into train/val/test
    print("Step 3: Split data into training (70%), validation (15%), and test (15%) sets.")
    split_data(resized_image_dir, dataset_dir)

    print("Image preprocessing completed. Check the log file for details.")
