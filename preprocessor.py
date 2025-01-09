import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define directories
DATA_DIR = "dataset"
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# Function to load and preprocess data
def load_data(data_dir, img_size=224):
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, (img_size, img_size))
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image: {e}")

    # Normalize and reshape
    data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
    labels = to_categorical(np.array(labels), num_classes=2)

    return data, labels

# Load the dataset
X, y = load_data(DATA_DIR)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
