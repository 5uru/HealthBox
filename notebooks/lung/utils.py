import random
import numpy as np
import cv2


def preprocess_chest_xray(image):
    # Convert image to NumPy array if it's not already
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Check if image is already grayscale
    if len(image.shape) == 2:
        # Image is already grayscale
        gray_image = image
    else:
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize to [0,1]
    image = gray_image.astype('float32')
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

    # Scale to [0,255] for image processing
    image_uint8 = (image * 255).astype('uint8')

    # --- Noise Reduction ---
    denoised = cv2.GaussianBlur(image_uint8, (3, 3), 0)
    denoised = cv2.medianBlur(denoised, 3)

    # --- CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=0.03, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # --- Gamma Correction ---
    gamma = 0.7
    contrast_float = contrast_enhanced.astype('float32') / 255.0
    gamma_corrected = np.power(contrast_float, gamma)

    # Convert to 8-bit
    final_image = (gamma_corrected * 255).astype(np.uint8)

    # Convert to RGB
    final_image_rgb = np.stack((final_image,) * 3, axis=-1)

    # Resize to EfficientNet default size (224x224)
    final_image_rgb = cv2.resize(final_image_rgb, (224, 224))

    return final_image_rgb


def  data_loader(dataset, labels : dict):
    """
    Function to load data from a dataset and return images and labels.

    Args:
        dataset: The dataset to load data from.
        labels: A dictionary mapping label names to indices.

    Returns:
        images: A list of images.
        labels: A list of labels corresponding to the images.
    """
    images = []
    labels_list = []

    for item in dataset:
        image = preprocess_chest_xray(item["image"]).transpose(2, 0, 1)  # Change to (C, H, W)

        images.append(image)

        # Convert label to index
        label_index = labels[item["label"]]
        labels_list.append(label_index)

    # Shuffle the data to randomize it
    _data = list(zip(images, labels_list))
    random.shuffle(_data)

    shuffled_images, shuffled_labels = zip(*_data)

    # Convert to numpy arrays
    shuffled_images = np.array(shuffled_images, dtype=np.float32)
    shuffled_labels = np.array(shuffled_labels)

    return shuffled_images, shuffled_labels