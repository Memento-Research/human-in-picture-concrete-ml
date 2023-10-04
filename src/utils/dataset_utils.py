import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# Load the dataset from the given paths
def get_loaded_dataset(humans_path: str, not_humans_path: str, image_size: int, max_images_to_load: int = None):
    humans_files, humans_quantity = load_dir(humans_path)
    not_humans_files, not_humans_quantity = load_dir(not_humans_path)

    humans_to_load, not_humans_to_load = get_quantity_to_load(humans_quantity, not_humans_quantity, max_images_to_load)

    humans_images = process_images(humans_files, image_size, humans_to_load)
    not_humans_images = process_images(not_humans_files, image_size, not_humans_to_load)

    images = np.concatenate((humans_images, not_humans_images))
    labels = np.concatenate((np.zeros(humans_to_load), np.ones(not_humans_to_load)))

    return train_test_split(images, labels, test_size=0.2, random_state=42)


def load_dir(dir_path: str):
    dir_files = []
    for path in os.listdir(dir_path):
        if '.jpg' in path:
            dir_files.append(os.path.join(dir_path, path))

    length = len(dir_files)
    print(f'Loaded {length} files from {dir_path}')

    return dir_files, length


# Return the quantity of images to load for each class based on the max_images_to_load parameter
# If max_images_to_load is None, return the original quantities
# If max_images_to_load is not None, return the minimum between the original quantities and max_images_to_load
def get_quantity_to_load(humans_quantity: int, not_humans_quantity: int, max_images_to_load: int = None):
    if max_images_to_load is None:
        return humans_quantity, not_humans_quantity

    humans_to_load = min(humans_quantity, max_images_to_load)
    not_humans_to_load = min(not_humans_quantity, max_images_to_load)
    return humans_to_load, not_humans_to_load


def process_images(images_paths, image_size, images_to_load):
    transposed_image_dimensions = (1, image_size, image_size)

    images = np.zeros((images_to_load, *transposed_image_dimensions), dtype='float32')
    for i in tqdm(range(images_to_load), desc=f"Processing images"):
        images[i] = process_image(images_paths[i], image_size)

    return images


def process_image(image_path, image_size):
    img = Image.open(image_path).resize((image_size, image_size)).convert('L')
    img_array = np.array(img)
    return img_array


# Create a DataLoader for training
def create_dataloader(x, y, batch_size):
    x_tensor = torch.Tensor(x)  # .to(get_device())
    y_tensor = torch.Tensor(y)  # .to(get_device())
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size)


def save_image(image, filename):
    image = Image.fromarray(image.squeeze().astype("uint8"), mode='L')
    image.save(filename)
