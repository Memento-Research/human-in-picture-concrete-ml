import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image


def get_loaded_dataset(humans_path: str, not_humans_path: str, size: int):
    humans_files, humans_quantity = load_dir(humans_path)
    not_humans_files, not_humans_quantity = load_dir(not_humans_path)
    print('Humans files: ' + str(len(humans_files)))
    print('Not humans files: ' + str(len(not_humans_files)))

    total_quantity = humans_quantity + not_humans_quantity

    image_dimensions = (size, size, 1)
    transposed_image_dimensions = (1, size, size)

    # Create an empty array to store the images
    # training_set = np.zeros((total_quantity, *transposed_image_dimensions), dtype='float32')
    training_set = np.zeros((200, *transposed_image_dimensions), dtype='float32')

    # TODO Refactor this @juarce
    # Load and preprocess images
    for i in range(200):  # range(total_quantity):
        if i < 100:  # not_humans_quantity:
            path = not_humans_files[i]
        else:
            path = humans_files[i - len(not_humans_files)]
        print(path)

        img = Image.open(path).resize(image_dimensions[:2]).convert('L')
        img_array = np.array(img)
        training_set[i] = img_array

    # Create labels (assuming 0 for non-humans and 1 for humans)
    # labels = np.concatenate((np.zeros(int(not_humans_quantity)), np.ones(int(humans_quantity))))
    labels = np.concatenate((np.zeros(100), np.ones(100)))

    return train_test_split(training_set, labels, test_size=0.2, random_state=42)


def load_dir(dir_path: str):
    dir_files = []
    for path in os.listdir(dir_path):
        if '.jpg' in path:
            dir_files.append(os.path.join(dir_path, path))
    return dir_files, len(dir_files)
