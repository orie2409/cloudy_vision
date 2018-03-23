import json
import numpy as np
import os
import shutil


def extract_and_relabel_images(input_path_root, output_path, count=1):
    """
    Randomly selects 'count' number of images from each folder in the CalTech 101
    dataset and saves to output_path after renaming.

    :param input_path_root: str
    :param output_path: str
    :param count: int
    :return: None
    """
    # First, clear any previous contents from output_path folder
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        os.unlink(file_path)

    # Now process images from Caltech 101
    folder_list = os.listdir(input_path_root)

    for folder in folder_list:
        # List of all images contained in one image type folder
        input_path = os.path.join(input_path_root, folder)
        image_files = os.listdir(input_path)
        # Select at random a number of the images defined by count
        selected_files = np.random.choice(image_files, size=count)
        for file in selected_files:
            # Rename it as the image type given by name of folder and copy to new location
            in_file = os.path.join(input_path, file)
            out_file = os.path.join(output_path, folder + '_' + file[:-4] + '.jpg')
            shutil.copy(in_file, out_file)


def create_image_tags(image_folder, tags_filepath):
    """
    Extracts the correct tag for each image from it's filename and stores in a dictionary.
    :param image_folder: str
    :param tags_filepath: str
    :return: dict
    """
    # Create dict of image labels
    image_list = os.listdir(image_folder)
    tags_dict = {}
    for image in image_list:
        tag = image[:-15]
        tags_dict[image] = [tag]

    # Write to location for Cloudy Vision
    with open(tags_filepath, 'w') as json_file:
        # Delete any existing data and write tags to file
        json_file.truncate()
        json.dump(tags_dict, json_file, indent=4, separators=(',', ': '))
    return tags_dict