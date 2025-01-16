import os
import kagglehub
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait

def load_celeba(transform):
    # Download CelebA
    celeba_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    # Load attributes
    attr_df = pd.read_csv(os.path.join(celeba_path, "list_attr_celeba.csv"))
    attr_df.replace(-1, 0, inplace=True)
    image_path = os.path.join(celeba_path, "img_align_celeba", "img_align_celeba")
    attr_df["image_path"] = attr_df["image_id"].apply(lambda x: os.path.join(image_path, x))
    # Loads and transforms one image
    def read_and_transform_image(path):
        img = Image.open(path).convert("RGB")
        img = transform(img)
        return img
    # Prepares location for images to be stored
    num_images = len(attr_df["image_path"])
    assert num_images > 0, "No images to load"
    sample_img = read_and_transform_image(attr_df["image_path"][0])
    result_array = np.empty((num_images, *sample_img.shape))
    # Loads image into result_array
    def load_one_img(i, path):
        result_array[i] = read_and_transform_image(path)
        del img
    # Load images in parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        with tqdm(total=num_images, desc="Loading Images...") as pbar:
            loading_jobs = enumerate(attr_df["image_path"])
            futures = []
            for job in loading_jobs:
                future = executor.submit(load_one_img, *job)
                future.add_done_callback(lambda _: pbar.update())
                futures.append(future)
            wait(futures)
    attr_df.drop(["image_id", "image_path"], axis=1, inplace=True)
    # Return the result
    return result_array, attr_df