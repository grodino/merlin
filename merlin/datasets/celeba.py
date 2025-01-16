import os
import kagglehub
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, wait

from numpy.typing import ArrayLike

def load_celeba(transform, subset: None | int | ArrayLike=None):
    """
    Loads the CelebA dataset, applies transformations to the images, and returns the images along with their attributes.

    Parameters:
    -----------
    transform : callable
        A function or transform to apply to each image.
    subset : None, int, or ArrayLike, optional
        If None, loads the entire dataset. If an integer, loads the first `n` images. If an array-like, loads the images at the specified indices.

    Returns: (result_array, attr_df)
    --------
    result_array : numpy.ndarray
        An array of transformed images.
    attr_df : pandas.DataFrame
        A DataFrame containing the attributes of the images.
    """
    # Download CelebA
    celeba_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    # Load attributes
    attr_df = pd.read_csv(os.path.join(celeba_path, "list_attr_celeba.csv"))
    match subset:
        case None:
            pass
        case int(n):
            attr_df = attr_df[:n]
        case _:
            attr_df = attr_df.iloc[subset]
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
    sample_img = np.array(read_and_transform_image(attr_df["image_path"][0]))
    result_array = np.empty((num_images, *sample_img.shape), dtype=sample_img.dtype)
    # Loads image into result_array
    def load_one_img(i, path):
        result_array[i] = read_and_transform_image(path)
        del img
    # Load images in parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        with tqdm(total=num_images, desc="Loading Images") as pbar:
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