import os
import kagglehub
import PIL.Image

import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CelebADataset(Dataset):
    """
    A dataset class for the CelebA dataset.
    """
    kagglehub_path = "jessicali9530/celeba-dataset"
    
    def __init__(self, *, 
                target_columns: list[str] | None=None, 
                transform=None):
        celeba_path = kagglehub.dataset_download(self.kagglehub_path)
        self._attr_df = CelebADataset._load_attr_df(celeba_path)
        self.target_columns = target_columns
        self.transform = transform
        
    @staticmethod
    def _load_attr_df(path):
        attr_df = pd.read_csv(os.path.join(path, "list_attr_celeba.csv"))
        attr_df.replace(-1, 0, inplace=True)
        image_path = os.path.join(path, "img_align_celeba", "img_align_celeba")
        attr_df["image_path"] = attr_df["image_id"].apply(lambda x: os.path.join(image_path, x))
        return attr_df
    
    def __len__(self):
        return len(self._attr_df)
    
    def __getitem__(self, index):
        assert index < len(self), "Index out of bounds"
        entry = self._attr_df.iloc[index]
        img = PIL.Image.open(entry["image_path"])
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_columns is None:
            targets = []
        else:
            targets = [None] * len(self.target_columns)
            for i, column in enumerate(self.target_columns):
                targets[i] = entry[column]
        return img, targets