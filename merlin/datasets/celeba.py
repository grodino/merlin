import os
import kagglehub
import PIL.Image

import pandas as pd

from torch.utils.data import Dataset


class CelebADataset(Dataset):
    """
    A dataset class for the CelebA dataset.
    """

    kagglehub_path = "jessicali9530/celeba-dataset"

    # Define image ranges for each split
    SPLIT_RANGES = {
        "train": (1, 162770),
        "val": (162771, 182637),
        "test": (182638, 202599),
    }

    TRAINING_TARGETS = [
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        # "Male", This is used as the sensitive attribute...
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        # "Receding_Hairline", optimizer issue in LinearRelaxation
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        # "Wearing_Hat", optimizer issue in LinearRelaxation
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ]

    def __init__(
        self,
        *,
        split: str = "train",  # Specify the partition: 'train', 'val', or 'test'
        target_columns: list[str] | None = None,
        transform=None,
    ):
        """
        Initialize the dataset.

        Args:
            split (str): The partition of the dataset ('train', 'val', 'test').
            target_columns (list[str] | None): The target columns for labels.
            transform: Optional transformation to be applied on the images.
        """
        assert (
            split in self.SPLIT_RANGES
        ), f"Invalid split: {split}. Must be one of {list(self.SPLIT_RANGES.keys())}"

        celeba_path = kagglehub.dataset_download(self.kagglehub_path)
        self._attr_df = CelebADataset._load_attr_df(celeba_path)
        self._attr_df = CelebADataset._filter_split(self._attr_df, split)

        self.target_columns = target_columns
        self.transform = transform

    @staticmethod
    def _load_attr_df(path):
        attr_df = pd.read_csv(os.path.join(path, "list_attr_celeba.csv"))
        attr_df.replace(-1, 0, inplace=True)
        image_path = os.path.join(path, "img_align_celeba", "img_align_celeba")
        attr_df["image_path"] = attr_df["image_id"].apply(
            lambda x: os.path.join(image_path, x)
        )
        return attr_df

    @staticmethod
    def _filter_split(attr_df, split):
        """
        Filter the DataFrame to include only rows corresponding to the specified split.

        Args:
            attr_df (pd.DataFrame): The attribute DataFrame.
            split (str): The desired split ('train', 'val', 'test').

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        start_idx, end_idx = CelebADataset.SPLIT_RANGES[split]
        return attr_df.iloc[start_idx - 1 : end_idx]

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
