from functools import cache
from folktables import ACSDataSource
import folktables
import numpy as np
from numpy.random import SeedSequence
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms

from merlin.models.torch import MODEL_INPUT_TRANSFORMATION_FACTORY
from merlin.utils import random_state
from .celeba import CelebADataset
from .utils import load_whole_dataset


@cache
def get_data(
    dataset: str,
    audit_pool_size: int,
    binarize_group: bool = False,
    traintest_seed: SeedSequence | None = None,
    auditset_seed: SeedSequence | None = None,
    **extra_args,
):
    if dataset == "ACSEmployment":
        data_source = ACSDataSource(
            survey_year="2018", horizon="1-Year", survey="person"
        )
        # group_col = "AGEP"
        group_col = "RAC1P"

        # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2022.pdf
        # RAC1P Character 1
        # Recoded detailed race code
        # 1 .White alone
        # 2 .Black or African American alone
        # 3 .American Indian alone
        # 4 .Alaska Native alone
        # 5 .American Indian and Alaska Native tribes specified; or
        #   .American Indian or Alaska Native, not specified and no other
        #   .races
        # 6 .Asian alone
        # 7 .Native Hawaiian and Other Pacific Islander alone
        # 8 .Some Other Race alone
        # 9 .Two or More Races²
        ACSEmployment = folktables.BasicProblem(
            features=[
                "AGEP",
                "SCHL",
                "MAR",
                "RELP",  # Not present in 2019
                "DIS",
                "ESP",
                "CIT",
                "MIG",
                "MIL",
                "ANC",
                "NATIVITY",
                "DEAR",
                "DEYE",
                "DREM",
                "SEX",
                "RAC1P",
            ],
            target="ESR",
            target_transform=lambda x: x == 1,
            group=group_col,
            preprocess=lambda x: x,
            # postprocess=lambda x: np.nan_to_num(x, -1),
        )

        if binarize_group:
            acs_data = data_source.get_data(states=["MN"], download=True)
            features, labeldf, groupdf = ACSEmployment.df_to_pandas(acs_data)

            label: pd.Series = labeldf["ESR"].astype(int)
            group: pd.Series = groupdf[group_col] != 1

        else:
            acs_data = data_source.get_data(states=["MN"], download=True)
            features, labeldf, groupdf = ACSEmployment.df_to_pandas(acs_data)

            label: pd.Series = labeldf["ESR"].astype(int)
            group: pd.Series = groupdf[group_col].astype(int)

            # Remove groups that have too few representatives
            n_per_group = group.value_counts()
            groups_to_remove = n_per_group.loc[n_per_group < 200].index
            samples_to_remove = group.isin(groups_to_remove)

            features, label, group = (
                features.loc[~samples_to_remove],
                label.loc[~samples_to_remove],
                group[~samples_to_remove],
            )

        train_test_idx, audit_idx = train_test_split(
            np.arange(len(features)),
            test_size=audit_pool_size,
            random_state=random_state(auditset_seed),
        )

        # The model owner splits its data into train/test
        train_idx, test_idx = train_test_split(
            train_test_idx,
            test_size=0.3,
            random_state=random_state(traintest_seed),
        )
        audit_idx: np.ndarray
        train_idx: np.ndarray
        test_idx: np.ndarray

    elif dataset == "celeba":
        transformation = transforms.Compose([transforms.ToTensor()])
        if "torch_model_architecture" in extra_args:
            meanstd = extra_args.get("meanstd", None)
            transformation_factory = MODEL_INPUT_TRANSFORMATION_FACTORY[
                extra_args["torch_model_architecture"]
            ]
            transformation = transformation_factory(meanstd)
        label_col = "Smiling"
        group_col = "Male"

        # Sample a subset of CelebA test split for the audit set
        rng = np.random.default_rng(auditset_seed)
        celeba = CelebADataset(
            target_columns=[label_col, group_col],
            transform=transformation,
            split="test",
        )
        indices = rng.choice(len(celeba), size=audit_pool_size, replace=False)
        celeba = torch.utils.data.Subset(celeba, indices=indices.tolist())

        features, [label, group] = load_whole_dataset(celeba)
        label = pd.Series(label)
        group = pd.Series(group).astype(bool)

        train_idx = np.array([])
        test_idx = np.array([])
        audit_idx = np.arange(audit_pool_size)
    else:
        raise NotImplementedError(f"The dataset {dataset} is not supported")

    print(f"dataset size: {features.shape[0]}")

    return features, label, group, train_idx, test_idx, audit_idx
