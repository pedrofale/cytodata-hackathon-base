from upath import UPath as Path
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nbvv
import os
import warnings
from aicsimageio import transforms, AICSImage
from aicsimageprocessing import diagnostic_sheet, read_ome_zarr, rescale_image, imgtoprojection

from serotiny.io.image import image_loader
from cytodata_aics.io_utils import rescale_image

logging.getLogger("bfio").setLevel(logging.ERROR)
logging.getLogger("bfio.backends").setLevel(logging.ERROR)
logging.getLogger("aicsimageio").setLevel(logging.ERROR)



#From Chapter 5
#loading library, making path for 
def split_dataframe_(
    dataframe,
    train_frac,
    val_frac,
    seed,
    return_splits = True
):
    """Given a pandas dataframe, perform a train-val-test split and either return three
    different dataframes, or append a column identifying the split each row belongs to.
    TODO: extend this to enable balanced / stratified splitting
    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe
    train_frac: float
        Fraction of data to use for training. Must be <= 1
    val_frac: Optional[float]
        Fraction of data to use for validation. By default,
        the data not used for training is split in half
        between validation and test
    return_splits: bool = True
        Whether to return the three splits separately, or to append
        a column to the existing dataframe and return the modified
        dataframe
    """

    # import here to optimize CLIs / Fire usage
    from sklearn.model_selection import train_test_split

    train_ix, val_test_ix = train_test_split(
        dataframe.index.tolist(), train_size=train_frac
    )
    if val_frac is not None:
        val_frac = val_frac / (1 - train_frac)
    else:
        # by default use same size for val and test
        val_frac = 0.5

    val_ix, test_ix = train_test_split(val_test_ix, train_size=val_frac, random_state=seed)

    if return_splits:
        return dict(
            train=dataframe.loc[train_ix],
            valid=dataframe.loc[val_ix],
            test=dataframe.loc[test_ix],
        )

    dataframe.loc[train_ix, "split"] = "train"
    dataframe.loc[val_ix, "split"] = "valid"
    dataframe.loc[test_ix, "split"] = "test"

    return dataframe

def split_dataframe_with_seed(df, train_frac, val_frac, seed):
    Path("/home/aicsuser/serotiny_data/").mkdir(parents=True, exist_ok=True)
    
    # Sample n cells per group
    n = 2000 # number of cells per mitotic class
    cells_to_include=[]
    for name, group in df.groupby('cell_stage'):    
        sampled_group = group.sample(min([n,len(group)]))
        cells_to_include.append(sampled_group)
    df_mitocells = pd.concat(cells_to_include).reset_index(drop=True)

    # Discarding all the M6M7_single cells
    df_mitocells = df_mitocells.drop(df_mitocells[df_mitocells['cell_stage']=='M6M7_single'].index)

    # Add the train, test and validate split
    df_mitocells = split_dataframe_(dataframe=df_mitocells, train_frac=train_frac, val_frac=val_frac, return_splits=False, seed=seed)

    # df_mitocells.to_csv("/home/aicsuser/serotiny_data/mitocells.csv") 
    print(f"Number of cells: {len(df_mitocells)}")
    print(f"Number of columns: {len(df_mitocells.columns)}")

    return df_mitocells


def generate_dataset(train_frac=0.7, val_frac=0.2, seed=42):
    df = pd.read_parquet("s3://allencell-hipsc-cytodata/hackathon_manifest_17oct2022.parquet")
    print(f'Number of cells: {len(df)}')
    print(f'Number of columns: {len(df.columns)}')
    data_frame = split_dataframe_with_seed(df, train_frac=0.7, val_frac=0.2, seed=42)

    # M0               2000
    # M1M2             2000
    # M4M5             2000
    # M6M7_complete    1198
    # M3                981

    CLASS_DICT = {
        "M0": 0,
        "M1M2": 1,
        "M3": 2,
        "M4M5": 3,
        "M6M7_complete": 4,
    }
    data_frame['cell_stage_code'] = data_frame['cell_stage'].map(CLASS_DICT)
    print(data_frame['cell_stage_code'].value_counts())
    return data_frame
