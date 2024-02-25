import subprocess
import sys
from pathlib import Path


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("pandas==2.*")
install("numpy==1.*")
install("openpyxl==3.*")
install("datasets==2.*")


import argparse
import logging
import os
import re

import pandas as pd
from datasets import Dataset

logging.basicConfig(format="%(asctime)s %(funcName)s %(message)s")
logging.getLogger().setLevel(logging.INFO)
logging.info("Setting up log level and format")


def _parse_args():

    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default, this is a S3 path under the default bucket.
    parser.add_argument("--filepath", type=str, default="/opt/ml/processing/input/")
    # filename-alt correspond to Alternate Titles
    parser.add_argument("--filename-alt", type=str, default="Alternate Titles.xlsx")
    # filename-occ corresponds to Occupation Data
    parser.add_argument("--filename-occ", type=str, default="Occupation Data.xlsx")
    parser.add_argument(
        "--filename-soc-structure", type=str, default="soc_structure_2018.csv"
    )

    parser.add_argument("--outputpath", type=str, default="/opt/ml/processing/output/")

    return parser.parse_known_args()


if __name__ == "__main__":
    # Process arguments
    args, _ = _parse_args()

    # todo: remove below when all fixed
    def print_directory_structure(directory_path):
        p = Path(directory_path)
        for f in p.glob("**/*"):
            print(f)

    print_directory_structure("/opt/ml/processing")

    df_alt = pd.read_excel(os.path.join(args.filepath, args.filename_alt))
    df_occ = pd.read_excel(os.path.join(args.filepath, args.filename_occ))

    # Extract relevant job titles and their corresponding Codes from both dataframes
    texts = []
    codes = []
    for texts_df in [
        df_alt["Alternate Title"].str.extract(r"\(([^()]*)\)")[0].dropna(),
        df_alt["Alternate Title"].apply(lambda s: re.sub(r"\(.*?\)", "", s)),
        df_occ[
            ~(
                df_occ["Title"].str.contains("Other")
                | df_occ["Title"].str.contains("and")
            )
        ]["Title"],
    ]:
        codes.extend(df_alt["O*NET-SOC Code"].loc[texts_df.index].to_list())
        texts.extend(texts_df.to_list())

    df_data = pd.DataFrame({"text": texts, "code": codes}).drop_duplicates()

    # Further processing
    df_data["text"] = df_data["text"].str.lower().str.strip()

    # Obtain mapping between O*NET codes and the corresponding major groups
    mapping_df = pd.read_csv(os.path.join(args.filepath, args.filename_soc_structure))
    mapping_df = mapping_df[["Major Group", "Name"]].dropna()
    code_to_major = {
        c[:2]: m for c, m in zip(mapping_df["Major Group"], mapping_df["Name"])
    }
    major_to_code = {
        m: c[:2] for c, m in zip(mapping_df["Major Group"], mapping_df["Name"])
    }

    # Use the obtained mapping to assign labels to each job title in our dataframe
    df_data["label"] = df_data["code"].apply(lambda c: code_to_major[c[:2]])

    # Transform to HuggingFace Datasets
    ds = Dataset.from_pandas(df_data)
    ds = ds.class_encode_column("label")

    # Stratified splits
    data_splits = ds.train_test_split(
        shuffle=True, seed=2140, test_size=1 - 0.9, stratify_by_column="label"
    )

    logging.info(
        f"Data split > train:{len(data_splits['train'])} |  test:{len(data_splits['test'])}"
    )

    data_splits["train"].to_parquet(
        os.path.join(args.outputpath, "train/train.parquet")
    )
    data_splits["test"].to_parquet(os.path.join(args.outputpath, "test/test.parquet"))

    logging.info("## Processing complete. Exiting.")
