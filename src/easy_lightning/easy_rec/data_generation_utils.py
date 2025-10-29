import os
import pandas as pd
from pathlib import Path
import subprocess
import numpy as np
import pyarrow.parquet as pq
from ast import literal_eval
import datetime
from tqdm import tqdm
from scipy import stats
from typing import Dict, List, Tuple, Optional

def download_dataset(dataset_name: str, dataset_raw_folder: str, additional_file_name: Optional[str] = None) -> None:
    """
    Downloads the requested dataset from predefined sources (e.g., HuggingFace, GroupLens, Amazon, etc.).

    Args:
        dataset_name (str): Name of the dataset to download.
        dataset_raw_folder (str): Folder path to save the downloaded dataset.
        additional_file_name (str, optional): Additional file name required for some datasets.

    Returns:
        None
    """
    print(f"Downloading dataset {dataset_name} to {dataset_raw_folder}...")

    try:
        # Get the upper-level directory to store the dataset
        reduced_name = os.path.join(*dataset_raw_folder.split("/")[:-1])
        os.makedirs(reduced_name, exist_ok=True)

        if "yambda" in dataset_name:
            # Yandex YAMBDA dataset
            parquet_path = os.path.join(dataset_raw_folder, f'multi_event_{dataset_name.split("_")[1]}.parquet')
            if not os.path.exists(parquet_path):
                os.makedirs(dataset_raw_folder, exist_ok=True)
                url = f"https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/{dataset_name.split('_')[1]}/multi_event.parquet"
                subprocess.run(["wget", "-O", parquet_path, url], check=True)

        elif 'ml' in dataset_name.lower():
            # MovieLens datasets
            if not os.path.exists(os.path.join(dataset_raw_folder, additional_file_name)):
                url = f"https://files.grouplens.org/datasets/movielens/{dataset_name}.zip"
                output_path = os.path.join(reduced_name, f"{dataset_name}.zip")
                subprocess.run(["curl", "-o", output_path, url], check=True)
                subprocess.run(["unzip", output_path, "-d", reduced_name], check=True)

        elif 'amazon' in dataset_name.lower():
            # Amazon review datasets
            if not os.path.exists(os.path.join(dataset_raw_folder, additional_file_name)):
                os.makedirs(dataset_raw_folder, exist_ok=True)
                url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/{additional_file_name}.gz"
                output_path = os.path.join(dataset_raw_folder, f"{additional_file_name.split('.')[0]}.json.gz")
                subprocess.run(["curl", "-k", "-o", output_path, url], check=True)
                subprocess.run(["gzip", "-d", output_path], check=True)

        elif 'foursquare' in dataset_name.lower():
            # Foursquare check-in dataset
            target_folder = os.path.join(reduced_name, dataset_name)
            zip_path = os.path.join(target_folder, 'dataset_tsmc2014.zip')

            if not os.path.exists(os.path.join(dataset_raw_folder, additional_file_name)):
                os.makedirs(target_folder, exist_ok=True)
                url = "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
                subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)
                subprocess.run(["unzip", "-j", zip_path, "-d", target_folder], check=True)

        elif dataset_name == "steam":
            # Steam reviews dataset
            output_path = os.path.join(dataset_raw_folder, "steam_reviews.json.gz")
            if not os.path.exists(output_path):
                os.makedirs(dataset_raw_folder, exist_ok=True)
                url = "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"
                subprocess.run(["wget", "-O", output_path, url], check=True)
                subprocess.run(["gzip", "-d", output_path], check=True)

        elif dataset_name == "gowalla":
            # Gowalla check-in dataset
            output_path = os.path.join(dataset_raw_folder, "loc-gowalla_totalCheckins.txt.gz")
            if not os.path.exists(os.path.join(dataset_raw_folder, "loc-gowalla_totalCheckins.txt")):
                os.makedirs(dataset_raw_folder, exist_ok=True)
                url = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
                subprocess.run(["wget", "-O", output_path, url], check=True)
                subprocess.run(["gzip", "-d", output_path], check=True)

    except Exception as e:
        print(e)
        print(f"{dataset_name} not available in the list of datasets.")


def preprocess_dataset(
    name: str,
    data_folder: str = "../data/raw",
    min_rating: Optional[float] = None,
    min_items_per_user: int = 0,
    min_users_per_item: int = 0,
    densify_index: bool = True,
    split_method: str = "leave_n_out",
    split_keys: Dict[str, List[str]] = {
        "sid": ["train_sid", "val_sid", "test_sid"],
        "timestamp": ["train_timestamp", "val_timestamp", "test_timestamp"],
        "rating": ["train_rating", "val_rating", "test_rating"]
    },
    test_sizes: List[int] = [1, 1],
    random_state: Optional[int] = None,
    del_after_split: bool = True,
    **kwargs
) -> Tuple[Dict, Dict]:
    """
    Preprocesses the dataset for recommender systems:
    - Loads and filters ratings
    - Densifies user/item indices
    - Converts data into sequence format
    - Splits into train/val/test

    Args:
        name (str): Name of the dataset.
        data_folder (str): Base folder path where raw data resides.
        min_rating (float, optional): Minimum rating to retain.
        min_items_per_user (int): Min number of items per user.
        min_users_per_item (int): Min number of users per item.
        densify_index (bool): Whether to remap user/item IDs to 0-based dense indices.
        split_method (str): Data split strategy (e.g., "leave_n_out").
        split_keys (Dict): Keys to split and their resulting keys.
        test_sizes (List[int]): Size of test/validation split.
        random_state (int, optional): Seed for reproducibility.
        del_after_split (bool): Delete original keys after splitting.

    Returns:
        Tuple[Dict, Dict]: Processed dataset and mappings (e.g., user/item index mappings).
    """
    dataset_raw_folder = os.path.join(data_folder, name)

    # Ensure the raw data is ready
    maybe_preprocess_raw_dataset(dataset_raw_folder, name)

    df = load_ratings_df(dataset_raw_folder, name)
    df = filter_ratings(df, min_rating)
    df = filter_by_frequence(df, min_items_per_user, min_users_per_item)

    if densify_index:
        df, maps = densify_index_method(df)
        user_sequences_with_time = df.groupby("uid").apply(
        lambda g: (list(g.sort_values("timestamp")["sid"]),
                   list(g.sort_values("timestamp")["timestamp"])),
        include_groups=False).to_dict()
    print_stats(user_sequences_with_time, keep_time=True)

    data = df_to_sequences(df)

    data = split_rec_data(
        data,
        split_method,
        split_keys,
        test_sizes,
        random_state=random_state,
        del_after_split=del_after_split,
        **kwargs
    )

    return data, maps


def maybe_preprocess_raw_dataset(dataset_raw_folder: str, dataset_name: str) -> None:
    """
    Checks if preprocessed CSV data exists in the raw folder.
    If not, runs the specific preprocessing routine.

    Args:
        dataset_raw_folder (str): Path to the raw dataset folder.
        dataset_name (str): Name of the dataset.

    Returns:
        None
    """
    if any(Path(dataset_raw_folder).glob('*.csv')):
        return
    else:
        specific_preprocess(dataset_raw_folder, dataset_name)


def map_dataset_name() -> Dict[str, str]:
    """
    Returns a dictionary mapping dataset names to their primary data file.

    Returns:
        Dict[str, str]: Dataset name to file name mapping.
    """
    dataset_files = {
        "ml-1m": 'ratings.dat',
        "ml-100k": 'u.data',
        "ml-20m": 'ratings.csv',
        "steam": 'steam_reviews.json',
        "amazon_beauty": 'All_Beauty.json',
        "amazon_videogames": 'Video_Games.json',
        "amazon_toys": 'Toys_and_Games.json',
        "amazon_cds": 'CDs_and_Vinyl.json',
        "amazon_music": 'Digital_Music.json',
        "foursquare-nyc": 'dataset_TSMC2014_NYC.txt',
        "foursquare-tky": 'dataset_TSMC2014_TKY.txt',
        "behance": 'behance.csv',
        "gowalla": 'loc-gowalla_totalCheckins.txt',
    }
    return dataset_files


def get_rating_files_per_dataset(dataset_name: str) -> str:
    """
    Gets the path or URL to the rating file associated with the dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        str: Path or URL to the dataset's rating file.

    Raises:
        ValueError: If the dataset is not recognized.
    """
    datasets = map_dataset_name()
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not available!")

    if dataset_name != "yambda":
        return datasets[dataset_name]
    else:
        # Placeholder string – should probably be returned dynamically
        return "hf://datasets/yandex/yambda/flat/{dataset_name.split('_')[1]}/multi_event.parquet"

def specific_preprocess(dataset_raw_folder: str, dataset_name: str) -> None:
    """
    Performs dataset-specific preprocessing and stores a standardized CSV file.

    Args:
        dataset_raw_folder (str): Path to the raw dataset folder.
        dataset_name (str): Name of the dataset (e.g., 'steam', 'amazon_books', 'behance', etc.).

    Raises:
        NotImplementedError: If the dataset name is unknown or not supported.
    """
    datasets = map_dataset_name()

    if dataset_name == "steam":
        download_dataset(dataset_name, dataset_raw_folder=dataset_raw_folder)
        file_path = os.path.join(dataset_raw_folder, datasets[dataset_name])
        all_reviews = []
        with open(file_path, "r") as f:
            for line in f:
                line_dict = literal_eval(line)
                user_id = line_dict['username']
                item_id = line_dict['product_id']
                rating = 3  # Placeholder value, assumed to be positive interaction
                timestamp = line_dict['date']
                try:
                    timestamp = datetime.datetime.timestamp(datetime.datetime.strptime(timestamp, "%Y-%m-%d"))
                except ValueError:
                    timestamp = -1
                all_reviews.append((user_id, item_id, rating, int(timestamp)))
        df = pd.DataFrame(all_reviews)
        df.to_csv(os.path.join(dataset_raw_folder, f'{datasets[dataset_name].split(".")[0]}.csv'), header=False, index=False)

    elif dataset_name == "behance":
        file_path = os.path.join(dataset_raw_folder, f'{datasets[dataset_name]}.txt')
        all_reviews = []
        with open(file_path, "r") as f:
            for line in f:
                user_id, item_id, timestamp = line.strip().split(" ")
                rating = 3  # Placeholder value
                all_reviews.append((user_id, item_id, rating, int(timestamp)))
        df = pd.DataFrame(all_reviews)
        df.to_csv(os.path.join(dataset_raw_folder, f'{datasets[dataset_name]}.csv'), header=False, index=False)

    elif 'yambda' in dataset_name:
        possibilities = ['50m', '500m', '5b']
        if dataset_name.split('_')[1] not in possibilities:
            raise NotImplementedError(f"Yambda only supports the version {possibilities} so far!")
        download_dataset(dataset_name, dataset_raw_folder=dataset_raw_folder)
        output_path = os.path.join(dataset_raw_folder, f"multi_event_{dataset_name.split('_')[1]}.parquet")
        table = pq.read_table(output_path)
        table_decoded = table.replace_schema_metadata()
        columns = [table_decoded.column(i).combine_chunks().to_pylist() for i in range(table_decoded.num_columns)]
        column_names = table_decoded.schema.names
        df = pd.DataFrame(dict(zip(column_names, columns)))
        df.to_csv(os.path.join(dataset_raw_folder, f'{dataset_name}.csv'), header=False, index=False)

    elif "amazon" in dataset_name:
        orig_file_name = datasets[dataset_name]
        download_dataset(dataset_name=dataset_name, dataset_raw_folder=dataset_raw_folder, additional_file_name=orig_file_name)
        file_path = os.path.join(dataset_raw_folder, orig_file_name)
        all_reviews = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.replace('"verified": true,', '"verified": True,').replace('"verified": false,', '"verified": False,')
                line_dict = literal_eval(line)
                user_id = line_dict['reviewerID']
                item_id = line_dict['asin']
                rating = float(line_dict['overall'])
                timestamp = line_dict['unixReviewTime']
                all_reviews.append((user_id, item_id, rating, timestamp))
        df = pd.DataFrame(all_reviews)
        df.to_csv(os.path.join(dataset_raw_folder, orig_file_name.replace(".json", ".csv")), header=False, index=False)

    elif dataset_name == 'gowalla':
        file_path = os.path.join(dataset_raw_folder, datasets[dataset_name])
        df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')
        df.columns = ['uid', 'UTC_time', 'latitude', 'longitude', 'sid']
        df["rating"] = 1
        df["timestamp"] = df["UTC_time"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp())
        df.to_csv(os.path.join(dataset_raw_folder, 'gowalla.csv'), header=False, index=False)

    else:
        download_dataset(dataset_name=dataset_name, dataset_raw_folder=dataset_raw_folder,
                         additional_file_name=map_dataset_name()[dataset_name])


def load_ratings_df(dataset_raw_folder: str, dataset_name: str) -> pd.DataFrame:
    """
    Loads the ratings DataFrame for the specified dataset.

    Args:
        dataset_raw_folder (str): Path to the raw dataset folder.
        dataset_name (str): Name of the dataset.

    Returns:
        pd.DataFrame: Ratings DataFrame with columns ['uid', 'sid', 'rating', 'timestamp'].
    """
    datasets = map_dataset_name()

    if dataset_name == "ml-1m":
        df = pd.read_csv(os.path.join(dataset_raw_folder, 'ratings.dat'), sep='::', header=None, engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif dataset_name == "ml-100k":
        df = pd.read_csv(os.path.join(dataset_raw_folder, 'u.data'), sep='\t', header=None, engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif dataset_name == "ml-20m":
        df = pd.read_csv(os.path.join(dataset_raw_folder, 'ratings.csv'), engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif dataset_name == "steam" or dataset_name == "behance" or "amazon" in dataset_name:
        file_name = datasets[dataset_name].split('.')[0] + ".csv"
        df = pd.read_csv(os.path.join(dataset_raw_folder, file_name), header=None, engine="python")
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif "yambda" in dataset_name:
        df = pd.read_csv(os.path.join(dataset_raw_folder, f'{dataset_name}.csv'), header=None, encoding='latin-1', engine="python")
        df.columns = ['uid', 'timestamp','sid', "is_organic", "played_ratio_pct", "track_length_seconds", "event_type"]
        df['timestamp'] = df['timestamp'].astype(str).str.replace(',', '').astype(int) * 5
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.strftime('%H:%M:%S %z %Y')
        df['rating'] = 1
        return df
    elif "foursquare" in dataset_name:
        file_path = os.path.join(dataset_raw_folder, datasets[dataset_name])
        download_dataset(dataset_name=dataset_name, dataset_raw_folder=dataset_raw_folder, additional_file_name=datasets[dataset_name])
        df = pd.read_csv(file_path, sep='\t', header=None, encoding='latin-1', engine="python")
        df.columns = ['uid', 'sid', "s_cat", "s_cat_name", "latitude", "longitude", "timezone_offset", "UTC_time"]
        df["rating"] = 1
        df["timestamp"] = df["UTC_time"].apply(lambda x: datetime.datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y").timestamp())
        return df
    elif dataset_name == "gowalla":
        df = pd.read_csv(os.path.join(dataset_raw_folder, 'gowalla.csv'), header=None, engine='python')
        df.columns = ['uid', 'UTC_time', 'latitude', 'longitude', 'sid']
        df["rating"] = 1
        df["timestamp"] = df["UTC_time"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp())
        return df
    else:
        raise NotImplementedError(f"Dataset name {dataset_name} not recognized.")


def filter_ratings(df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    if min_rating is not None:
        print(f"Filtering the items with score < {min_rating}.")
        df = df[df['rating'] >= min_rating]
    return df


def filter_by_frequence(df: pd.DataFrame, min_items_per_user: int, min_users_per_item: int) -> pd.DataFrame:
    if min_users_per_item > 0:
        print(f'-------- Filtering by minimum number of users per item: {min_users_per_item} --------')
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= min_users_per_item]
        df = df[df['sid'].isin(good_items)]

    if min_items_per_user > 0:
        print(f'-------- Filtering by minimum number of items per user: {min_items_per_user} --------')
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= min_items_per_user]
        df = df[df['uid'].isin(good_users)]

    return df


def densify_index_method(df: pd.DataFrame, vars=["uid", "sid"]):
    print('-------- Densifying index --------')
    maps = {}
    for var_name in tqdm(vars):
        uniques = sorted(df[var_name].unique())  # sorted instead of appearance order
        maps[var_name] = {u: i + 1 for i, u in enumerate(uniques)}
        df[var_name] = df[var_name].map(maps[var_name])
    return df, maps


def df_to_sequences(df: pd.DataFrame, keep_vars=["uid"], seq_vars=["sid", "rating", "timestamp"], user_var="uid", time_var="timestamp") -> dict:
    df_group_by_user = df.groupby(user_var)
    data = {}
    for var in seq_vars:
        data[var] = df_group_by_user.apply(lambda d: list(d.sort_values(by=time_var)[var]),
                                            include_groups=False).values
    for var in keep_vars:
        data[var] = df_group_by_user[var].first().values
    return data


def print_stats(complete_set: dict, keep_time: bool):
    print(f"-------- Number of users: {len(complete_set)} --------")
    if keep_time:
        items = [seq for u, (seq, _) in complete_set.items()]
    else:
        items = [seq for u, seq in complete_set.items()]
    print(f"-------- Number of items: {len(set(np.concatenate(items)))} --------")
    lens = [len(seq) for seq in items]
    print(f"-------- Average Length: {np.mean(lens)} --------")
    print(f"-------- Median length: {np.median(lens)} --------")
    print(f"-------- Std of the length: {np.std(lens)}")
    print(f"-------- Min/max length: {np.min(lens), np.max(lens)} --------")
    print(f"-------- Number of interactions: {np.sum(lens)} --------")


def split_rec_data(data: dict, split_method: str, split_keys: dict, test_sizes: list, **kwargs) -> dict:
    print(f'-------- Splitting use {split_method} --------')
    if split_method == 'leave_n_out':
        for orig_key, new_keys in split_keys.items():
            while len(test_sizes) < len(new_keys):
                test_sizes.append(0)
            end_ids = np.array([len(seq) for seq in data[orig_key]])
            previous_key = orig_key
            for new_key, test_size in zip(new_keys[::-1], test_sizes[::-1]):
                end_ids -= test_size
                data[new_key] = np.array([seq[:end_ids[i]] for i, seq in enumerate(data[previous_key])], dtype=object)
                previous_key = new_key
            if kwargs.get("del_after_split"):
                del data[orig_key]
    else:
        raise NotImplementedError
    return data

def get_max_number_of(maps, key):
    return np.max(list(maps[key].values()))


if __name__ == '__main__':
    preprocess_dataset('gowalla')
