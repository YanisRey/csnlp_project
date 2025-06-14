import os
import requests
import pandas as pd
from zipfile import ZipFile
from io import StringIO

# Updated dictionary with more reliable sources
DATASETS = {
    "WordSim353": {
        "url": "https://gist.githubusercontent.com/ymerz/1a4e2b5f8a3e3e9d0a7c3e4d5b6a7b8c/raw/wordsim353.csv",
        "files": ["wordsim353.csv"],
        "sep": ","
    },
    "SimLex999": {
        "url": "https://fh295.github.io/SimLex-999.zip",
        "files": ["SimLex-999/SimLex-999.txt"],
        "sep": "\t"
    },
    "RG65": {
        "url": "https://raw.githubusercontent.com/kudkudak/word-embeddings-benchmarks/master/data/word-sim/rg65.csv",
        "files": ["rg65.csv"],
        "sep": ","
    },
    "MTurk287": {
        "url": "https://raw.githubusercontent.com/kudkudak/word-embeddings-benchmarks/master/data/word-sim/mturk-287.csv",
        "files": ["mturk-287.csv"],
        "sep": ","
    },
    "MTurk771": {
        "url": "https://raw.githubusercontent.com/kudkudak/word-embeddings-benchmarks/master/data/word-sim/mturk-771.csv",
        "files": ["mturk-771.csv"],
        "sep": ","
    }
}

def download_dataset(dataset_name, data_dir="word_similarity_data"):
    """Download and extract a word similarity benchmark dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset. Available datasets: {list(DATASETS.keys())}")
    
    os.makedirs(data_dir, exist_ok=True)
    dataset_info = DATASETS[dataset_name]
    url = dataset_info["url"]
    
    print(f"Downloading {dataset_name} dataset...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Handle zip files
        if url.endswith('.zip'):
            zip_file = ZipFile(BytesIO(response.content))
            for file in dataset_info["files"]:
                if file in zip_file.namelist():
                    with zip_file.open(file) as zf:
                        content = zf.read().decode('utf-8')
                    df = pd.read_csv(StringIO(content), sep=dataset_info.get("sep", "\t"))
                    save_path = os.path.join(data_dir, os.path.basename(file))
                    df.to_csv(save_path, index=False)
                    return df
        else:
            # Handle direct CSV/TXT files
            content = response.text
            df = pd.read_csv(StringIO(content), sep=dataset_info.get("sep", ","))
            file_name = url.split('/')[-1]
            save_path = os.path.join(data_dir, file_name)
            df.to_csv(save_path, index=False)
            return df
            
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    # Download WordSim353 specifically
    dataset_name = "WordSim353"
    df = download_dataset(dataset_name)
    
    if df is not None:
        print(f"\nSuccessfully downloaded {dataset_name} dataset:")
        print(df.head())
    else:
        print(f"\nFailed to download {dataset_name} dataset")