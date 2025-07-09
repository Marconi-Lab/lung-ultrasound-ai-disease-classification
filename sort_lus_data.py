import os
import shutil
import pandas as pd
import random

# A script to sort patient folders into covid, other, and healthy subdirectories by matching _uuid and Conclusion fields from the metadata

def sort_patient_folders_by_metadata(base_dir):
    """
    Sorts patient folders from the 'unsorted' directory into 'covid', 'other', or
    'healthy' subfolders based on the '_uuid' and 'Conclusion' fields 
    in the metadata Excel file.

    Parameters:base_dir (str): The base directory containing the 'unsorted' folder and metadata.xlsx
    """
    # Define paths
    unsorted_dir = os.path.join(base_dir, "unsorted")
    covid_dir = os.path.join(base_dir, "covid")
    other_dir = os.path.join(base_dir, "other")
    healthy_dir = os.path.join(base_dir, "healthy")    
    metadata_path = os.path.join(base_dir, "metadata.xlsx")

    # Create destination folders if they don't exist
    os.makedirs(covid_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)   

    # Load metadata
    df = pd.read_excel(metadata_path)

    # Normalize UUIDs to strings
    df["_uuid"] = df["_uuid"].astype(str)

    # Create a mapping from UUID to Conclusion
    uuid_to_conclusion = dict(zip(df["_uuid"], df["Conclusion"]))

    # Iterate over folders in unsorted
    for folder_name in os.listdir(unsorted_dir):
        folder_path = os.path.join(unsorted_dir, folder_name)

        if os.path.isdir(folder_path):
            folder_uuid = folder_name.strip()

            conclusion = uuid_to_conclusion.get(folder_uuid)

            # Decide target folder
            if conclusion == "Healthy Lung":
                target_dir = healthy_dir
            elif conclusion == "Probably Covid":
                target_dir = covid_dir
            elif conclusion == "Diseased lung but probably Not Covid":
                target_dir = other_dir            
            else:
                print(f"Unrecognized conclusion '{conclusion}' for UUID: {folder_uuid}")
                target_dir = no_uuid_dir

            # Copy folder
            dest_path = os.path.join(target_dir, folder_name)
            if not os.path.exists(dest_path):
                shutil.copytree(folder_path, dest_path)
                print(f"Copied {folder_uuid} to {target_dir}")
            else:
                print(f"Folder already exists at destination: {dest_path}")

# Example usage
base_dir = "/path/to/dataset"            # ← Update this
sort_patient_folders_by_metadata(base_dir)


# Script for splitting folders in the covid, other, and healthy categories into train, validation, and test subsets at a 70:20:10 ratio, based on folder-level partitioning.


def split_folders_by_ratio(source_dir, dest_base_dir, split_ratios=(0.7, 0.2, 0.1), seed=42):
    """
    Split folders from source_dir into train, val, test directories
    based on split_ratios.
    """
    categories = ['covid', 'healthy', 'other']
    splits = ['train', 'val', 'test']
    split_names = dict(zip(splits, split_ratios))

    random.seed(seed) 

    for category in categories:
        category_path = os.path.join(source_dir, category)
        if not os.path.exists(category_path):
            print(f"Category folder not found: {category_path}")
            continue

        # Get all patient folders in category
        folder_names = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
        random.shuffle(folder_names)

        # Split indexes
        total = len(folder_names)
        train_end = int(total * split_ratios[0])
        val_end = train_end + int(total * split_ratios[1])

        split_indices = {
            'train': folder_names[:train_end],
            'val': folder_names[train_end:val_end],
            'test': folder_names[val_end:]
        }

        # Copy folders to destination
        for split in splits:
            split_dir = os.path.join(dest_base_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for folder in split_indices[split]:
                src = os.path.join(category_path, folder)
                dst = os.path.join(split_dir, folder)
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
                else:
                    print(f"Already exists: {dst}")

    print("Folder splitting completed.")

# Example usage
source_directory = "/path/to/dataset"            # ← Update this
destination_directory = "/path/to/dataset_split"  # ← Update this

split_folders_by_ratio(source_directory, destination_directory)

