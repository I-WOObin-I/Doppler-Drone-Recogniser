import os
import random
from torch.utils.data import Dataset
import numpy as np

"""
file tree:
- ./archive
    - ./Drones
            - XX-XX
                - YYY.csv
    - ./Cars
            - XX-XX
                - YYY.csv
    - ./People
            - XX-XX
                - YYY.csv
"""

BASE_DIR = "../archive"
DRONES_DIR = BASE_DIR + "/Drones"
CARS_DIR = BASE_DIR + "/Cars"
PEOPLE_DIR = BASE_DIR + "/People"

class RangeDopplerDataset(Dataset):
    def __init__(self, range_start=0, range_end=1):

        self.range_start = range_start
        self.range_end = range_end

        self.with_drone_files = self._get_sub_catalog_file_paths(DRONES_DIR)
        self.wout_drone_files = self._get_sub_catalog_file_paths(CARS_DIR)

        self._build_dataset()

        self.all_file_paths = [(p, 1) for p in self.with_drone_files] + [(p, 0) for p in self.wout_drone_files]
        random.shuffle(self.all_file_paths)


    def __len__(self):
        return len(self.all_file_paths)


    def __getitem__(self, idx):
        file_path, label = self.all_file_paths[idx]
        arr = np.loadtxt(file_path, delimiter=",")
        arr = self._transform_data(arr)
        return arr, label


    def _get_sub_catalog_file_paths(self, catalog_dir):
        paths = []
        for d in os.listdir(catalog_dir):
            sub_dir = os.path.join(catalog_dir, d)
            if os.path.isdir(sub_dir):
                for f in os.listdir(sub_dir):
                    if f.endswith(".csv"):
                        paths.append(os.path.join(sub_dir, f))
        return paths
    

    def _build_dataset(self):
        with_drone_files = self.with_drone_files

        files_len = len(with_drone_files)
        start_index = self.range_start * files_len
        end_index = self.range_end * files_len

        with_drone_files = with_drone_files[start_index:end_index]
        self.with_drone_files = with_drone_files
        print(f"Number of with_drone_files: {len(with_drone_files)}")


        wout_drone_files = self.wout_drone_files

        files_len = len(wout_drone_files)
        start_index = self.range_start * files_len
        end_index = self.range_end * files_len

        wout_drone_files = wout_drone_files[start_index:end_index]
        self.wout_drone_files = wout_drone_files
        print(f"Number of wout_drone_files: {len(wout_drone_files)}")


    def _transform_data(self, arr: np.ndarray):
        arr = arr * -1
        arr_min = arr.min()
        arr_max = arr.max()
        arr = (arr - arr_min) / (arr_max - arr_min)

        return arr




# if __name__ == "__main__":
#     dataset = RangeDopplerDataset()

#     print(f"Total dataset size: {len(dataset)}")

#     np.set_printoptions(precision=2, suppress=True)
#     # print(f"file at index 123:\n{dataset[123][0]}")

#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(7, 5))
#     im = plt.imshow(dataset[123][0], cmap="magma", aspect="auto", origin="lower")
#     plt.colorbar(im, label="Normalized intensity")
#     plt.title(f"Range-Doppler sample (label={'Drone' if dataset[123][1] == 1 else 'Other'})")
#     plt.xlabel("Doppler")
#     plt.ylabel("Range")
#     plt.tight_layout()
#     plt.show()