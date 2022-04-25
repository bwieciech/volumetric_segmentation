import os
import cv2
import numpy as np
import torch.utils.data
from skimage.transform import resize

file_path = os.path.dirname(os.path.abspath(__file__))


class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_root, folders, reshape_dhw=None, cache_loaded=False):
        self.path_to_root = path_to_root
        self.folders = folders
        self.reshape_dhw = reshape_dhw
        self.cache_loaded = cache_loaded
        self.cache = {}

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        if self.cache_loaded and idx in self.cache:
            return self.cache[idx]

        path = os.path.join(self.path_to_root, self.folders[idx])
        image = BrainMRIDataset.__load_image(path, is_mask=False)
        mask = BrainMRIDataset.__load_image(path, is_mask=True)
        if self.reshape_dhw is not None:
            image = resize(image, (image.shape[0], *self.reshape_dhw))
            mask = resize(mask, self.reshape_dhw)
        if self.cache_loaded:
            self.cache[idx] = (image, mask)
        return image, mask

    @staticmethod
    def __load_image(path_to_directory, is_mask):
        if is_mask:
            return np.array(
                [
                    cv2.imread(
                        os.path.join(path_to_directory, filename), cv2.IMREAD_GRAYSCALE
                    )
                    for filename in sorted(os.listdir(path_to_directory))
                    if "mask" in filename
                ]
            )
        else:
            image = np.array(
                [
                    cv2.cvtColor(
                        cv2.imread(os.path.join(path_to_directory, filename)),
                        cv2.COLOR_BGR2RGB,
                    )
                    for filename in sorted(os.listdir(path_to_directory))
                    if "mask" not in filename
                ]
            )
            image = np.moveaxis(image, -1, 0)
            return image
