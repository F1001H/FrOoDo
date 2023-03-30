import torch
from torch.utils.data import Dataset
from . import SampleDataset
from ... import Sample
import nibabel as nib
from os.path import join


class NiiGzDataset(Dataset, SampleDataset):
    def __init__(self, image_folder, mask_folder) -> None:
        super().__init__()
        self.images = image_folder
        self.masks = mask_folder

    def __getitem__(self, idx):
        file_name = "lung_" + str(idx) + ".nii"
        img = nib.load(join(self.images, file_name)).get_fdata()
        img = img + 1024
        img = img / 1024.0
        masks = nib.load(join(self.masks, file_name)).get_fdata()
        masks = masks + 1024
        masks = masks / 1024.0
        sample = Sample(torch.from_numpy(img))
        sample["mask"] = torch.from_numpy(masks)
        return sample

