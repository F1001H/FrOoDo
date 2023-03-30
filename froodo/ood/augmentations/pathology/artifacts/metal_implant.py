import random
from os import listdir
from os.path import join

import numpy as np
import torch

from ....augmentations import OODAugmentation, SampableAugmentation
from .artifacts import ArtifactAugmentation, data_folder
from .....ood.severity import PixelPercentageSeverityMeasurement, SeverityMeasurement
from .....ood.severity import ParameterSeverityMeasurement
from .....data.metadata import *
from ...utils import *
from .....data.samples import Sample


class MetalImplantAugmentation(ArtifactAugmentation, OODAugmentation, SampableAugmentation):
    def __init__(
        self,
        scale=0.5,
        intensity=2,
        severity: SeverityMeasurement = None,
        path=join(data_folder, "metal_implant/metal_implant_01.png"),
        mask_threshold=0.5,
        sample_intervals=None,
        keep_ignorred=True,
        position=None
    ) -> None:
        super().__init__()
        self.scale = scale
        self.intensity = intensity
        self.path = path
        self.position = position
        self.mask_threshold = mask_threshold
        self.severity_class: SeverityMeasurement = (
            PixelPercentageSeverityMeasurement() if severity == None else severity
        )
        if sample_intervals == None:
            self.sample_intervals = [(0.1, 5)]
        else:
            self.sample_intervals = sample_intervals
        self.keep_ignorred = keep_ignorred
        self.severity_class = (
            ParameterSeverityMeasurement(
                "scale", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity == None
            else severity
        )

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"scale": self.scale,
             "intensity": self.intensity}
        )

    def _augment(self, sample: Sample) -> Sample:
        sample_image = sample["image"].numpy()
        sample_mask = sample["mask"].numpy()
        for i in range(sample_image.shape[2]):
            image_slice = sample_image[:, :, i]
            mask_slice = sample_mask[:, :, i]
            image_slice, mask_slice = ArtifactAugmentation().overlay(self.path, image_slice, self.scale, self.intensity, self.position)
            sample_image[:, :, i] = image_slice
            sample_mask[:, :, i] = mask_slice
        sample["image"] = torch.from_numpy(sample_image)
        sample = full_image_ood(sample, self.keep_ignorred)
        return sample
