import numpy as np

from ....augmentations import OODAugmentation, SampableAugmentation
from .artifacts import ArtifactAugmentation
from .....ood.severity import ParameterSeverityMeasurement, SeverityMeasurement
from ...utils import *
from .....data.samples import Sample
import cv2


class AliasingAugmentation(
    OODAugmentation, ArtifactAugmentation, SampableAugmentation
):
    def __init__(self,
                 vertical=0.05,
                 horizontal=0.05,
                 sample_intervals=None,
                 severity: SeverityMeasurement = None,
                 keep_ignorred=True, ) -> None:
        super().__init__()
        self.vertical = vertical
        self.horizontal = horizontal
        if sample_intervals == None:
            self.sample_intervals = [(1.2, 2.5)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "vertical", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity == None
            else severity
        )
        self.keep_ignorred = keep_ignorred

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"alasing": self.sample_intervals}
        )

    def _get_parameter_dict(self):
        return {"vertical": self.vertical,
                "horizontal": self.horizontal,
                }

    def _augment(self, sample: Sample) -> Sample:
        sample_image = sample["image"].numpy()
        for i in range(sample_image.shape[2]):
            sample_image[:, :, i] = self.change_fov(sample_image[:, :, i])
        sample["image"] = torch.from_numpy(sample_image)
        sample = full_image_ood(sample, self.keep_ignorred)
        return sample

    def change_fov(self, image):
        delta_v = round(image.shape[0] * self.vertical)
        delta_h = round(image.shape[1] * self.horizontal)
        rescaled_image = image[delta_v:image.shape[0]-delta_v, delta_h:image.shape[1]-delta_h]

        right_cutoff = np.zeros_like(rescaled_image)
        right_cutoff.fill(-1024.0)
        right_cutoff[:, :delta_h] = image[delta_v:image.shape[0]-delta_v, image.shape[1]-delta_h:]
        right_cutoff = cv2.resize(right_cutoff, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        left_cutoff = np.zeros_like(rescaled_image)
        left_cutoff.fill(-1024.0)
        left_cutoff[:, left_cutoff.shape[1]-delta_h:] = image[delta_v:image.shape[0]-delta_v, :delta_h]
        left_cutoff = cv2.resize(left_cutoff, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        bottom_cutoff = np.zeros_like(rescaled_image)
        bottom_cutoff.fill(-1024.0)
        bottom_cutoff[:delta_v, :] = image[image.shape[0]-delta_v:, delta_h:image.shape[1]-delta_h]
        bottom_cutoff = cv2.resize(bottom_cutoff, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        top_cutoff = np.zeros_like(rescaled_image)
        top_cutoff.fill(-1024.0)
        top_cutoff[top_cutoff.shape[0]-delta_v:, :] = image[:delta_v, delta_h:image.shape[1]-delta_h]
        top_cutoff = cv2.resize(top_cutoff, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        rescaled_image = cv2.resize(rescaled_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        stacked_image = cv2.add(rescaled_image, right_cutoff)
        stacked_image = cv2.add(stacked_image, left_cutoff)
        stacked_image = cv2.add(stacked_image, bottom_cutoff)
        stacked_image = cv2.add(stacked_image, top_cutoff)
        return stacked_image
