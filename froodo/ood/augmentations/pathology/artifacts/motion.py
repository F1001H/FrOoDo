import numpy as np

from ....augmentations import OODAugmentation, SampableAugmentation
from .artifacts import ArtifactAugmentation
from .....ood.severity import ParameterSeverityMeasurement, SeverityMeasurement
from torchio.transforms import RandomMotion
from ...utils import *
from .....data.samples import Sample


class MotionAugmentation(
    OODAugmentation, ArtifactAugmentation, SampableAugmentation
):
    def __init__(self,
                 degrees=10,
                 translation=10,
                 num_movements=2,
                 sample_intervals=None,
                 severity: SeverityMeasurement = None,
                 keep_ignorred=True, ) -> None:
        super().__init__()
        self.degrees = degrees
        self.translation = translation
        self.num_movements = num_movements
        if sample_intervals == None:
            self.sample_intervals = [(1.2, 2.5)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "degrees", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity == None
            else severity
        )
        self.keep_ignorred = keep_ignorred

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"motion": self.sample_intervals}
        )

    def _get_parameter_dict(self):
        return {"degrees": self.degrees,
                "translation": self.translation,
                "num_movements": self.num_movements}

    def _augment(self, sample: Sample) -> Sample:
        sample_image = sample["image"].numpy()
        '''
        for i in range(sample_image.shape[2]):
            slice = sample_image[:, :, i]
            augmentation_sample = np.expand_dims(slice, axis=0)
            augmentation_sample = np.expand_dims(augmentation_sample, axis=3)
            transformation = RandomMotion(degrees=self.degrees, translation=self.translation,
                                          num_transforms=self.num_movements, image_interpolation="linear")
            transformed_image = transformation(augmentation_sample)
            transformed_image = np.squeeze(transformed_image, axis=0)
            transformed_image = np.squeeze(transformed_image, axis=2)
            sample_image[:, :, i] = transformed_image
        '''
        augmentation_sample = np.expand_dims(sample_image, axis=0)
        augmentation_sample = np.swapaxes(augmentation_sample, 2, 3)
        transformation = RandomMotion(degrees=self.degrees, translation=self.translation,
                                      num_transforms=self.num_movements, image_interpolation="linear")
        transformed_image = transformation(augmentation_sample)
        transformed_image = np.swapaxes(transformed_image, 2, 3)
        transformed_image = np.squeeze(transformed_image, axis=0)
        sample["image"] = torch.from_numpy(transformed_image)
        sample = full_image_ood(sample, self.keep_ignorred)
        return sample
