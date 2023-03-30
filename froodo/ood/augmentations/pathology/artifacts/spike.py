import numpy as np

from ....augmentations import OODAugmentation, SampableAugmentation
from .artifacts import ArtifactAugmentation
from .....ood.severity import ParameterSeverityMeasurement, SeverityMeasurement
from torchio.transforms import RandomSpike
from ...utils import *
from .....data.samples import Sample

class SpikeAugmentation(
    OODAugmentation, ArtifactAugmentation, SampableAugmentation
):
    def __init__(self,
                 num_spikes=3,
                 intensity=1,
                 sample_intervals=None,
                 severity: SeverityMeasurement = None,
                 keep_ignorred=True, ) -> None:
        super().__init__()
        self.num_spikes = num_spikes
        self.intensity = intensity
        if sample_intervals == None:
            self.sample_intervals = [(1.2, 2.5)]
        else:
            self.sample_intervals = sample_intervals
        self.severity_class = (
            ParameterSeverityMeasurement(
                "number of spikes", (self.sample_intervals[0][0], self.sample_intervals[-1][1])
            )
            if severity == None
            else severity
        )
        self.keep_ignorred = keep_ignorred

    def _apply_sampling(self):
        return super()._set_attr_to_uniform_samples_from_intervals(
            {"spike": self.sample_intervals}
        )

    def _get_parameter_dict(self):
        return {"number of spikes": self.num_spikes,
                "intensity": self.intensity}

    def _augment(self, sample: Sample) -> Sample:
        sample_image = sample["image"].numpy()
        augmentation_sample = np.expand_dims(sample_image, axis=0)
        transformation = RandomSpike(self.num_spikes, self.intensity)
        transformed_image = transformation(augmentation_sample)
        transformed_image = np.squeeze(transformed_image, axis=0)
        sample["image"] = torch.from_numpy(transformed_image)
        sample = full_image_ood(sample, self.keep_ignorred)
        return sample
