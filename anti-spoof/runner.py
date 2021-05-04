# flake8: noqa
from collections import OrderedDict

from catalyst import utils
from catalyst.contrib.datasets import MNIST
from catalyst.data.sampler import BalanceClassSampler
from catalyst.data.transforms import ToTensor
from catalyst.dl import IRunner, SupervisedConfigRunner


class IRunnerMixin(IRunner):
    def get_model(self, stage: str):
        if self.model is None:
            # first stage
            model = super().get_model(stage=stage)
        else:
            model = self.model
        conv_layres = ["conv_net"]
        if stage == "tune":
            # second stage logic
            model = self.model
            for key in conv_layres:
                layer = getattr(model, key)
                utils.set_requires_grad(layer, requires_grad=False)
        return model

    def get_transform(self, stage: str = None, mode: str = None):
        return ToTensor()

    def get_datasets(
            self, stage: str, num_samples_per_class: int = None
    ) -> "OrderedDict[str, Dataset]":
        """Provides train/validation datasets from MNIST dataset."""
        num_samples_per_class = num_samples_per_class or 320
        datasets = OrderedDict()
        for mode in ("train", "valid"):
            dataset = MNIST(
                "./data",
                train=(mode == "train"),
                download=True,
                transform=self.get_transform(stage=stage, mode=mode),
            )
            if mode == "train":
                dataset = {
                    "dataset": dataset,
                    "sampler": BalanceClassSampler(
                        labels=dataset.targets, mode=num_samples_per_class
                    ),
                }
            datasets[mode] = dataset

        return datasets


class CustomSupervisedConfigRunner(IRunnerMixin, SupervisedConfigRunner):
    pass


__all__ = ["CustomSupervisedConfigRunner"]
