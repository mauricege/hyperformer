"""Implements different tasks and defines the processors to convert each dataset
to a sequence to sequence format."""
from collections import OrderedDict

import abc
import datasets
import functools
import logging
import numpy as np
import torch
import torchaudio
import pandas as pd
from hyperformer.metrics import metrics
from typing import Callable, Dict, Mapping, List
from datasets import Dataset
from glob import glob
from os.path import join

from .utils import round_stsb_target, compute_task_max_decoding_length

logger = logging.getLogger(__name__)


class AbstractSpeechPersonalisationDataset(abc.ABC):
    """Defines the abstract class for all the tasks.
    metrics: specifies the metrics to evaluate the task based on them.
    """

    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    target: str = NotImplemented

    def __init__(
        self,
        df_train,
        df_val,
        df_test,
        target,
        groups,
        processor,
        target_sampling_rate=16000,
        seed=42,
    ):
        self.seed = seed
        self.groups = groups
        self.name = "_".join(sorted(self.groups))
        self.df_train = df_train[df_train.group.isin(groups)]
        self.df_val = df_val[df_val.group.isin(groups)]
        self.df_test = df_test[df_test.group.isin(groups)]
        self.target_sampling_rate = target_sampling_rate
        self.processor = processor
        self.splits = {
            "train": self.df_train,
            "validation": self.df_val,
            "test": self.df_test,
        }

    def get_sampled_split(self, split: int, n_obs: int = None):
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        split = self.split_to_data_split[split]
        dataset = self.load_dataset(split)
        total_size = len(dataset)
        n_obs = self.check_n_obs(n_obs, total_size)
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def get_shuffled_sampled_split(self, split: int, n_obs: int = None):
        # Defines the random generator.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(mapped_split)
        # shuffle the dataset and get the random samples.
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        dataset = self.select_dataset_samples(indices, dataset, n_obs=n_obs)
        return dataset

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def select_dataset_samples(self, indices, dataset, n_obs: int = None):
        """
        Given a dataset for the split, obtains the sample indices for this split
        and returns the subsampled dataset.
        :param indices: the selected indices.
        :param dataset: dataset corresponding to this split.
        :return: subsampled dataset.
        """
        n_obs = self.check_n_obs(n_obs, len(indices))
        indices = indices[:n_obs] if n_obs is not None else indices
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return Dataset.from_pandas(self.splits[split])

    def get_train_split_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["train"]
        dataset = self.load_dataset(mapped_split)
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        validation_size = 1000
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def get_half_validation_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["validation"]
        dataset = self.load_dataset(mapped_split)
        validation_size = len(dataset)
        indices = torch.randperm(validation_size, generator=generator).tolist()
        if split == "validation":
            return indices[: (validation_size // 2)]
        else:
            return indices[validation_size // 2 :]

    def get_dataset(self, split, n_obs=None, split_validation_test=False):

        # TODO: later we can join these as one.
        # if n_obs == -1:
        #     # split = self.get_sampled_split(split, n_obs)
        #     dataset = self.load_dataset(split=split)
        # else:
        #     # shuffles the data and samples it.
        #     dataset = self.get_shuffled_sampled_split(split, n_obs)
        dataset = self.load_dataset(split=split)
        return dataset.map(
            self.preprocessor,
            remove_columns=dataset.column_names,
        )

    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(
            sampling_rate, self.target_sampling_rate
        )
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def preprocessor(self, example):
        speech_list = self.speech_file_to_array_fn(example["filename"])
        target_list = float(
            example[self.target]
        )  # Do any preprocessing on your float/integer data

        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
        result["labels"] = target_list
        result["filename"] = example["filename"]
        result["task"] = self.name

        return result


class RegressionSpeechPersonalisationDataset(AbstractSpeechPersonalisationDataset):
    metrics = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]

    def __init__(
        self,
        label_base,
        data_base,
        group_column,
        groups,
        target,
        processor,
        target_sampling_rate,
        seed=42,
        apply_group_fn=lambda x: x,
    ):
        self.target = target
        all_files = list(glob(f"{data_base}/**/*.wav", recursive=True))

        df_train = pd.read_csv(f"{label_base}/train.csv", encoding="utf-8")
        df_dev = pd.read_csv(f"{label_base}/dev.csv", encoding="utf-8")
        df_test = pd.read_csv(f"{label_base}/test.csv", encoding="utf-8")

        df_train["filename"] = df_train["filename"].apply(lambda x: join(data_base, x))
        df_train["group"] = df_train[group_column].apply(apply_group_fn)
        df_train = df_train[df_train["filename"].isin(all_files)]

        df_dev["filename"] = df_dev["filename"].apply(lambda x: join(data_base, x))
        df_dev["group"] = df_dev[group_column].apply(apply_group_fn)
        df_dev = df_dev[df_dev["filename"].isin(all_files)]

        df_test["filename"] = df_test["filename"].apply(lambda x: join(data_base, x))
        df_test = df_test[df_test["filename"].isin(all_files)]
        df_test["group"] = df_test[group_column].apply(apply_group_fn)
        super().__init__(
            df_train,
            df_dev,
            df_test,
            target=self.target,
            groups=groups,
            processor=processor,
            target_sampling_rate=target_sampling_rate,
            seed=seed,
        )


TASK_MAPPING = OrderedDict(
    [
        ("Regression", RegressionSpeechPersonalisationDataset),
    ]
)


class AutoTask:
    @classmethod
    def get(
        self,
        task_name,
        label_base,
        data_base,
        group_column,
        groups,
        target,
        processor,
        target_sampling_rate,
        seed=42,
        apply_group_fn=lambda x: x,
    ):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](
                label_base,
                data_base,
                group_column,
                groups,
                target,
                processor,
                target_sampling_rate,
                seed,
                apply_group_fn,
            )
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
