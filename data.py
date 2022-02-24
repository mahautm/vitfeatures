# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import random
from typing import Optional

import torch

# from PIL import ImageFilter
from torchvision import datasets, transforms


def get_dataloader(
    dataset_dir: str,
    dataset_name: str,
    batch_size: int = 32,
    image_size: int = 32,
    num_workers: int = 1,
    return_original_image: bool = False,
    seed: int = 111,
):

    transformations = ImageTransformation(
        image_size, return_original_image, dataset_name
    ) # TODO : check what is happening here

    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root="./data", download=True, transform=transformations
        )
    else:
        train_dataset = datasets.ImageFolder(dataset_dir, transform=transformations)

    train_sampler = None

    test_dataset, train_dataset = torch.utils.data.random_split(
        train_dataset,
        [len(train_dataset) // 10, len(train_dataset) // 10 * 9],
        torch.Generator().manual_seed(seed),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return test_loader, train_loader


class ImageTransformation:
    """
    A stochastic data augmentation module that transforms any given data example
    randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(
        self,
        size: int,
        return_original_image: bool = False,
        dataset_name: Optional[str] = None,
    ):

        transformations = [
            transforms.Resize(size=(size, size)),
        ]

        transformations.append(transforms.ToTensor())

        if dataset_name == "imagenet":
            transformations.extend(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.CenterCrop(299),
                ]
            )
        elif dataset_name == "cifar100":
            transformations.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

        self.transform = transforms.Compose(transformations)

        self.return_original_image = return_original_image
        if self.return_original_image:
            self.original_image_transform = transforms.Compose(
                [transforms.Resize(size=(size, size)), transforms.ToTensor()]
            )

    def __call__(self, x):
        x_i, x_j = self.transform(x), self.transform(x)
        if self.return_original_image:
            return x_i, x_j, self.original_image_transform(x)
        return x_i, x_j
