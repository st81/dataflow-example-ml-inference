# TODO: Add the reference to the training codes repository

from typing import Callable, List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, imgs, targets: Optional[List[int]] = None, transform: Optional[Callable] = None) -> None:
        self.imgs = imgs
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx: int):
        img = self.imgs[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.targets is not None:
            target = self.targets[idx]
            return img, target
        else:
            return img

    def __len__(self) -> int:
        return len(self.imgs)


class MNISTModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims

        # architecture attributes
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return F.log_softmax(x, dim=1)
