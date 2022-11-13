import pytest
import torch

from direct.nn.classifiers.resnet.resnet import *


def create_input(shape):
    data = torch.rand(shape).float()
    return data


@pytest.mark.parametrize(
    "shape",
    [[3, 2, 32, 32], [1, 4, 64, 100]],
)
@pytest.mark.parametrize(
    "model",
    [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152],
)
@pytest.mark.parametrize(
    "num_classes",
    [3],
)
def test_inception(shape, model, num_classes):
    x = create_input(shape)
    m = model(shape[1], num_classes=num_classes)
    out = m(x)
    assert list(out.shape) == [shape[0], num_classes]
