import pytest
import torch

from direct.nn.classifiers.vgg.vgg import *


def create_input(shape):
    data = torch.rand(shape).float()
    return data


@pytest.mark.parametrize(
    "shape",
    [[3, 2, 32, 32], [1, 4, 64, 100]],
)
@pytest.mark.parametrize(
    "model",
    [VGG11, VGG13, VGG16, VGG19],
)
@pytest.mark.parametrize(
    "num_classes",
    [3],
)
@pytest.mark.parametrize(
    "batchnorm",
    [True, False],
)
def test_inception(shape, model, num_classes, batchnorm):
    x = create_input(shape)
    m = model(shape[1], num_classes=num_classes, batchnorm=batchnorm)
    out = m(x)
    assert list(out.shape) == [shape[0], num_classes]
