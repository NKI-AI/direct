import pytest
import torch

from direct.nn.classifiers.inception.inception import Inception
from direct.nn.types import ActivationType


def create_input(shape):
    data = torch.rand(shape).float()
    return data


@pytest.mark.parametrize(
    "shape",
    [[3, 2, 32, 32], [4, 23, 100, 134]],
)
@pytest.mark.parametrize(
    "hidden_channels",
    [4, 6],
)
@pytest.mark.parametrize(
    "num_classes",
    [3, 10],
)
@pytest.mark.parametrize(
    "activation",
    [ActivationType.relu, ActivationType.prelu, ActivationType.leaky_relu],
)
def test_inception(shape, hidden_channels, num_classes, activation):
    x = create_input(shape)
    m = Inception(shape[1], hidden_channels=hidden_channels, num_classes=num_classes, activation_name=activation)
    out = m(x)
    assert list(out.shape) == [shape[0], num_classes]
