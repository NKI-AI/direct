import pytest
import torch

from direct.nn.classifiers.densenet.densenet import DenseNet
from direct.nn.types import ActivationType


def create_input(shape):
    data = torch.rand(shape).float()
    return data


@pytest.mark.parametrize(
    "shape",
    [[3, 2, 32, 32], [4, 23, 100, 134]],
)
@pytest.mark.parametrize(
    "bottleneck_channels",
    [4, 6],
)
@pytest.mark.parametrize(
    "expansion",
    [4, 6],
)
@pytest.mark.parametrize(
    "num_classes",
    [3, 10],
)
@pytest.mark.parametrize(
    "num_layers",
    [
        [1, 2, 3, 4],
        [
            2,
            2,
            2,
        ],
    ],
)
@pytest.mark.parametrize(
    "activation",
    [ActivationType.tanh, ActivationType.gelu],
)
def test_inception(shape, num_classes, num_layers, bottleneck_channels, expansion, activation):
    x = create_input(shape)
    m = DenseNet(
        shape[1],
        num_classes=num_classes,
        num_layers=num_layers,
        bottleneck_channels=bottleneck_channels,
        activation=activation,
        expansion=expansion,
    )
    out = m(x)
    assert list(out.shape) == [shape[0], num_classes]
