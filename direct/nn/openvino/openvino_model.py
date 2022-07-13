import io

import torch
from openvino.inference_engine import IECore
from openvino_extensions import get_extensions_path
from torch import nn


class InstanceNorm2dFunc(torch.autograd.Function):
    """This class is used as a simple wrapper over original InstanceNorm2dONNX function. Required for ONNX conversion."""

    @staticmethod
    def symbolic(g, cls, inp):
        """ONNX node definition for custom nodes"""
        c_scale = g.op("Constant", value_t=cls.scale_one)
        c_bias = g.op("Constant", value_t=cls.bias_zero)
        return g.op("InstanceNormalization", inp, c_scale, c_bias)

    @staticmethod
    def forward(ctx, cls, inp):  # pylint: disable=unused-argument
        """Fallback to origin custom function."""
        return cls.origin_forward(inp)


class InstanceNorm2dONNX(nn.InstanceNorm2d):
    """
    This is a support class which helps export network with InstanceNorm2d in ONNX format.
    """

    def __init__(self, num_features):
        """Inits InstanceNorm2dONNX

        Parameters
        ----------
        num_features: int
        """
        super().__init__(num_features)
        self.origin_forward = super().forward
        self.scale_one = torch.ones(num_features)
        self.bias_zero = torch.zeros(num_features)

    def forward(self, inp):
        """This is a helper function that calls InstanceNorm2dFunc wrapper methods.

        Parameters
        ----------
        inp: torch.Tensor
        """
        return InstanceNorm2dFunc.apply(self, inp).clone()


def convert_layer(model):
    """This function recursively replaces the InstanceNorm2d layer with the custom InstanceNorm2dONNX layer.

    Parameters
    ----------
    model: torch.nn.Module subclass
    """
    for name, l in model.named_children():
        layer_type = l.__class__.__name__
        if layer_type == "InstanceNorm2d":
            new_layer = InstanceNorm2dONNX(l.num_features)
            setattr(model, name, new_layer)
        else:
            convert_layer(l)


class OpenVINOModel(nn.Module):
    """OpenVINO implementation of RIM and UNet2d models"""

    def __init__(self, model):
        """Inits OpenVINOModel.

        Parameters
        ----------
        model: direct.nn.rim.rim.RIM or direct.nn.unet.unet_2d.Unet2d
        """
        super().__init__()
        self.inputs: tuple
        self.input_names: list
        self.model = model
        self.exec_net = None
        self.model_name = self.model.__class__.__name__

    def create_net(self, *args, **kwargs):
        """This function export PyTorch model to ONNX and create OpenVINO model.
        Parameters are the same as for the corresponding PyTorch model.
        """
        self.inputs = args

        if self.model_name == "RIM":
            self.input_names = ["input_image", "masked_kspace", "sampling_mask", "sensitivity_map"]
            output_names = ["cell_outputs", "previous_state"]
        elif self.model_name == "Unet2d":
            output_names = ["output"]
            self.input_names = ["masked_kspace"]
            if self.model.image_initialization == "sense":
                self.input_names.append("sensitivity_map")
        else:
            raise ValueError(f"The model is not supported by OpenVINO: {self.model_name}")

        for k in self.input_names:
            if k in kwargs:
                self.inputs += tuple([kwargs[k]])

        ie = IECore()
        ie.add_extension(get_extensions_path(), "CPU")

        convert_layer(self.model)

        with torch.no_grad():
            buf = io.BytesIO()
            torch.onnx.export(
                self.model,
                self.inputs,
                buf,
                opset_version=12,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                input_names=self.input_names,
                output_names=output_names,
            )

            net = ie.read_network(buf.getvalue(), b"", init_from_buffer=True)
            self.exec_net = ie.load_network(net, "CPU")

    def postprocess(self, res):
        """This function custom the output of an OpenVINO model to the output format of the original PyTorch model.

        Parameters
        ----------
        res: dict

        Returns
        -------
        Output is the same as the output of the corresponding PyTorch model.
        """
        if self.model_name == "RIM":
            out = ([torch.Tensor(res["cell_outputs"])], torch.Tensor(res["previous_state"]))
        elif self.model_name == "Unet2d":
            out = torch.Tensor(res["output"])
        return out

    def forward(self, *args, **kwargs):
        """Creates an OpenVINO model if it doesn't exist and computes forward pass of the model.

        Parameters
        ----------
        Parameters are the same as for the corresponding PyTorch model.

        Returns
        -------
        Output is the same as for the corresponding PyTorch model.
        """
        if self.exec_net is None:
            self.create_net(*args, **kwargs)

        ov_input = dict(zip(self.input_names, self.inputs))
        res = self.exec_net.infer(ov_input)

        return self.postprocess(res)
