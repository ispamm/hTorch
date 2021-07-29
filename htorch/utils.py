import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import re
import sys
from .functions import *
import torch.fx

grayscale = torchvision.transforms.Grayscale(num_output_channels=1)


def convert_data_for_quaternion(batch):
    """
    converts batches of RGB images in 4 channels for QNNs
    """
    assert all(batch[i][0].size(0) == 3 for i in range(len(batch)))
    inputs, labels = [], []
    for i in range(len(batch)):
        inputs.append(torch.cat([batch[i][0], grayscale(batch[i][0])], 0))
        labels.append(batch[i][1])

    return torch.stack(inputs), torch.LongTensor(labels)


# does not find an application yet
def apply_quaternion_gradient(model, layers):
    """
    hooks real-valued gradients and transforms them into one for 
    quaternion gradient descent
    
    @type model: nn.Module
    """

    for n, ((_, layer), parameter) in enumerate(zip(model.named_children(), model.parameters())):

        layer_name = re.match("^\w+", str(layer)).group()
        if layer_name in layers and len(parameter.shape) > 1 and n != 1:
            parameter.register_hook(to_conj)

    return model


@torch.fx.wrap
def check_shapes(x):
    if x.dim() in [3, 5]:
        x = torch.cat([*x.chunk()], 2).squeeze()
    return x


def convert_to_quaternion(Net, verbose=False, spinor=False):
    """
    converts a real_valued initialized Network to a quaternion one
    
    @type Net: nn.Module
    @type verbose: bool
    @type spinor: bool
    """
    last_module = len([mod for mod in Net.children()])
    layers = ["Linear", "Conv1d", "Conv2d", "Conv3d",
              "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]
    for n, (name, layer) in enumerate(Net.named_children()):
        layer_name = re.match("^\w+", str(layer)).group()

        if n != last_module - 1:
            if layer_name in layers[1:]:

                params = re.findall("(?<!\w)\d+(?<=\w)", str(layer))
                in_features, out_features, kernel_size, stride = \
                    int(params[0]), int(params[1]), (int(params[2]), int(params[3])), (int(params[4]), int(params[5]))

                assert in_features % 4 == 0, "number of in_channels must be divisible by 4"
                assert out_features % 4 == 0, "number of out_channels must be divisible by 4"

                init_func = initialize_conv
                args = (in_features // 4, out_features // 4, kernel_size)

            elif layer_name == layers[0]:

                params = re.findall("(?<==)\w+", str(layer))
                in_features, out_features, bias = int(params[0]), int(params[1]), bool(params[2])

                assert in_features % 4 == 0, "number of in_channels must be divisible by 4"
                assert out_features % 4 == 0, "number of out_channels must be divisible by 4"

                init_func = initialize_linear
                args = (in_features // 4, out_features // 4)

            else:
                continue

            quaternion_weight = init_func(*args)

            if spinor:
                weight = quaternion_weight._real_rot_repr
            else:
                weight = quaternion_weight._real_repr

            getattr(Net, name).weight = nn.Parameter(weight)
            if getattr(Net, name).bias != None:
                getattr(Net, name).bias = nn.Parameter(torch.zeros(out_features))

            traced = torch.fx.symbolic_trace(layer)

            for node in traced.graph.nodes:
                if node.op == 'placeholder':
                    with traced.graph.inserting_after(node):
                        new_node = traced.graph.call_function(
                            check_shapes, args=(node,))

                if any(lay in node.name for lay in ["conv", "lin"]):
                    with traced.graph.inserting_before(node):
                        all_nodes = [node for node in traced.graph.nodes]
                        new_node = traced.graph.call_function(node.target,
                                                              (all_nodes[1], *node.args[1:]), node.kwargs)
                        node.replace_all_uses_with(new_node)
                    traced.graph.erase_node(node)

                if node.op == 'output':
                    all_nodes = [node for node in traced.graph.nodes]
                    with traced.graph.inserting_before(node):
                        new_node = traced.graph.call_function(
                            Q, args=(node.prev,))
                        node.replace_all_uses_with(new_node)
                    traced.graph.erase_node(node)
                    with traced.graph.inserting_after(node):
                        new_node = traced.graph.output(node.prev, )

            if verbose:
                print("-" * 20, layer_name, "-" * 20, sep="\n")
                print(torch.fx.GraphModule(layer, traced.graph))

            traced.graph.lint()
            setattr(Net, name, torch.fx.GraphModule(layer, traced.graph))

    return Net
