# Import necessary libraries
import torch  # Import the PyTorch library for deep learning
import torchvision  # Import torchvision for pre-trained models
from types import MethodType  # Import MethodType for method modification
from .model import BaseNN  # Import the BaseNN class from the model module
from copy import deepcopy  # Import deepcopy for model copying

# Function to get a pre-trained TorchVision model
def get_torchvision_model(name, torchvision_params={}, in_channels=None, out_features=None, out_as_image=False, keep_image_size=False, **kwargs):
    """
    Get a pre-trained TorchVision model with optional modifications.

    Parameters:
    - name: Name of the TorchVision model.
    - torchvision_params: Parameters for the TorchVision model.
    - in_channels: Number of input channels (optional).
    - out_features: Number of output features (optional).
    - out_as_image: Modify the model for image output (optional).
    - keep_image_size: Keep image size during modifications (optional).
    - kwargs: Additional keyword arguments.

    Returns:
    - module: The modified TorchVision model.
    """
    # Create the base TorchVision model
    # module = get_torchvision_model_split_name(name)(**torchvision_params)
    module = torchvision.models.get_model(name, **torchvision_params)

    # Modify the model if in_channels is specified
    if in_channels is not None:
        change_in_channels(name, module, in_channels)
    
    # Modify the model for image output if out_as_image is True
    if out_as_image:
        change_conv_out_features(name, module, out_features)
        if keep_image_size:
            change_all_paddings(name, module)
    # Modify the model if out_features is specified
    elif out_features is not None:
        change_fc_out_features(name, module, out_features)
    
    return module

# Function to split and get a TorchVision model by name
def get_torchvision_model_split_name(name):
    """
    Split and get a TorchVision model by name.

    Parameters:
    - name: Name of the TorchVision model.

    Returns:
    - app: The TorchVision model.
    """
    name = name.split(".")
    app = torchvision.models
    for i in range(len(name)):
        app = getattr(app, name[i])
    return app

# Function to change the number of input channels in the model
def change_in_channels(name, module, in_channels):
    """
    Change the number of input channels in the model.

    Parameters:
    - name: Name of the model.
    - module: The model to be modified.
    - in_channels: Number of input channels.

    Returns:
    - None
    """
    if "resnet" in name:
        module_section = module
        attr_name = "conv1"
    elif "squeezenet" in name:
        module_section = module.features
        attr_name = "0"
    elif "deeplab" in name:
        module_section = getattr(module.backbone, "0")
        attr_name = "0"
    elif name in ["mc3_18"]:
        module_section = module.stem
        attr_name = "0"
    else:
        raise NotImplementedError("Model name", name)

    current_conv = getattr(module_section, attr_name)
    setattr(module_section, attr_name, type(current_conv)(in_channels=in_channels,
                                                          out_channels=current_conv.out_channels,
                                                          kernel_size=current_conv.kernel_size,
                                                          stride=current_conv.stride,
                                                          padding=current_conv.padding,
                                                          bias=[True, False][current_conv.bias is None]))

# Function to change the output features of convolutional layers in the model
def change_conv_out_features(name, module, out_features=None):
    """
    Change the output features of convolutional layers in the model.

    Parameters:
    - name: Name of the model.
    - module: The model to be modified.
    - out_features: Number of output features (optional).

    Returns:
    - None
    """
    if "resnet" in name:
        # Drop the last layer and modify convolutional layers
        module._forward_impl = MethodType(resnet_forward_impl, module)
        del module.avgpool
        del module.fc

        module_section = getattr(module.layer4, "1")
        attr_name = "conv2"
    elif "squeezenet" in name:
        # Drop the last layer and modify convolutional layers
        module.forward = lambda x: module.features(x)
        del module.classifier

        module_section = getattr(module.features, "12")
        attr_name = "expand3x3"
    elif "deeplab" in name:
        module_section = module.classifier
        attr_name = "4"
    elif name in ["mc3_18"]:
        # Drop the last layer and modify convolutional layers
        module.forward = MethodType(video_resnet_forward, module)
        del module.avgpool
        del module.fc

        module_section = getattr(getattr(module.layer4, "1"),"conv2")
        attr_name = "0"
    else:
        raise NotImplementedError("Model name", name)

    current_conv = getattr(module_section, attr_name)

    if name in ["mc3_18"]:
        pass
        # setattr(module_section, attr_name, type(current_conv)(in_planes=current_conv.in_channels,
        #                                                     out_planes=[out_features, current_conv.out_channels][out_features is None],
        #                                                     stride=current_conv.stride[-1], #[-1] because internally the layer does stride=(1, stride, stride),
        #                                                     padding=current_conv.padding[-1] #[-1] because internally the layer does padding=(0, padding, padding),
        #                                                     ))
        # batch_norm_name = str(int(attr_name)+1)
        # batch_norm_layer = getattr(module_section, batch_norm_name)
        # setattr(module_section, batch_norm_name, type(batch_norm_layer)(num_features=out_features,
        #                                                                 eps=batch_norm_layer.eps,
        #                                                                 momentum=batch_norm_layer.momentum,
        #                                                                 affine=batch_norm_layer.affine,
        #                                                                 track_running_stats=batch_norm_layer.track_running_stats))
        # TODO: now, it gives error because residual has different size
    else:
        setattr(module_section, attr_name, type(current_conv)(in_channels=current_conv.in_channels,
                                                            out_channels=[out_features, current_conv.out_channels][out_features is None],
                                                            kernel_size=current_conv.kernel_size,
                                                            stride=current_conv.stride,
                                                            padding=current_conv.padding,
                                                            bias=[True, False][current_conv.bias is None]))

# Function to change the output features of fully connected layers in the model
def change_fc_out_features(name, module, out_features):
    """
    Change the output features of fully connected layers in the model.

    Parameters:
    - name: Name of the model.
    - module: The model to be modified.
    - out_features: Number of output features.

    Returns:
    - None
    """
    if "resnet" in name:
        module_section = module
        attr_name = "fc"
    elif "squeezenet" in name:
        module_section = module.classifier
        attr_name = "1"
    else:
        raise NotImplementedError("Model name", name)

    current_fc = getattr(module_section, attr_name)
    if "resnet" in name:
        setattr(module_section, attr_name, type(current_fc)(in_features=current_fc.in_features,
                                                            out_features=out_features,
                                                            bias=current_fc.bias is not None))
    elif "squeezenet" in name:
        setattr(module_section, attr_name, type(current_fc)(in_channels=current_fc.in_channels,
                                                            out_channels=out_features,
                                                            kernel_size=current_fc.kernel_size,
                                                            stride=current_fc.stride,
                                                            padding=current_fc.padding,
                                                            bias=current_fc.bias is not None))
    else:
        raise NotImplementedError("Model name", name)

# Function to change padding in convolutional layers to "same"
def change_all_paddings(name, module):
    """
    Change padding in convolutional layers to "same".

    Parameters:
    - name: Name of the model.
    - module: The model to be modified.

    Returns:
    - None
    """
    if "squeezenet" in name:
        for i in range(1, 13):
            current_conv = getattr(module.features, str(i))
            if isinstance(current_conv, torch.nn.Conv2d) or isinstance(current_conv, torch.nn.MaxPool2d):
                current_conv.padding = "same"
                current_conv.stride = 1  # Cause with padding same stride needs to be 1
    else:
        raise NotImplementedError("Model name", name)

# Custom forward method for ResNet
def resnet_forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward method for ResNet.

    Parameters:
    - self: The ResNet model.
    - x: Input tensor.

    Returns:
    - x: Output tensor.
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = self.fc(x)
    return x

# Custom forward method for VideoResNet
def video_resnet_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Custom forward method for VideoResNet.

    Parameters:
    - self: The VideoResNet model.
    - x: Input tensor.

    Returns:
    - x: Output tensor.
    """
    x = self.stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    # x = self.avgpool(x)
    # # Flatten the layer to fc
    # x = x.flatten(1)
    # x = self.fc(x)
    return x



# Function to load a TorchVision model from a checkpoint
def load_torchvision_model(model_cfg, path):
    """
    Load a TorchVision model from a checkpoint.

    Parameters:
    - model_cfg: Configuration parameters for the model.
    - path: Path to the checkpoint file.

    Returns:
    - model: The loaded TorchVision model.
    """
    # Create an instance of the specified TorchVision model
    torchvision_model = getattr(torchvision.models, model_cfg["name"])(**model_cfg["torchvision_params"])
    
    # Replace the fully connected layer (fc) with a new one based on output_size
    torchvision_model.fc = torch.nn.Linear(torchvision_model.fc.in_features, model_cfg["output_size"])
    
    # Load the model from the checkpoint file using BaseNN.load_from_checkpoint
    # Note: You'll need to have the BaseNN class defined or import it here.
    # For this code to work, you should also have the appropriate model_cfg
    # that specifies the name and torchvision_params of the model.
    model = BaseNN.load_from_checkpoint(path, model=torchvision_model, **model_cfg)
    
    return model


def invert_model(model, example_datum, keep_order=False):
    """
    Invert a model.

    Parameters:
    - model: The model to be inverted.
    - example_datum: An example datum for the model.

    Returns:
    - inverted_model: The inverted model.
    """

    # Get the model's layers
    layers = model.named_children()
    #layers = [module for module in model.modules() if not isinstance(module, type(model))]

    current_input = example_datum.detach().clone()

    inverted_layers = torch.nn.ModuleList()
    for layer_name,layer in layers:
        if isinstance(layer, torch.nn.Sequential):
            inverted_layer = invert_model(layer, current_input)
        else:
            inverted_layer, current_input = invert_layer(layer, current_input, inverted_layers)
        inverted_layers.append(inverted_layer)
        current_input = layer(current_input) #could decrease computational cost by separating computations for sequential layers

    # Create a new model with the reversed layers
    # if keep_order:
    #     inverted_model = torch.nn.Sequential(*inverted_layers)
    # else:
    #     
    inverted_model = torch.nn.Sequential(*inverted_layers[::-1])
    
    return inverted_model

def invert_layer(layer, current_input, inverted_layers=[]):
    if isinstance(layer, torch.nn.Linear):
        if len(current_input.shape) > 2:
            inverted_layers.append(torch.nn.Unflatten(1, current_input.shape[1:]))
            current_input = torch.flatten(current_input, 1)
        inverted_layer = torch.nn.Linear(in_features=layer.out_features,
                                            out_features=layer.in_features,
                                            bias=layer.bias is not None)
    elif isinstance(layer, torch.nn.MaxPool2d) or isinstance(layer, torch.nn.AvgPool2d) or isinstance(layer, torch.nn.AdaptiveAvgPool2d):
        inverted_layer = torch.nn.Upsample(size=current_input.shape[2:])
    elif isinstance(layer, torch.nn.Conv2d):
        rems = [(current_input.shape[2+i] + 2*layer.padding[i] - layer.dilation[i]*(layer.kernel_size[i]-1) - 1)%layer.stride[i] for i in range(2)]
        if rems[0] != 0 or rems[1] != 0:
            inverted_layers.append(torch.nn.Upsample(size=(current_input.shape[2], current_input.shape[3])))
        inverted_layer = torch.nn.ConvTranspose2d(in_channels=layer.out_channels,
                                                    out_channels=layer.in_channels,
                                                    kernel_size=layer.kernel_size,
                                                    stride=layer.stride,
                                                    padding=layer.padding,
                                                    output_padding=layer.output_padding,
                                                    bias=layer.bias is not None,
                                                    padding_mode=layer.padding_mode)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        inverted_layer = torch.nn.BatchNorm2d(num_features=current_input.shape[1],
                                                eps=layer.eps,
                                                momentum=layer.momentum,
                                                affine=layer.affine,
                                                track_running_stats=layer.track_running_stats)
    elif isinstance(layer, torch.nn.ReLU):
        inverted_layer = torch.nn.ReLU()
    elif isinstance(layer, torchvision.models.resnet.BasicBlock):
        inplanes, planes = layer.conv1.in_channels, layer.conv1.out_channels
        inverted_layer = torchvision.models.resnet.BasicBlock(inplanes=planes,
                                                                planes=inplanes,
                                                                stride=layer.stride,
                                                                downsample=deepcopy(layer.downsample),
                                                                norm_layer=type(layer.bn1))
        inverted_layer.conv1 = torch.nn.ConvTranspose2d(in_channels=inverted_layer.conv1.in_channels,
                                                            out_channels=inverted_layer.conv1.out_channels,
                                                            kernel_size=inverted_layer.conv1.kernel_size,
                                                            stride=inverted_layer.conv1.stride,
                                                            padding=inverted_layer.conv1.padding,
                                                            output_padding=inverted_layer.conv1.output_padding,
                                                            bias=inverted_layer.conv1.bias is not None,
                                                            padding_mode=inverted_layer.conv1.padding_mode)
        inverted_layer.conv2 = torch.nn.ConvTranspose2d(in_channels=inverted_layer.conv2.in_channels,
                                                            out_channels=inverted_layer.conv2.out_channels,
                                                            kernel_size=inverted_layer.conv2.kernel_size,
                                                            stride=inverted_layer.conv2.stride,
                                                            padding=inverted_layer.conv2.padding,
                                                            output_padding=inverted_layer.conv2.output_padding,
                                                            bias=inverted_layer.conv2.bias is not None,
                                                            padding_mode=inverted_layer.conv2.padding_mode)
        if inverted_layer.downsample is not None:
            inverted_layer.downsample[0] = torch.nn.ConvTranspose2d(in_channels=inverted_layer.downsample[0].out_channels,
                                                                        out_channels=inverted_layer.downsample[0].in_channels,
                                                                        kernel_size=inverted_layer.downsample[0].kernel_size,
                                                                        stride=inverted_layer.downsample[0].stride,
                                                                        padding=inverted_layer.downsample[0].padding,
                                                                        output_padding=inverted_layer.downsample[0].output_padding,
                                                                        bias=inverted_layer.downsample[0].bias is not None,
                                                                        padding_mode=inverted_layer.downsample[0].padding_mode)
            inverted_layer.downsample[1], _ = invert_layer(inverted_layer.downsample[1], current_input, inverted_layers)
        
        # inverted_layer = deepcopy(layer)
        # previous_input2 = previous_input.clone()
        # current_input2 = current_input.clone()
        # for children_layer_name, children_layer in layer.named_children():
        #     if children_layer_name == "downsample" and children_layer is not None:
        #         setattr(inverted_layer, children_layer_name, invert_model(layer.downsample, current_input, previous_input, keep_order=True))
        #         previous_input2 = current_input2
        #         current_input2 = children_layer(current_input) + current_input2
        #     else:
        #         print(children_layer_name, current_input2.shape, previous_input2.shape)
        #         new_layer, current_input2, previous_input2 = invert_layer(children_layer, current_input2, previous_input2, inverted_layers)
        #         setattr(inverted_layer, children_layer_name, new_layer)
        #         current_input2 = children_layer(current_input2)
    else:
        raise NotImplementedError("Layer type", type(layer))
    
    return inverted_layer, current_input