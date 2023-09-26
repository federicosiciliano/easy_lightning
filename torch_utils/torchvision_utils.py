# Import necessary libraries
import torch  # Import the PyTorch library for deep learning
import torchvision  # Import torchvision for pre-trained models
from types import MethodType  # Import MethodType for method modification
from .model import BaseNN  # Import the BaseNN class from the model module

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
    module = get_torchvision_model_split_name(name)(**torchvision_params)
    
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
    else:
        raise NotImplementedError("Model name", name)

    current_conv = getattr(module_section, attr_name)

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
                                                            bias=[True, False][current_fc.bias is None]))
    elif "squeezenet" in name:
        setattr(module_section, attr_name, type(current_fc)(in_channels=current_fc.in_channels,
                                                            out_channels=out_features,
                                                            kernel_size=current_fc.kernel_size,
                                                            stride=current_fc.stride,
                                                            padding=current_fc.padding,
                                                            bias=[True, False][current_fc.bias is None]))
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
