import torch
import torchvision
from types import MethodType
#from .model import Identity

def get_torchvision_model(name, torchvision_params = {}, in_channels = None, out_features=None, out_as_image = False, keep_image_size = False, **kwargs):
    module = get_torchvision_model_split_name(name)(**torchvision_params)
    if in_channels is not None: change_in_channels(name, module, in_channels)
    if out_as_image:
        change_conv_out_features(name, module, out_features)
        if keep_image_size: change_all_paddings(name, module)
    elif out_features is not None: change_fc_out_features(name, module, out_features)
    # torchvision_model.fc = torch.nn.Linear(torchvision_model.fc.in_features, output_size)
    return module

def get_torchvision_model_split_name(name):
    name = name.split(".")
    app = torchvision.models
    for i in range(len(name)):
        app = getattr(app,name[i])
    return app

# def load_torchvision_model(model_cfg, path):
#     torchvision_model = getattr(torchvision.models,model_cfg["name"])(**model_cfg["torchvision_params"])
#     torchvision_model.fc = torch.nn.Linear(torchvision_model.fc.in_features, model_cfg["output_size"])
#     model = BaseNN.load_from_checkpoint(path, model=torchvision_model, **model_cfg)
#     return model

def change_in_channels(name, module, in_channels):
    if "resnet" in name:
        module_section = module
        attr_name = "conv1"
    elif "squeezenet" in name:
        module_section = module.features
        attr_name = "0"
    elif "deeplab" in name:
        module_section = getattr(module.backbone,"0")
        attr_name = "0"
    else:
        raise NotImplementedError("Model name",name)
    
    current_conv = getattr(module_section, attr_name)
    setattr(module_section,attr_name,type(current_conv)(in_channels = in_channels,
                                    out_channels = current_conv.out_channels,
                                    kernel_size = current_conv.kernel_size,
                                    stride = current_conv.stride,
                                    padding = current_conv.padding,
                                    bias = [True,False][current_conv.bias is None]))

def change_conv_out_features(name, module, out_features=None):
    if "resnet" in name:
        # drop last layer
        module._forward_impl = MethodType(resnet_forward_impl, module)
        del module.avgpool
        del module.fc
        
        module_section = getattr(module.layer4,"1")
        attr_name = "conv2"
    elif "squeezenet" in name:
        # drop last layer
        module.forward = lambda x: module.features(x) #MethodType(lambda x: module.features(x), module)
        del module.classifier
    
        module_section = getattr(module.features,"12")
        attr_name = "expand3x3"
    elif "deeplab" in name:
        module_section = module.classifier
        attr_name = "4"
    else:
        raise NotImplementedError("Model name",name)
    
    current_conv = getattr(module_section,attr_name)

    setattr(module_section,attr_name,type(current_conv)(in_channels = current_conv.in_channels,
                                        out_channels = [out_features,current_conv.out_channels][out_features is None],
                                        kernel_size = current_conv.kernel_size,
                                        stride = current_conv.stride,
                                        padding = current_conv.padding,
                                        bias = [True,False][current_conv.bias is None]))


def change_fc_out_features(name, module, out_features):
    if "resnet" in name:
        module_section = module
        attr_name = "fc"
    elif "squeezenet" in name:
        module_section = module.classifier
        attr_name = "1"
    else:
        raise NotImplementedError("Model name",name)

    current_fc = getattr(module_section, attr_name)
    if "resnet" in name:
        setattr(module_section,attr_name,type(current_fc)(in_features = current_fc.in_features,
                                                          out_features = out_features,
                                                          bias = [True,False][current_fc.bias is None]) #is there a better way to have bias True/False?
                                                        )
    elif "squeezenet" in name: #fc is actually a conv
        setattr(module_section,attr_name,type(current_fc)(in_channels = current_fc.in_channels,
                                        out_channels = out_features,
                                        kernel_size = current_fc.kernel_size,
                                        stride = current_fc.stride,
                                        padding = current_fc.padding,
                                        bias = [True,False][current_fc.bias is None]))
    else:
        raise NotImplementedError("Model name",name)

def change_all_paddings(name, module):
    # if "resnet" in name:
    #     pass
    if "squeezenet" in name:
        for i in range(1,13):
            current_conv = getattr(module.features,str(i))
            if isinstance(current_conv,torch.nn.Conv2d) or isinstance(current_conv,torch.nn.MaxPool2d):
                current_conv.padding = "same"
                current_conv.stride = 1 #cause with padding same stride needs to be 1
    else:
        raise NotImplementedError("Model name",name)

def resnet_forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    # See note [TorchScript super()]
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


# def get_torchvision_model(name, torchvision_params = {}, input_channels = None, output_size=1, **kwargs):
#     torchvision_model = getattr(torchvision.models,name)(**torchvision_params)
#     if input_channels is not None:
#         torchvision_model.conv1 = torch.nn.Conv2d(torchvision_model.fc.in_features, output_size)
    
#     torchvision_model.fc = torch.nn.Linear(torchvision_model.fc.in_features, output_size)
#     return torchvision_model

# def load_torchvision_model(model_cfg, path):
#     torchvision_model = getattr(torchvision.models,model_cfg["name"])(**model_cfg["torchvision_params"])
#     torchvision_model.fc = torch.nn.Linear(torchvision_model.fc.in_features, model_cfg["output_size"])
#     model = BaseNN.load_from_checkpoint(path, model=torchvision_model, **model_cfg)
#     return model