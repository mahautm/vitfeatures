import torchvision
import timm
from torch import nn


def initialize_vision_module(name: str = "resnet50", pretrained: bool = False):
    print("initialize module", name)
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
        "inception": torchvision.models.inception_v3(pretrained=pretrained),
        "vgg11": torchvision.models.vgg11(pretrained=pretrained),
        "vit": timm.create_model("vit_base_patch16_384", pretrained=pretrained),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    if name in ["resnet50", "resnet101", "resnet152"]:
        n_features = model.fc.in_features
        fc_layer = model.fc
        model.fc = nn.Identity()

    elif name == "vgg11":
        n_features = model.classifier[6].in_features
        fc_layer = model.classifier[6]
        model.classifier[6] = nn.Identity()

    elif name == "inception":
        n_features = model.fc.in_features
        fc_layer = model.AuxLogits.fc
        model.AuxLogits.fc = nn.Identity()
        model.fc = nn.Identity()

    else:  # vit
        n_features = model.head.in_features
        fc_layer = model.head
        model.head = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in fc_layer.parameters():
            param.requires_grad = False
        model = (
            model.eval()
        )  # Mat : --> dropout blocked, as well as all other training dependant behaviors

    return model, fc_layer, n_features
