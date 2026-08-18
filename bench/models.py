"""Reference-architecture construction, shared by every bench script.

The torchvision ImageNet stem is catastrophic on 32x32 inputs: conv1 is
7x7 stride 2 and is followed by a 3x3 stride-2 maxpool, so 32x32 is
crushed to 8x8 before layer1 runs at all. Measured consequence on
resnet18/CIFAR:

    layer1   148K params ( 1.3%) on 8x8
    layer2   526K params ( 4.7%) on 4x4
    layer3  2.10M params (18.8%) on 2x2
    layer4  8.39M params (75.1%) on 1x1   <-- 75% of the model on ONE pixel

94% of the parameters end up on <=2x2 maps. That is not capacity, it is
overfitting surface, and it caps accuracy around 87-88% where a correctly
stemmed CIFAR ResNet-18 reaches ~95%. It also manufactures enormous
artificial pruning slack, which would let ANY pruning method "discover"
a large parameter reduction with no accuracy loss — an artifact of the
broken stem, not evidence for the method.

`cifar_stem=True` applies the standard CIFAR surgery: 3x3 stride-1 conv,
no maxpool. Use it for every CIFAR experiment, on BOTH arms.
"""

import torch.nn as nn
import torchvision


def build_model(arch: str, num_classes: int, cifar_stem: bool = True) -> nn.Module:
    fn = {"resnet18": torchvision.models.resnet18,
          "vgg16_bn": torchvision.models.vgg16_bn}[arch]
    model = fn(num_classes=num_classes)

    if cifar_stem and arch.startswith("resnet"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    return model


def describe(model: nn.Module, arch: str, input_hw: int = 32) -> dict:
    import torch
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spatial = {}
    if arch.startswith("resnet"):
        hooks, mods = [], dict(model.named_modules())
        for n in ("layer1", "layer2", "layer3", "layer4"):
            if n in mods:
                hooks.append(mods[n].register_forward_hook(
                    lambda m, i, o, n=n: spatial.__setitem__(n, tuple(o.shape[2:]))))
        was_training = model.training
        model.eval()
        with torch.no_grad():
            dev = next(model.parameters()).device
            model(torch.zeros(1, 3, input_hw, input_hw, device=dev))
        model.train(was_training)
        for h in hooks:
            h.remove()
    return {"params": params, "spatial": spatial}
