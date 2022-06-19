'''
    modified from original to include required image transforms,
    and to enable loading the full checkpoints.
'''
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torchvision
import resnet
from main_vicreg import Projector

dependencies = ["torch", "torchvision"]

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform

class VICRegExtractor(torch.nn.Module):
    def __init__(self, arch='resnet50', mlp='8192-8192-8192'):
        super().__init__()

        self.num_features = int(mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[arch](
            zero_init_residual=True
        )
        self.projector = Projector(mlp, self.embedding)

    def forward(self, x, y=None):
        x = self.projector(self.backbone(x))
        if y is None:
            return x
        
        y = self.projector(self.backbone(y))

        return x,y
    
def resnet50_vicreg_backbone(pretrained=True, **kwargs):
    model, _ = resnet.resnet50(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicreg/resnet50.pth",
            map_location="cpu",
            file_name="resnet50-c843e76524.pth",
            check_hash=True
        )
        model.load_state_dict(state_dict, strict=True)
        model.hashid = 'c843e76524'
        
    transform = _transform()
    
    return model, transform


def resnet50w2_vicreg_backbone(pretrained=True, **kwargs):
    model, _ = resnet.resnet50x2(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicreg/resnet50x2.pth",
            map_location="cpu",
            file_name="resnet50x2-65af82c4ad.pth",
            check_hash=True
        )
        model.load_state_dict(state_dict, strict=True)
        model.hashid = "65af82c4ad"
        
    transform = _transform()
    
    return model, transform


def resnet200w2_vicreg_backbone(pretrained=True, **kwargs):
    model, _ = resnet.resnet200x2(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/vicreg/resnet200x2.pth",
            map_location="cpu",
            file_name="resnet200x2-b8eeaf14f5.pth",
            check_hash=True
        )
        model.load_state_dict(state_dict, strict=True)
        model.hashid = "b8eeaf14f5"
        
    transform = _transform()
    
    return model, transform

# ---------------------------------------------------
#  To load the full checkpoint, we needed to make a copy of the weights
#  that excluded the optimizer, which was saved with a function defined
#  in the main scope (exclude_bias_and_norm in main_vicreg.py), which
#  means you can't load the official .pth files without this function
#  defined in the __main__ scope.
# ---------------------------------------------------

def resnet50_vicreg(pretrained=True, model_dir=None, **kwargs):
    model = VICRegExtractor(arch='resnet50', mlp='8192-8192-8192')
    
    if pretrained: 
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vicreg/resnet50_vicreg_fullckpt_weights_only.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
        model.hashid = "6cc22efa80"
    
    transform = _transform()
    
    return model, transform

def resnet50x2_vicreg(pretrained=True, model_dir=None, **kwargs):
    model = VICRegExtractor(arch='resnet50x2', mlp='8192-8192-8192')
    
    if pretrained: 
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vicreg/resnet50x2_vicreg_fullckpt_weights_only.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '6716990428'
     
    transform = _transform()
    
    return model, transform

def resnet200x2_vicreg(pretrained=True, model_dir=None, **kwargs):
    model = VICRegExtractor(arch='resnet200x2', mlp='8192-8192-8192')
    
    if pretrained: 
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://visionlab-pretrainedmodels.s3.amazonaws.com/model_zoo/vicreg/resnet200x2_vicreg_fullckpt_weights_only.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '9c58a5871c'
    
    transform = _transform()
    
    return model, transform


