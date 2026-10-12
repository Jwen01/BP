
import torch
from networks.VNet import VNet
from networks.newVnet import VNet as newVNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'net_factory using device: {device}')

def net_factory(net_type="unet", in_chns=1, class_num=4, mode="train"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).to(device)
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).to(device)
    elif net_type == "vnets" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True, has_residual=True).to(device)
    elif net_type == "vnets" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False, has_residual=True).to(device)
    return net