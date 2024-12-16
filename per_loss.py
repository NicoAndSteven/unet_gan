import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(9):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 13):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 22):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        self.slice4.add_module(str(22), nn.ReLU())
        for x in range(23, 31):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        return [h_relu3, h_relu4]


class Per_Loss(nn.Module):
    def __init__(self):

        super(Per_Loss, self).__init__()
        # self.vgg = Vgg19()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.contrast_weights = [1.0 / 2, 1.0]
        self.std = torch.tensor(
            [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False).cuda()
        self.mean = torch.tensor(
            [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False).cuda()

    def forward(self, clear_fake, clear_real, rain):

        clear_fake = clear_fake.sub(self.mean.detach()).div(self.std.detach())
        clear_real = clear_real.sub(self.mean.detach()).div(self.std.detach())
        rain = rain.sub(self.mean.detach()).div(self.std.detach())
        f_clear_fake, f_clear_real, f_rain \
            = self.vgg(clear_fake), self.vgg(clear_real), self.vgg(rain)

        per_loss = 0.0
        # per loss
        for i in range(1, len(self.contrast_weights) + 1):
            d_ap = self.l1(f_clear_fake[-i], f_clear_real[-i].detach())
            per_loss += self.contrast_weights[-i] * d_ap

        return per_loss
