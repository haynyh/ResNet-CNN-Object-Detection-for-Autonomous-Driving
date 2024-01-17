import torch
import torch.nn as nn

###############################################################################
# redefine conv layer
###############################################################################
def redefine_conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def redefine_conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


##############################################################################
# The implementation of Basic Block for reference
###############################################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = redefine_conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = redefine_conv3x3(planes, planes * BasicBlock.expansion)
        self.bn2 = nn.BatchNorm2d(planes * BasicBlock.expansion)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


##############################################################################
# Implementation of Bottleneck Block
###############################################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        ##############################################################
        ##############################################################
        
        self.conv1 = redefine_conv1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = redefine_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = redefine_conv1x1(planes, planes * Bottleneck.expansion)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)

        ###############################################################
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x

        ##############################################################
        ##############################################################
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        ###############################################################
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
