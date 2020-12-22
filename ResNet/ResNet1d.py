'''ResNet1d in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385


This is a copy-paste from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
I just changed 2d conv and batch norms by 1d ones
'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes//2, planes)
        #self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes,    planes, kernel_size=3, stride=1,      padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes//2, planes)
        #self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                    nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(self.expansion*planes)
                    )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv1d(in_planes, planes,                kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes//2, planes)
        #self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes,    planes,                kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes//2, planes)
        #self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes,    self.expansion*planes, kernel_size=1, bias=False)
        self.bn2 = nn.GroupNorm(self.expansion*planes//2, self.expansion*planes)
        #self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                    nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(self.expansion*planes)
                    )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear1 = nn.Linear(19968*block.expansion, 1024)
        self.linear2 = nn.Linear(1024, num_classes)

    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        #out = F.avg_pool1d(out, 4)
        out = F.avg_pool1d(out, 16)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)

        return out







class ResNet_Custom(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet_Custom, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear1 = nn.Linear(19968*block.expansion, 1024)
        self.linear2 = nn.Linear(1024 + 1, num_classes)

    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    

    def forward(self, x, external):
        out = F.relu(self.bn1(self.conv1(x)))
 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        #out = F.avg_pool1d(out, 4)
        out = F.avg_pool1d(out, 16)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = [out, external]
        out = torch.cat(out,dim=1)
        out = self.linear2(out)

        return out







def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])







# These architectures are for Nanopore Translocation Signal Features Prediction
# In our case we predict the number of translocation events inside a window in a trace
def ResNet10_Counter():
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=1)

def ResNet18_Counter():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)

def ResNet34_Counter():
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1)

def ResNet50_Counter():
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1)

def ResNet101_Counter():
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=1)

def ResNet152_Counter():
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=1)







# These architectures are for Nanopore Translocation Signal Features Prediction
# In our case we predict Average Duration and Amplitude inside a window of translocation events in a trace
def ResNet10_Custom():
    return ResNet_Custom(BasicBlock, [1, 1, 1, 1], num_classes=2)

def ResNet18_Custom():
    return ResNet_Custom(BasicBlock, [2, 2, 2, 2], num_classes=2)

def ResNet34_Custom():
    return ResNet_Custom(BasicBlock, [3, 4, 6, 3], num_classes=2)

def ResNet50_Custom():
    return ResNet_Custom(Bottleneck, [3, 4, 6, 3], num_classes=2)

def ResNet101_Custom():
    return ResNet_Custom(Bottleneck, [3, 4, 23, 3], num_classes=2)

def ResNet152_Custom():
    return ResNet_Custom(Bottleneck, [3, 8, 36, 3], num_classes=2)



















