import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from torch.autograd import Variable

# https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

class Bottleneck(nn.Module): # ResNet
    # 순서 4
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__() # 64, 64, 1 -> 256, 64, 1 -> 256, 128, 2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False) # 200 * 200 -> 200 * 200
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # 200 * 200 -> 200 * 200
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, stride=1, bias=False)# 200 * 200 -> 200 * 200
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        # 순서 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 800*800 원본 이미지 입력 -> 400 * 400
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) # 400 * 400 -> 200 * 200

        # Bottom-up layers
        # 순서 2
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # num_blocks[0] = 2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # num_blocks[1] = 2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # num_blocks[2] = 2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # num_blocks[3] = 2

        # Top layer
        self.top_layer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth_layer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth_layer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers # 채널 수를 256로 맞추기 위한 필터
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)


    def _make_layer(self, block, planes, num_blocks, stride):
        # 순서 3
        strides = [stride] + [1] * (num_blocks-1) # [1, 1] -> [2, 1]
        print(strides)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride)) # Bottleneck layer 생성
            self.in_planes = planes * block.expansion
        # print(*layers)
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y): # x = p5, y = p4
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.


        '''

        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y # 256 * 25 * 25크기의 p5를 256 * 50 50으로 upsample하고 latlayer에 의해 256 * 50 *50 으로 바뀐 y랑 합체

    def forward(self, x):
        #Bottom-up
        print("Bottom-Up")
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        print("c1 : ", c1.shape)
        tensor_img_c1= self.tensor_to_image(c1)
        c2 = self.layer1(c1)
        tensor_img_c2 = self.tensor_to_image(c2)
        print("c2 : ", c2.shape)
        c3 = self.layer2(c2)
        tensor_img_c3 = self.tensor_to_image(c3)
        print("c3 : ", c3.shape)
        c4 = self.layer3(c3)
        tensor_img_c4 = self.tensor_to_image(c4)
        print("c4 : ", c4.shape)
        c5 = self.layer4(c4)
        tensor_img_c5 = self.tensor_to_image(c5)
        print("c5 : ", c5.shape)

        #Top-down
        p5 = self.top_layer(c5)
        print("p5 : ", p5.shape)
        tensor_img_p5 = self.tensor_to_image(p5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        print("p4 : ", p4.shape)
        tensor_img_p4 = self.tensor_to_image(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        print("p3 : ", p3.shape)
        tensor_img_p3 = self.tensor_to_image(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        print("p2 : ", p2.shape)
        tensor_img_p2 = self.tensor_to_image(p2)

        #Smooth
        p4 = self.smooth_layer1(p4)
        tensor_img_sp4 = self.tensor_to_image(p4)
        p3 = self.smooth_layer2(p3)
        tensor_img_sp3 = self.tensor_to_image(p3)
        p2 = self.smooth_layer3(p2)
        tensor_img_sp2 = self.tensor_to_image(p2)
        self.merge_tensor_image(tensor_img_c2, tensor_img_c3, tensor_img_c4, tensor_img_c5, tensor_img_p2, tensor_img_p3,
                                tensor_img_p4, tensor_img_p5, tensor_img_sp2, tensor_img_sp3, tensor_img_sp4)

        return p2, p3, p4, p5

    def tensor_to_image(self, tensor):
        tensor = tensor.squeeze(0)
        merged_tensor = tensor.mean(dim=0, keepdim=True).unsqueeze(0)
        return merged_tensor
        # plt.figure(figsize=(5, 5))
        # plt.imshow(merged_tensor.squeeze().detach().numpy())
        # plt.axis('off')
        # plt.show()

    def merge_tensor_image(self, c2, c3, c4, c5, p2, p3, p4, p5, sp2, sp3, sp4):
        fig = plt.figure(figsize=(10, 5))
        ax0_0 = fig.add_subplot(3, 4, 1)
        ax0_0.imshow(c2.squeeze().detach().numpy())
        ax0_1 = fig.add_subplot(3, 4, 2)
        ax0_1.imshow(c3.squeeze().detach().numpy())
        ax0_2 = fig.add_subplot(3, 4, 3)
        ax0_2.imshow(c4.squeeze().detach().numpy())
        ax0_3 = fig.add_subplot(3, 4, 4)
        ax0_3.imshow(c5.squeeze().detach().numpy())
        ax1_0 = fig.add_subplot(3, 4, 5)
        ax1_0.imshow(p2.squeeze().detach().numpy())
        ax1_1 = fig.add_subplot(3, 4, 6)
        ax1_1.imshow(p3.squeeze().detach().numpy())
        ax1_2 = fig.add_subplot(3, 4, 7)
        ax1_2.imshow(p4.squeeze().detach().numpy())
        ax1_3 = fig.add_subplot(3, 4, 8)
        ax1_3.imshow(p5.squeeze().detach().numpy())
        ax2_0 = fig.add_subplot(3, 4, 9)
        ax2_0.imshow(sp2.squeeze().detach().numpy())
        ax2_1 = fig.add_subplot(3, 4, 10)
        ax2_1.imshow(sp3.squeeze().detach().numpy())
        ax2_2 = fig.add_subplot(3, 4, 11)
        ax2_2.imshow(sp4.squeeze().detach().numpy())

        plt.axis('off')
        plt.show()




if __name__ == '__main__':
    image_path = './selfie.jpg'
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((800,800)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    save_image(image_tensor, 'b.png', nrow=8, normalize=True)
    # print(image_tensor.shape)
    print('image', image_tensor.shape)

    fpn = FPN(Bottleneck, [2,2,2,2])
    x = torch.randn((2, 3, 800, 800))
    x = fpn(image_tensor)



