from JupyterTest.Src.Net import Net, Net1
from torch import optim
import cv2
import torch
import torchvision

image=cv2.imread(r'D:\Code\Test\Self\PytorchTest\JupyterTest\Image\1.png')

imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


cv2.threshold(imageGray,125,255,type=cv2.THRESH_BINARY_INV,dst=imageGray)

# opsize=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(2,2))

# cv2.morphologyEx(imageGray,cv2.MORPH_CLOSE,opsize,imageGray)

input=cv2.resize(imageGray,(28,28))

cv2.imshow('dfs',input)

cv2.waitKey(0)

network = Net1()

p=torch.load('JupyterTest\Moudle\moudle6.pth')

network.load_state_dict(p)


tr=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

input=tr(input).view(-1,1,28,28)

output=network(input)

print(output)

output=output.argmax(1)

print(output)