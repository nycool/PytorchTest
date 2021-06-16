import cv2
import torch
from torch import nn
from torch._C import PyTorchFileReader
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision

dataSet=datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_data=torch.utils.data.DataLoader(dataSet,64,True)

test_data=torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.137,),(0.3081,))
                   ])))




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)


        self.moudle=nn.Sequential(
            nn.Conv2d(1,10,3), 
            nn.MaxPool2d(2),
            nn.ReLU(), 
            nn.Conv2d(10,20,3), 
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            nn.ReLU(), 
        )

    def forward(self, x):
       
        x=self.moudle(x)
   
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=0.01)
lossF=torch.nn.CrossEntropyLoss()

# for epoch in range(1000):
#     network.train()
#     for batch_idx,(image,target) in enumerate(train_data):
#         optimizer.zero_grad()
#         output=network(image)
#         loss=lossF(output,target)
#         loss.backward()
#         optimizer.step()

#         if batch_idx%100==0:
#             print(loss.item())
#             if loss.item()<0.02:
#                 torch.save(network.state_dict(),'../moudle.pth')

p=torch.load('JupyterTest/moudle1.pth')

network.load_state_dict(p)


import cv2

image=cv2.imread('/home/ncl/Src/JupyterTest/test8.png')

imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.threshold(imageGray,125,255,type=cv2.CV_8UC1,dst=imageGray)

input=cv2.resize(imageGray,(28,28))

tr=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

print(input.shape)

input=tr(input).view(-1,1,28,28)

output=network(input)

output=output.argmax(1)

print(output)

# for image,target in (test_data):
#     print("target: {}".format(target))
#     output=network(image)
#     output=torch.argmax(output,1)
#     print("output: {}".format(output))
#     print(target==output)

