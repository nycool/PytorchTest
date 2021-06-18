from torch.utils.data import dataloader
import torchvision
from JupyterTest.Src.Net import Net, Net1
import cv2
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

dataSet=datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))


train_data=dataloader.DataLoader(dataset=dataSet,batch_size=64,shuffle=True)


test_data=dataloader.DataLoader(datasets.MNIST('../data', train=False,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.137,),(0.3081,))
                   ])),shuffle=True)



images, label = next(iter(train_data))
images_example = torchvision.utils.make_grid(images)
images_example = images_example.numpy().transpose(1,2,0) # 将图像的通道值置换到最后的维度，符合图像的格式
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]
images_example = images_example * std + mean
plt.imshow(images_example )
plt.show()



network = Net1()

lr=1e-5

optimizer = optim.SGD(network.parameters(), lr=lr,momentum=0.9)
lossF=torch.nn.NLLLoss()

# p=torch.load(r'D:\Code\Test\Self\PytorchTest\JupyterTest\Moudle\moudle8.pth')

# network.load_state_dict(p)


for epoch in range(6000):
    network.train()
    for batch_idx,(image,target) in enumerate(train_data):
        optimizer.zero_grad()
        output=network(image)
        loss=lossF(output,target)
        loss.backward()
        optimizer.step()

        if batch_idx%640==0:
            print(loss.item())
            # torch.save(network.state_dict(),r'D:\Code\Test\Self\PytorchTest\JupyterTest\Moudle\moudle8.pth')
            if(loss.item()<0.05):
                i=0;
                with torch.no_grad():
                    for image,target in (test_data):
                        output=network(image)
                        output=torch.argmax(output,1)

                        if(output!=target):
                            i+=1

                a=i/len(test_data)
                print(1-a)      
                       

 





    
