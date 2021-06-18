from torch import nn
import torch.nn.functional as F
import cv2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1000, 300)
        self.fc2 = nn.Linear(300, 10)
        self.moudle=nn.Sequential(
            nn.Conv2d(1,20,5), 
            nn.MaxPool2d(2),
            nn.ReLU(), 
            nn.Conv2d(20,40,3), 
            nn.MaxPool2d(2),
            nn.Dropout2d(inplace=True),
            nn.ReLU(), 
        )

    def forward(self, x):
       
        x=self.moudle(x)

        

        x = x.view(-1, 1000)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.fc1 = nn.Linear(640, 300)
        self.fc2 = nn.Linear(300, 10)
        self.moudle=nn.Sequential(
            nn.Conv2d(1,20,5), 
            nn.MaxPool2d(2),
            nn.ReLU(), 
            nn.Conv2d(20,40,5), 
            nn.MaxPool2d(2),
            nn.Dropout2d(inplace=True),
            nn.ReLU(), 
        )

    def forward(self, x):
       
        x=self.moudle(x)

        x = x.view(-1, 640)


        x=self.fc1(x)

        print(x)
        
        x = F.relu(x)

        print(x)

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)