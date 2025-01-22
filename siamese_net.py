import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import torchvision
import random
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0,1) 
        
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])

test = torchvision.datasets.ImageFolder(r"C:\Users\mehme\OneDrive\Desktop\pytorch\siamese_net\data\faces\testing")
test = SiameseNetworkDataset(test,transform)
test = DataLoader(test,1,True)

train = torchvision.datasets.ImageFolder(r"C:\Users\mehme\OneDrive\Desktop\pytorch\siamese_net\data\faces\training")
train = SiameseNetworkDataset(train,transform)
train = DataLoader(train,64,True)

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self,dist,label):
      loss_contrastive = torch.mean((1-label) * torch.pow(dist,2) +
                                    (label) * torch.pow(torch.clamp(self.margin - dist,min=0.0),2))
      
      return loss_contrastive

class ConvNet(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channel,96,11,4)
        self.mp1 = nn.MaxPool2d(3,2)
        self.conv2 = nn.Conv2d(96,256,5)
        self.mp2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(256,384,3)

        self.fc1 = nn.Linear(384,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,2)
    
    def forward(self,x):
        x = F.relu(self.mp1(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = F.relu(self.fc1(torch.flatten(x,1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SiameseNet(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.conv_net = ConvNet(in_channel).to(device)

    def forward(self,in1,in2):
        out1 = self.conv_net(in1)
        out2 = self.conv_net(in2)

        return F.pairwise_distance(out1,out2,keepdim = True)

lr = 0.0005
epochs = 100

siamese_net = SiameseNet(1).to(device)

criterion = ContrastiveLoss().to(device)
optimizer = torch.optim.Adam(siamese_net.parameters(),lr)

for epoch in range(1,epochs+1):
    for imgs1,imgs2,labels in train:
        imgs1,imgs2 = imgs1.to(device),imgs2.to(device)
        labels = labels.to(device)

        loss = criterion(siamese_net(imgs1,imgs2),labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"{epoch} EPOCH DONE. LOSS : {loss}")
