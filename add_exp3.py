### Nikhil Jamdade Penn ID: 56849791

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision

class Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X = np.load('train_images.npy')
        y = np.load('train_labels.npy')  
        
#        X = X.astype(float)
#        i = 0
#        Y= np.zeros((10000,3,22,22))
#        for image in X:
#            meanv = np.mean(image)
#            stdev = np.std(image)
#            image = image - meanv
#            image = np.divide(image,stdev)  
#            image=np.flip(image,1)
#            image=np.flip(image,0)
#            image=(image[:,5:27,5:27])
#            Y[i]=image
#            i=i+1
            
        
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).long()
        

    def __len__(self):
        
        return self.len

    def __getitem__(self, idx):
        
        return self.x_data[idx], self.y_data[idx]


class Dataset_test(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X = np.load('test_images.npy')
        y = np.load('test_labels.npy')  
       
#        X = X.astype(float)
#        i = 0
#        Y= np.zeros((2000,3,22,22))
#        for image in X:
#            meanv = np.mean(image)
#            stdev = np.std(image)
#            image = image - meanv
#            image = np.divide(image,stdev)  
#            image=np.flip(image,1)
#            image=np.flip(image,0)
#            image=(image[:,5:27,5:27])
#            Y[i]=image
#            i=i+1
            
        
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).long()
        
    def __len__(self):
        
        return self.len

    def __getitem__(self, idx):
        
        return self.x_data[idx], self.y_data[idx]



class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        
        x = self.features(x)
        #print(x)
        x = x.view(x.size(0), 256 * 1 * 1)
        
        x = self.classifier(x)
        #print(x)
        return x

# Specify the newtork architecture
net = AlexNet()

# Specify the training dataset
dataset = Dataset()


# Specify the testing dataset
dataset_test=Dataset_test()


#Specify the testing dataset
train_loader = DataLoader(dataset=dataset,batch_size=128,shuffle=True)

test_loader= DataLoader(dataset=dataset_test,batch_size=128,shuffle=True)


# Visualize the dataset
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.title('Visualize the dataset')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

#get some random testing images 
dataiter1=iter(test_loader)
img,leb=dataiter1.next()

# show images
#imshow(torchvision.utils.make_grid(images))
#imshow(torchvision.utils.make_grid(img))



# Specify the loss function
#criterion = nn.BCELoss()
criterion=nn.CrossEntropyLoss()

# Specify the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

max_epochs = 50

loss_np = np.zeros((max_epochs))
accuracy = np.zeros((max_epochs))

for epoch in range(max_epochs):
    
    correct = 0
    for i, data in enumerate(train_loader, 0):
        
        # Get inputs and labels from data loader 
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        
        # Feed the input data into the network 
        y_pred = net(inputs)
        
    
        #print(y_pred)
        # Calculate the loss using predicted labels and ground truth labels
        loss = criterion(y_pred, labels)
        
        
        #print("epoch: ", epoch, "loss: ", loss.data[0])
        
        # zero gradient
        optimizer.zero_grad()
        
        # backpropogates to compute gradient
        loss.backward()
        
        # updates the weghts
        optimizer.step()
        
        # convert predicted laels into numpy
        y_pred_np = y_pred.data.numpy()
        #print(y_pred_np)
        # calculate the training accuracy of the current model
        #pred_np = np.where(y_pred_np>0.5, 1, 0) 
        pred_np=np.argmax(y_pred_np, axis=1)
        pred_np = pred_np.reshape(len(labels),1)

        label_np = labels.data.numpy().reshape(len(labels),1)
            
        
        
        
        for j in range(y_pred_np.shape[0]):
            if pred_np[j,:] == label_np[j,:]:
                correct += 1
        
        #accuracy[epoch] = float(correct)/float(len(label_np))
        
        #loss_np[epoch] = loss.data.numpy()
        loss_np[epoch] += loss.data.numpy()
    accuracy[epoch]= float(correct)/float(10000)
    loss_np[epoch] = loss_np[epoch]/float(len(train_loader))
    print("epoch: ", epoch, "loss: ", loss_np[epoch])


print("final training accuracy: ", accuracy[max_epochs-1])

epoch_number = np.arange(0,max_epochs,1)

# Plot the loss over epoch
plt.figure()
plt.plot(epoch_number, loss_np)
plt.title('loss over epoches')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss')

# Plot the training accuracy over epoch
plt.figure()
plt.plot(epoch_number, accuracy)
plt.title('training accuracy over epoches')
plt.xlabel('Number of Epoch')



# Code for test accuracy and loss 

correct = 0
total = 0
for data in test_loader:
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    y_pred = net(inputs)
    y_pred_np = y_pred.data.numpy()
    pred_np=np.argmax(y_pred_np, axis=1)
    pred_np = pred_np.reshape(len(labels),1)
    label_np = labels.data.numpy().reshape(len(labels),1)
            
    for j in range(y_pred_np.shape[0]):
            if pred_np[j,:] == label_np[j,:]:
                correct += 1
        
accuracy= float(correct)/float(2000)

print("final testing accuracy: ", accuracy)