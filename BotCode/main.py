import torch
import torch.nn as nn
import torchvision.transforms as transforms
from LoadData import *
from CNNnet import *
import sys
from torchvision import datasets
from pympler import asizeof
import random



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(40), transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])]) # m
    path='C:\\Users\\Alan Tian\\Documents\\cars\\Dataset\\SubsetVMMR'
    getpath = getpaths(path)
    all=getpath.item()
    cars=getpath.numtocar()

    train_size = int(0.8 * len(all)//5000)
    test_size = len(all)//5000 - train_size
    random.shuffle(all)
    all=all[:train_size+test_size]
    train, test= torch.utils.data.random_split(all, [train_size, test_size])

    temp = toDataset(train,transform)
    trainloader = DataLoader(temp, batch_size=8, shuffle=True, num_workers=8)
    print ("Preprocessing done for training set")
    testloader = DataLoader(toDataset(test,transform),batch_size = 8, shuffle = True, num_workers = 8)
    print ("Preprocessing done for testing set")
    model = Net()
#    print(model)

    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    n_epochs = 100
    train_loss_progress = []
    test_accuracy_progress = []

    model.train()

    for epoch in range(n_epochs):

        train_loss = 0.0
#yay you actually get to train now
        for data, target in trainloader:


            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(data)
               # calculate the loss
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        # if you have a learning rate scheduler - perform a its step in here
        scheduler.step()
        train_loss = train_loss / len(trainloader.dataset)
        train_loss_progress.append(train_loss)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

        # Run the test pass:
        correct = 0
        total = 0
        model.eval()  # prep model for testing

        with torch.no_grad():
            for data, target in testloader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        test_accuracy_progress.append(100 * correct / total)
        print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
  #      print (train_loss_progress)
   #     print (test_accuracy_progress)