#!/usr/bin/env JONY python
import torch
import torchvision
import torchvision.models as models
from torchvision.transforms import transforms
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve , auc


# Specify transforms using torchvision.transforms as transforms
transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Load in each dataset and apply transformations using
# the torchvision.datasets as datasets library
train_set = datasets.ImageFolder("/homedtic/ikoren/yalla/sel/torch/train", transform = transformations)
val_set = datasets.ImageFolder("/homedtic/ikoren/yalla/sel/torch/test", transform = transformations)



# Put into a Dataloader using torch library
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =4, shuffle=True)

# Get pretrained model using torchvision.models as models library
model = models.densenet161(pretrained=True)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False


# Create new classifier for model using torch.nn as nn library

# Initialize classifier
classifier_input = model.classifier.in_features

# number of classes
classes = train_set.classes
num_labels = len(classes) 

classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

# Replace default classifier with new classifier
model.classifier = classifier



# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device specified above
model.to(device)

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()

# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.classifier.parameters())

# Load model from path
PATH = '/homedtic/ikoren/yalla/sel/models2/2.model'

# Load model - starting inference
net = model
net.load_state_dict(torch.load(PATH))
dataiter = iter(val_loader)
images, labels = dataiter.next()
images = images.cuda()
labels = labels.cuda()
outputs = net(images)


# Load training data - npy files
lossiloss = np.load("/homedtic/ikoren/yalla/sel/models2/npy/lossiloss.npy")
valiloss = np.load("/homedtic/ikoren/yalla/sel/models2/npy/valiloss.npy")
acc = np.load("/homedtic/ikoren/yalla/sel/models2/npy/acc.npy")
epoc = np.load("/homedtic/ikoren/yalla/sel/models2/npy/epoc.npy")

_, predicted = torch.max(outputs, 1)


def imshow(img):
    plt.figure(figsize=(20,10))
    img = img / 2 + 0.5  # unnormalize image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("/homedtic/ikoren/yalla/sel/img/pretrained_imgs.png")
    #plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# Print total validation accuracy of the model
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


# Classes accuracy
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()


# Loss plot
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.plot(epoc,lossiloss, color="red")
plt.plot(epoc,valiloss, color="blue")
plt.legend(['Loss', 'Val Loss'], loc='upper right')
plt.title("Train Loss", size=28)
plt.xlabel("Epochs", size=20)
plt.ylabel("Loss", size=20)
plt.savefig("/homedtic/ikoren/yalla/sel/models2/npy2/loss.png")
plt.show()

# Accuracy plot
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.plot(epoc,acc, color="blue")
plt.legend(['Train Accuracy'], loc='upper left')
plt.title("Accuracy", size=28)
plt.xlabel("Epochs", size=20)
plt.ylabel("Accuracy", size=20)
plt.savefig("/homedtic/ikoren/yalla/sel/models2/npy2/accuracy.png")
plt.show()

# Confusion Matrix
def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

actuals, predictions = test_label_predictions(model, device, val_loader)
print('Confusion matrix:\n==================')
print(confusion_matrix(actuals, predictions))
print()
print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
print("=========")
print()
print('Accuracy score: %f' % accuracy_score(actuals, predictions))
print("===============")


# Class Definition
dataset = datasets.ImageFolder('/homedtic/ikoren/yalla/sel/torch/train', transform=transformations)
classes1 = dataset.class_to_idx

# Get probabilities - test , predictions
def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    model.cpu()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.cpu()
            target = target.cpu()
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction) == which_class)
            probabilities.extend(np.exp(output[:, which_class]))
    return [i.item() for i in actuals], [i.item() for i in probabilities]


# Mapping Classes
under_w = classes1["Underweight"]
normal_w = classes1["Normal"]
over_w = classes1["Overweight"]
m_o = classes1["Medium Obesity"]
s_o = classes1["Super Obesity"]

name_cls = ["Underweight", "Normal", "Overweight", "Medium_Obesity", "Super_Obesity"]
vals_cls = [under_w, normal_w, over_w, m_o, s_o]


# Plot ROC Curve for each class - evaluate classifier: Sensitivity and Specificity
for i in range(len(name_cls)):
    title = name_cls[i]
    which_class = vals_cls[i]
    actuals, class_probabilities = test_class_probabilities(model, device, val_loader, which_class)
    DIR_X = '/homedtic/ikoren/yalla/sel/models2/npy2/'
    np.save(DIR_X+str(title)+"_y_pred.npy", class_probabilities)
    np.save(DIR_X+str(title)+"_y_true.npy", actuals)
    
    # plt config
    plt.style.use('ggplot')
    plt.figure(figsize=(20,10))
    
    fpr, tpr, _ = roc_curve(actuals, class_probabilities)
    roc_auc = auc(fpr, tpr)
    
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', size=20)
    plt.ylabel('True Positive Rate', size=20)
    plt.title('ROC for '+str(title)+' class', size=28)
    plt.legend(loc="lower right")
    plt.savefig(DIR_X+str(title)+"_ROC.png")
    plt.show()

