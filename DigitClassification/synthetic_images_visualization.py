import torch
import torchvision
import torch
import torch.nn as nn
from torchvision import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pathlib

# n_epochs = 3
# batch_size_train = 64
# batch_size_test = 1000
# learning_rate = 0.01
# momentum = 0.5
# log_interval = 10
#
# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)
#
# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('./mnist', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=batch_size_train, shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('./mnist', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=batch_size_test, shuffle=True)
#
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
#
# print(example_data.shape)
#
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(4,3))
# for i in range(30):
#   plt.subplot(3,10,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   #print(example_data[i][0])
#   #plt.title("Ground Truth: {}".format(example_targets[i]))
#   # plt.xticks([])
#   # plt.yticks([])
#   plt.axis('off')
#   plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                       wspace=None, hspace=None)
#   plt.tight_layout()
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()

#---------------------------------------------------------------------

transformtrain = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.RandomHorizontalFlip(),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transformtest = transforms.Compose([
    transforms.Resize((28,28)),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

train_data = datasets.ImageFolder('synthetic-digits/synthetic_digits/imgs_train', transform=transformtrain)
test_data = datasets.ImageFolder('synthetic-digits/synthetic_digits/imgs_valid', transform=transformtest)

test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)


import matplotlib.pyplot as plt

plt.figure(figsize=(4,3))
for i in range(30):
  plt.subplot(3,10,i+1)
  plt.tight_layout()
  img = example_data[i]
  img_new = img.transpose(0,2)
  #print(img_new.shape)
  #img.view(28,28,3)
  plt.imshow(img_new)
  #print(example_data[i][0])
  #plt.title("Ground Truth: {}".format(example_targets[i]))
  # plt.xticks([])
  # plt.yticks([])
  plt.axis('off')
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                      wspace=None, hspace=None)
  plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
