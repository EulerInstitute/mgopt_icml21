import torchvision
import torchvision.transforms as transforms
import torch

import sys; sys.path.append('..')  # useful if you're running locally
import mnist1d
from mnist1d.data import get_templates, get_dataset_args, get_dataset
from mnist1d.train import get_model_args, train_model
from mnist1d.models import ConvBase, GRUBase, MLPBase, LinearBase
from mnist1d.utils import set_seed, plot_signals, ObjectView, from_pickle


class MNIST1dDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x_data, labels):
        'Initialization'
        self.y_data = labels
        self.x_data = x_data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x_data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.x_data[index]

        # Load data and get label
        X = self.x_data[index]
        y = self.y_data[index]
        return X, y


def get_data(ndata, batch_size, debug_true=False):
    if debug_true:
        num_workers = 0
    else:
        num_workers = 2

    if ndata == "mnist":
        classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        fullset = torchvision.datasets.MNIST(root='./_data', train=True, download=True, transform=trans)
        testset = torchvision.datasets.MNIST(root='./_data', train=False, download=True, transform=trans)

        trainset  = fullset
        valset = testset

    elif ndata == "mnist1d":
        classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')

        args = get_dataset_args()
        data = get_dataset(args, path='./_data/mnist1d_data.pkl', download=False)  # This is the default setting

        trainset = MNIST1dDataset(torch.from_numpy(data['x']).float(), torch.from_numpy(data['y']).long())
        testset = MNIST1dDataset(torch.from_numpy(data['x_test']).float(), torch.from_numpy(data['y_test']).long())

        valset = testset

    else:
        print("we do not have: ", ndata)
        exit(0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader, classes