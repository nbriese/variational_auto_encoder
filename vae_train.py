# Variational Auto Encoder : Training
# Nathan Briese

# TODO print loss information while training?
# TODO add a silent option -q or conversely a verbose option -v

# import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torchvision.datasets
import torchvision.transforms
import sys
import getopt

def main(argv):
    learn_rate = 0.01
    num_epochs = 10
    # TODO add timestamp to file name (without spaces)
    model_save_path = "./vae.pth"
    model_load_path = "./vae.pth"

    # TODO update this information
    try:
        opts, args = getopt.getopt(argv,"l:e:",["lr=", "e="])
    except getopt.GetoptError:
        print("usage: python3 vae_train.py [-l learning rate | -e num epochs]")
        print("Options and arguments:")
        print("-l specify the learning rate (default=0.01)")
        print("-e specify the number of epochs (default=10)")
        print("-s specify relative file path to save the trained model (default=\"./\")")
        print("-m specify relative file path to load an existing model for testing (default=\"./\")")
        sys.exit()
        
    for opt, arg in opts:
        if opt == '-l':
            learn_rate = float(arg)
        elif opt == '-e':
            num_epochs = int(arg)
        elif opt == '-s':
            model_save_path = arg

    net = train_model(num_epochs, learn_rate)
    print("Finished training")
    torch.save(net.state_dict(), model_save_path)
    print(f"Saved model to {model_save_path}")

class VAE_NET(nn.Module):
    def __init__(self):
        super(VAE_NET, self).__init__()
        self.fc1  = nn.Linear(784, 400)
        self.fc2  = nn.Linear(400, 50)
        self.fc3  = nn.Linear(50, 400)
        self.fc4  = nn.Linear(400, 784)
        self.sig  = nn.Sigmoid()

    def encode(self, x):
        # Input: training image, size 28x28
        x = x.view(-1, 784)
        x = nn.functional.relu(self.fc1(x))
        u = self.fc2(x)
        s = self.fc2(x)
        # e is unit normal noise
        e = torch.randn(u.size())
        z = u + (s * e)
        # Output: compressed version of image, size 20
        return z

    def decode(self, z):
        # Input: compressed image, size 20
        z = nn.functional.relu(self.fc3(z))
        z = self.sig(self.fc4(z))
        # Output: denoised original image, size 784
        return z

    def forward(self, x):
        return self.decode(self.encode(x))

def train_model(num_epochs, learn_rate):
    if(torch.cuda.is_available()):
        return train_cuda(num_epochs, learn_rate)
    else:
        return train_non_cuda(num_epochs, learn_rate)

# TODO modify this function to work without cudo
def train_non_cuda(num_epochs, learn_rate):
    torch.set_default_tensor_type(torch.FloatTensor)
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, 
        transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    net = VAE_NET().cuda()
    criterion = nn.BCELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

    print("Beginning training with %d epochs" % num_epochs)
    print("Training without CUDA support")
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}")
        running_loss = 0.0
        for _, (inputs, _) in enumerate(trainloader, 0):
            inputs = inputs.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward and backward and optimize
            outputs = net(inputs)
            loss = criterion(outputs, inputs.view(-1, 784))
            loss.backward()
            optimizer.step()

            # aggregate loss
            # running_loss += loss.item()*inputs.size()[0]

        # loss_list[epoch] = running_loss/len(trainloader)

    return net

def train_cuda(num_epochs, learn_rate):
    torch.set_default_tensor_type(cuda.FloatTensor)
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, 
        transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator(device='cuda'))

    net = VAE_NET().cuda()
    criterion = nn.BCELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

    print("Beginning training with %d epochs" % num_epochs)
    print("Training with CUDA support")
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}")
        running_loss = 0.0
        for _, (inputs, _) in enumerate(trainloader, 0):
            inputs = inputs.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward and backward and optimize
            outputs = net(inputs)
            loss = criterion(outputs, inputs.view(-1, 784))
            loss.backward()
            optimizer.step()

            # aggregate loss
            # running_loss += loss.item()*inputs.size()[0]

        # loss_list[epoch] = running_loss/len(trainloader)

    return net

if __name__ == "__main__":
    main(sys.argv[1:])

