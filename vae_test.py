# Variational Auto Encoder
# Nathan Briese

# TODO make a separate file for executing and training? Separate file for VAE object
# TODO print loss information while training?
# TODO add a silent option -q or conversely a verbose option -v
# TODO change README file to md and add much more detail (and math)

# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import getopt

def main(argv):
    model_load_path = "./vae.pth"

    # TODO update this information
    try:
        opts, args = getopt.getopt(argv,"tl:e:",["lr=", "e="])
    except getopt.GetoptError:
        print("usage: python3 VAE.py [-t | -l learning rate | -e num epochs]")
        print("Options and arguments:")
        print("-t train a new model")
        print("-l specify the learning rate (default=0.01)")
        print("-e specify the number of epochs (default=10)")
        print("-s specify relative file path to save the trained model (default=\"./\")")
        print("-m specify relative file path to load an existing model for testing (default=\"./\")")
        sys.exit()
        
    for opt, arg in opts:
        if opt == '-m':
            model_load_path = arg

    net = load_model(model_load_path)
    outputs = generate_output(net, 16)
    output_grid(outputs, 16, "./VAE_output.png")

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

def load_model(model_load_path):
    # import model from the pth file
    test_net = VAE_NET()
    test_net.load_state_dict(torch.load(model_load_path))
    return test_net

def generate_output(test_net, num_outputs):
    # generate new images similar to MNIST dataset from unit Gaussian random noise
    inputs = torch.randn(num_outputs, 50)
    outputs = test_net.decode(inputs)
    return outputs

def output_grid(images, num_outputs, grid_save_path):
    # Show the generated images in a grid
    plt.figure()
    plt.suptitle("Generating Images from Noise")
    for i in range(num_outputs):
        plt.subplot(4,4,i+1)
        plt.imshow(torch.reshape(images[i], (28,28)).detach().cpu().numpy(), cmap='gray')
    plt.savefig(grid_save_path)
    plt.close(grid_save_path)

if __name__ == "__main__":
    main(sys.argv[1:])

