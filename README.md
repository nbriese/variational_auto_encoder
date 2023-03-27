# Variational Auto-Encoder
Nathan Briese 2020

## Introduction
The goal is to implement a variational auto-encoder (VAE).  This program is inspired by the paper, Auto-Encoding Variational Bayes
Citation: Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114v10.

The goal of a VAE is to take a set of training data and generate outputs that replecate the input data. After the model is trained, it takes in noise and produces a output resenbling the training data.

## How to run the program:
* Install necessary dependencies: `pip3 install numpy matplotlib torchvision torch`
* To run the full program flow at once: `python3 vae.py`
* To train a model run: `python3 vae_train.py`
  * Use -l to specify the learning rate
  * Use -e to specify the number of epochs
* To test an existing model, run: `python3 vae_test.py`

### Expected output
**TODO**

## How it works
**TODO**

## How do I plan to expand this program in the future?
* Increase output clarity and sharpness
  * Use convolutional neural networks instead of fully connected ones?
  * Tune the hyperparameters using bayesian optimization?
* Implement CUDA support so training is more efficient
* Make the use of this program more interactive
  * Maybe add a UI?
  * Use a different library to display the outputs (better presentation)
  
