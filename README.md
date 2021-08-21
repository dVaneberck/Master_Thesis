# Reinforcement learning based AI to play first-person video games

This is the repository for the thesis master Reinforcement learning based AI to play first-person video games from Ecole Polytechnique de Louvain at Uclouvain.

## Installation

The requirement file should contain all the library you need to run the code. However, with the default installation, the CPU version of Pytorch is installed and thus only the CPU is usable to run the program. If you want to use your GPU to accelerate the computation, you need to install CUDA (check if your GPU is compatible) and a pytorch version compatible with CUDA.

Here installation link for both:

[Pytorch](https://pytorch.org)

[CUDA](https://developer.nvidia.com/cuda-downloads)

## Running the code

You will find the program file in the directory Sources to run the code. It's the "reinforcement_learning.py" which have to be called to run the code and you need as argument an configuration file to run the environment you want. Here an example with Cartpole:

> python reinforcement_learning.py config_cartpole.yaml

If you want to run another envrionment, you can use one of the three configuration file we propose. If you want to create your own configuration, you can by editing the actual configuration files or by creating your own. Note that only Cartpole support both Convnet and MLP utilisation, others envrionment are only usable with MLP net.
