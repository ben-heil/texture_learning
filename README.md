# Texture Learning

Given a large, labeled set of images, it is straightforward to train a model to reach a high classification accuracy on an independent, identically distributed, test set. 
Exactly what these models learn is less clear.
In this repository we compare the performance of various models on texturised images find which architectures learn more about global features.

## Installation
Everything you need to run our code can be installed using [anaconda](https://www.anaconda.com/distribution/). 
Simply navigate to the root of the texture\_learning directory and run the commands below.
```sh
conda env create -f environment.yml
conda activate texture
```
