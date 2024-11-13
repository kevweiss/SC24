# SC24
This repository contains demonstration code for the Supercomputing 2024 conference in Atlanta. The demonstrations showcase PyTorch's Remote Procedure Call (RPC) framework, facilitating various GPU-intensive tasks by leveraging distributed computing across nodes.
## Accessing scinet24


For these demos, access the `scinet24` server via SSH:
``` ssh root@2620:118:5007:1002:5c24:a:0:730 ```

## Demos

### VectorOps Demo
This demo showcases vector operations across distributed nodes.

- On scinet24 (GPU node in Ashburn):
``` python3 ./SC24/DEMOS/VectorOperations/sc_demo_vectorOps.py --rank 0 --master_addr 162.250.137.149 --ifname enp5s0f1 ```

- On Voyager (Atlanta):
``` python3 ./SC24/DEMOS/VectorOperations/sc_demo_vectorOps.py --rank 1 --master_addr 162.250.137.149 --ifname eno2 ```

### MNIST Demo
This demonstration trains `EnhancedNet`, a fully connected neural network with batch normalization and dropout layers, on the MNIST dataset. MNIST contains 70,000 grayscale images of handwritten digits (0–9), each 28x28 pixels, making it a standard dataset for image classification tasks. The `EnhancedNet` model first flattens the input images to a 784-dimensional vector, then processes it through a series of fully connected layers with batch normalization and dropout. This architecture helps reduce overfitting and accelerates training by stabilizing input distributions, making it effective for classifying handwritten digits in MNIST.

- On scinet24 (GPU node in Ashburn):
``` python3 ./SC24/DEMOS/MNIST/sc_demo_mnist.py --rank 0 --master_addr 162.250.137.149 --ifname enp5s0f1 ```

- On Voyager:
``` python3 ./SC24/DEMOS/MNIST/sc_demo_mnist.py --rank 1 --master_addr 162.250.137.149 --ifname eno2 ```

### ResNet Demo

This demo trains ResNet18 and ResNet34. ResNet is a convolutional neural networks known for it's "skip connections," which help mitigate vanishing gradients, making it effective on deep networks. ResNet18 is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes, ideal for evaluating model performance on relatively simple image classification tasks. ResNet34 is trained on the CIFAR-100 dataset, which contains 60,000 32x32 color images across 100 classes, providing a more challenging test for generalization with a wider range of object categories. Both datasets feature varied backgrounds, lighting, and object orientations, making them an excellent test for ResNet's ability to generalize in complex visual environments.


#### ResNet18
- On scinet24 (GPU node in Ashburn):
``` python3 ./SC24/DEMOS/ResNet/sc_demo_resnet18.py --rank 0 --master_addr 162.250.137.149 --ifname enp5s0f1 ```


- On Voyager:
``` python3 ./SC24/DEMOS/ResNet/sc_demo_resnet18.py --rank 1 --master_addr 162.250.137.149 --ifname eno2 ```

#### ResNet34

- On Voyager:
``` python3 ./SC24/DEMOS/ResNet/sc_demo_resnet34.py --rank 1 --master_addr 162.250.137.149 --ifname eno2 ```

- On scinet24 (GPU node in Ashburn):
``` python3 ./SC24/DEMOS/ResNet/sc_demo_resnet34.py --rank 0 --master_addr 162.250.137.149 --ifname enp5s0f1 ```




