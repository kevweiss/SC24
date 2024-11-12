# SC24
Repo contains code for SC'24 Demonstrations


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

This demo trains ResNet34, a 34-layer convolutional neural network known for its "skip connections," which help mitigate vanishing gradients, making it effective on deep networks. The SVHN (Street View House Numbers) dataset is used, featuring real-world images of digits (0–9) from street signs, ideal for digit recognition tasks. With 600,000 labeled digits, SVHN challenges models to classify digits with varied lighting, angles, and backgrounds, making it an excellent test for ResNet34’s ability to generalize in complex visual environments.

- On scinet24 (GPU node in Ashburn):
``` python3 ./SC24/DEMOS/ResNet/sc_demo_resnet34.py --rank 0 --master_addr 162.250.137.149 --ifname enp5s0f1 ```


- On Voyager:
``` python3 ./SC24/DEMOS/ResNet/sc_demo_resnet34.py --rank 1 --master_addr 162.250.137.149 --ifname eno2 ```





