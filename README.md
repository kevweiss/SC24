# SC24
Repo contains code for SC'24 Demonstrations


## Accessing scinet24
For these demos, access the `scinet24` server via SSH:
``` ssh root@2620:118:5007:1002:5c24:a:0:730 ```

## Demos

### VectorOps Demo
This demo showcases vector operations across distributed nodes.

- On scinet24 (GPU node in Ashburn):
``` python3 ./SC24/Demos/VectorOperations/sc_demo_vectorOps.py --rank 0 --master_addr 162.250.137.149 --ifname enp5s0f1 ```

- On Voyager (Atlanta):
``` python3 ./SC24/Demos/VectorOperations/sc_demo_vectorOps.py --rank 1 --master_addr 162.250.137.149 --ifname eno2 ```

### MNIST Demo
This demo uses the MNIST dataset to demonstrate distributed neural network training.

- On scinet24 (GPU node in Ashburn):
``` python3 ./SC24/Demos/MNIST/sc_demo_mnist.py --rank 0 --master_addr 162.250.137.149 --ifname enp5s0f1 ```

- On Voyager:
``` python3 ./SC24/Demos/MNIST/sc_demo_mnist.py --rank 1 --master_addr 162.250.137.149 --ifname eno2 ```
