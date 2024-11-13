import os
import argparse
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Dummy function for synchronization
def sync_worker():
    pass

# Function to log epoch updates on rank 1
def log_epoch_update(epoch, accuracy):
    print(f"Worker1: Received epoch {epoch} update with accuracy: {accuracy:.2f}%")

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):  # CIFAR10 has 10 classes
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Adding Dropout to fully connected layers to reduce overfitting
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Model training function with updated learning rate and model architecture
def train_model_on_gpu(dataloader, iterations=1000):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = ResNet18().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        try:
            for epoch in range(iterations):
                model.train()
                for i, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                accuracy = 100 * (1 - loss.item())  # Dummy accuracy calculation
                rpc.rpc_async("worker1", log_epoch_update, args=(epoch + 1, accuracy))
                print(f"Iteration {epoch+1}/{iterations}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}%")

            return f"Final Loss: {loss.item()}"
        except Exception as e:
            print(f"Training encountered an error: {e}")
            return "Training failed due to an error."
    else:
        return "GPU not available on this rank."

def run_worker(rank, world_size, master_addr, master_port, ifname):
    os.environ['GLOO_SOCKET_IFNAME'] = ifname
    os.environ['TP_SOCKET_IFNAME'] = ifname
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    options = rpc.TensorPipeRpcBackendOptions(init_method=f"tcp://{master_addr}:{master_port}", rpc_timeout=300)

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )

    if rank == 0:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataloader = DataLoader(cifar10_dataset, batch_size=64, shuffle=True)

        result = train_model_on_gpu(dataloader)
        rpc.rpc_sync(to="worker1", func=print, args=(f"Training result from rank 0 (GPU): {result}",))

    elif rank == 1:
        rpc.rpc_sync(to="worker0", func=sync_worker)
        print("Worker1: Entering passive monitoring mode.")
        while True:
            time.sleep(2)

    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current node")
    parser.add_argument("--world_size", type=int, default=2, help="Total number of nodes")
    parser.add_argument("--master_addr", type=str, required=True, help="IP address of the master node")
    parser.add_argument("--master_port", type=str, default="29500", help="Port of the master node")
    parser.add_argument("--ifname", type=str, required=True, help="Network interface name")
    args = parser.parse_args()

    run_worker(args.rank, args.world_size, args.master_addr, args.master_port, args.ifname)
