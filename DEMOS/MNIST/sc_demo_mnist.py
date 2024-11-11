import os
import argparse
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import time
import warnings

# Suppress FurtureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Dummy function for synchronization
def sync_worker():
    pass

# Function to log epoch updates on rank 1
def log_epoch_update(epoch, accuracy):
    print(f"Worker1: Received epoch {epoch} update with accuracy: {accuracy:.2f}%")

# Define an updated neural network with dropout and batch normalization
class EnhancedNet(nn.Module):
    def __init__(self):
        super(EnhancedNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Model training function with updated learning rate and model architecture
def train_model_on_gpu(dataloader, iterations=1000):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = EnhancedNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        try:
            for epoch in range(iterations):
                model.train()
                for i, (data, target) in enumerate(dataloader):
                    data, target = data.view(data.size(0), -1).to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                # For demonstration purposes, we calculate a dummy accuracy and send the update to rank 1
                accuracy = 100 * (1 - loss.item())
                rpc.rpc_async("worker1", log_epoch_update, args=(epoch + 1, accuracy))
                print(f"Iteration {epoch+1}/{iterations}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}%")

            return f"Final Loss: {loss.item()}"
        except Exception as e:
            print(f"Training encountered an error: {e}")
            return "Training failed due to an error."
    else:
        return "GPU not available on this rank."

def run_worker(rank, world_size, master_addr, master_port, ifname):
    # Set network interface and environment variables
    os.environ['GLOO_SOCKET_IFNAME'] = ifname
    os.environ['TP_SOCKET_IFNAME'] = ifname
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Initialize RPC with TensorPipe backend without device_maps
    options = rpc.TensorPipeRpcBackendOptions(init_method=f"tcp://{master_addr}:{master_port}", rpc_timeout=300)

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )

    if rank == 0:
        # GPU server performs model training
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

        result = train_model_on_gpu(dataloader)
        rpc.rpc_sync(to="worker1", func=print, args=(f"Training result from rank 0 (GPU): {result}",))

    elif rank == 1:
        # CPU client does passive monitoring
        rpc.rpc_sync(to="worker0", func=sync_worker)  # Initial sync
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
