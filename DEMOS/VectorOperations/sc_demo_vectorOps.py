import os
import argparse
import torch
import torch.distributed.rpc as rpc
import logging
import time
import warnings

# Suppress FurtureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

print(torch.__version__)

# Disable IBV by specifying transports
os.environ['TP_TRANSPORTS'] = 'uv'

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Dummy function for synchronization
def sync_worker():
    pass

# Heavy computation function for the GPU on rank 0
def compute_heavy_on_gpu(iterations=500):
    if torch.cuda.is_available():
        # Generate smaller initial tensors to avoid overflow
        tensor1 = torch.randn(4000, 4000, device='cuda') * 0.01
        tensor2 = torch.randn(4000, 4000, device='cuda') * 0.01
        result = None
        for _ in range(iterations):
            # Perform multiple operations to keep GPU engaged for a longer time
            result = torch.matmul(tensor1, tensor2)  # Matrix multiplication
            result = torch.exp(result)  # Exponential operation

            # Clamp result to avoid very large values
            result = torch.clamp(result, max=1e4)

            result = torch.log(result + 1e-5)  # Logarithmic operation

        return result.mean().cpu()  # Return a summary (mean) to the client (rank 1)
    else:
        return "GPU not available on this rank."

def run_worker(rank, world_size, master_addr, master_port, ifname):
    # Set network interface and environment variables
    os.environ['GLOO_SOCKET_IFNAME'] = ifname
    os.environ['TP_SOCKET_IFNAME'] = ifname
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    options = rpc.TensorPipeRpcBackendOptions(
        rpc_timeout=300,
    )

# Initialize RPC with TensorPipe backend
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
    )

    if rank == 0:
        # Rank 0 (GPU server) performs continuous computation and sends results to rank 1
        while True:
            result = compute_heavy_on_gpu()  # Keep GPU busy with heavy computation
            rpc.rpc_sync(to="worker1", func=print, args=(f"Result from rank 0 (GPU): {result}",))
            time.sleep(1)

    elif rank == 1:
        # Rank 1 (CPU client) receives results for monitoring
        rpc.rpc_sync(to="worker0", func=sync_worker)  # Initial sync
        while True:
            time.sleep(2)  # Passive loop; can be adjusted for additional tasks

    # Shutdown RPC
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
