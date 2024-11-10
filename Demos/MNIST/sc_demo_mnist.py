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
