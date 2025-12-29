import torch


def _run_cuda_op() -> None:
    # Small matmul to confirm CUDA execution.
    a = torch.randn((128, 128), device="cuda")
    b = torch.randn((128, 128), device="cuda")
    _ = a @ b


def print_cuda_diagnostics(prefix: str = "cuda") -> None:
    if not torch.cuda.is_available():
        print(f"{prefix}: unavailable (torch.cuda.is_available()=False)")
        return

    device_index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    _run_cuda_op()
    print(
        f"{prefix}: available device={device_index} name={name} capability={capability} op=ok"
    )
