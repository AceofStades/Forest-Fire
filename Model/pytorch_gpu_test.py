import torch
import sys


def check_gpu_status():
    """Checks and reports PyTorch and CUDA availability and status."""

    print("--- PyTorch and CUDA Configuration Check ---")

    # 1. PyTorch Version Check
    try:
        print(f"PyTorch Version: {torch.__version__}")
    except:
        print("PyTorch not installed or accessible.")
        return

    # 2. CUDA Availability Check
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_cuda_available}")

    if not is_cuda_available:
        print("\nACTION REQUIRED: CUDA is NOT detected by PyTorch.")
        print(
            "Please check your NVIDIA installation, driver paths, and PyTorch package version."
        )
        return

    # 3. GPU/Device Details
    gpu_count = torch.cuda.device_count()
    print(f"CUDA Devices Found: {gpu_count}")

    for i in range(gpu_count):
        print(f"  GPU {i} Name: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA Capability: {torch.cuda.get_device_capability(i)}")

    # 4. Library Path Sanity Check (Optional, but often helpful)
    print(f"\nTorch CUDA Home: {torch.cuda.get_device_name(0)}")

    # 5. Simple GPU Calculation Test
    try:
        device = torch.device("cuda")

        # Create tensors and move them to the GPU
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        # Perform matrix multiplication on GPU
        c = torch.matmul(a, b)

        print("\n--- GPU Calculation Test ---")
        print("Test Result: SUCCESS!")
        print(f"Tensor is on Device: {c.device}")
        print(
            f"Verification: First 5 elements of result: {c.flatten()[:5].cpu().numpy()}"
        )

    except Exception as e:
        print("\n--- GPU Calculation Test ---")
        print(f"Test Result: FAILURE. Calculation failed to execute on GPU.")
        print(f"Error Details: {e}")
        print(
            "\nACTION REQUIRED: Your current Python environment cannot fully initialize CUDA."
        )


check_gpu_status()
