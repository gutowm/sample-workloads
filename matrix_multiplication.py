# matrix_multiplication.py
import torch
import time

# Define matrix size (N x N)
MATRIX_SIZE = 10000

print("Hello from MDIA cluster!")

print(f"Creating {MATRIX_SIZE}x{MATRIX_SIZE} matrices...")

# Create random matrices on CPU
A_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.float32)
B_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.float32)

print("Performing computation on CPU...")

start_cpu = time.time()
C_cpu = torch.matmul(A_cpu, B_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

print(f"CPU computation done in {cpu_time:.4f} seconds")

# --- GPU computation ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Performing same computation on GPU...")

    # Move tensors to GPU
    A_gpu = A_cpu.to(device)
    B_gpu = B_cpu.to(device)

    # Warm up GPU (optional but helps with fair timing)
    _ = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()  # Wait for GPU to finish (important for timing)
    start_gpu = time.time()
    C_gpu = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    print(f"GPU computation done in {gpu_time:.4f} seconds")

