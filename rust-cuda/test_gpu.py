import rust_cuda 
import time

def main():
    print("=== SciRS2 GPU Demo ===\n")
    
    # Test 1: GPU Availability
    print("1. Testing GPU Availability:")
    result = rust_cuda.test_gpu_availability()
    print(f"   {result}\n")
    
    # Test 2: Basic GPU Matrix Multiplication
    print("2. Basic GPU Matrix Multiplication:")
    sizes = [100, 500, 1000]
    for size in sizes:
        result = rust_cuda.gpu_matrix_multiply(size)
        print(f"   Size {size}x{size}: {result}")
    print()
    
    # Test 3: CPU vs GPU Performance
    print("3. CPU vs GPU Performance Comparison:")
    result = rust_cuda.compare_cpu_gpu_performance(512, 10)
    print(f"   {result}\n")
    
    # Test 4: Batch Operations
    print("4. GPU Batch Operations:")
    result = rust_cuda.gpu_batch_operations(10, 256)
    print(f"   {result}\n")
    
    # Test 5: Neural Network Demo
    print("5. Neural Network GPU Demo:")
    result = rust_cuda.gpu_neural_network_demo()
    print(f"   {result}\n")

if __name__ == "__main__":
    main()
