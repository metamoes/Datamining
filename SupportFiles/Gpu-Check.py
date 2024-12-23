import torch
import sys
import platform
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_check.log'),
        logging.StreamHandler()
    ]
)


class GPUChecker:
    """Comprehensive GPU checker for PyTorch"""

    def __init__(self):
        self.gpu_info = {}
        self.cuda_available = torch.cuda.is_available()

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed information about available GPUs"""
        if not self.cuda_available:
            return {"error": "CUDA not available"}

        try:
            gpu_count = torch.cuda.device_count()
            gpus = []

            for i in range(gpu_count):
                cuda_device = torch.device(f'cuda:{i}')
                gpu_properties = torch.cuda.get_device_properties(i)

                # Test GPU memory
                try:
                    # Attempt to allocate and free memory
                    test_tensor = torch.zeros((1000, 1000), device=cuda_device)
                    del test_tensor
                    torch.cuda.empty_cache()
                    memory_test = "Success"
                except Exception as e:
                    memory_test = f"Failed: {str(e)}"

                # Get memory info
                total_memory = torch.cuda.get_device_properties(i).total_memory
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_cached = torch.cuda.memory_reserved(i)

                gpu_info = {
                    "name": gpu_properties.name,
                    "compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}",
                    "total_memory": f"{total_memory / 1024 ** 2:.2f} MB",
                    "memory_allocated": f"{memory_allocated / 1024 ** 2:.2f} MB",
                    "memory_cached": f"{memory_cached / 1024 ** 2:.2f} MB",
                    "memory_test": memory_test,
                    "max_threads_per_block": gpu_properties.max_threads_per_block,
                    "max_block_dimensions": gpu_properties.max_block_dim,
                    "max_grid_dimensions": gpu_properties.max_grid_dim,
                    "max_shared_memory_per_block": f"{gpu_properties.max_shared_memory_per_block / 1024:.2f} KB",
                    "clock_rate": f"{gpu_properties.clock_rate / 1000:.2f} GHz"
                }
                gpus.append(gpu_info)

            return {
                "gpu_count": gpu_count,
                "gpus": gpus
            }

        except Exception as e:
            logging.error(f"Error getting GPU info: {str(e)}")
            return {"error": str(e)}

    def check_pytorch_cuda(self) -> Dict[str, Any]:
        """Check PyTorch CUDA configuration and run basic tests"""
        results = {
            "cuda_available": self.cuda_available,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if self.cuda_available else None,
            "cudnn_version": torch.backends.cudnn.version() if self.cuda_available and torch.backends.cudnn.is_available() else None,
            "cudnn_enabled": torch.backends.cudnn.enabled if self.cuda_available else False
        }

        if not self.cuda_available:
            return results

        try:
            # Test basic CUDA operations
            x = torch.rand(100, 100, device='cuda')
            y = torch.rand(100, 100, device='cuda')
            z = torch.matmul(x, y)
            del x, y, z
            torch.cuda.empty_cache()
            results["basic_operations_test"] = "Success"

            # Test CUDA memory operations
            results["memory_operations"] = self.test_memory_operations()

        except Exception as e:
            results["basic_operations_test"] = f"Failed: {str(e)}"

        return results

    def test_memory_operations(self) -> Dict[str, str]:
        """Test various CUDA memory operations"""
        results = {}

        try:
            # Test large tensor allocation
            test_size = 1000
            results["large_tensor_allocation"] = "Testing..."
            x = torch.randn(test_size, test_size, device='cuda')
            del x
            results["large_tensor_allocation"] = "Success"

            # Test multiple tensor operations
            results["multiple_operations"] = "Testing..."
            x = torch.randn(test_size, test_size, device='cuda')
            y = torch.randn(test_size, test_size, device='cuda')
            z = torch.matmul(x, y)
            del x, y, z
            results["multiple_operations"] = "Success"

            # Test memory cleanup
            results["memory_cleanup"] = "Testing..."
            torch.cuda.empty_cache()
            results["memory_cleanup"] = "Success"

        except Exception as e:
            results["error"] = str(e)

        return results

    def verify_gpu_capability(self) -> bool:
        """Verify if GPU is capable of running the desired operations"""
        if not self.cuda_available:
            logging.error("CUDA is not available")
            return False

        try:
            # Check compute capability
            gpu_props = torch.cuda.get_device_properties(0)
            compute_capability = float(f"{gpu_props.major}.{gpu_props.minor}")

            # Test memory allocation
            try:
                x = torch.rand(1000, 1000, device='cuda')
                y = torch.rand(1000, 1000, device='cuda')
                z = torch.matmul(x, y)
                del x, y, z
                torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"Memory allocation test failed: {str(e)}")
                return False

            # Check if GPU is RTX 2080 Super
            if "2080 Super" not in gpu_props.name:
                logging.warning(f"Expected RTX 2080 Super, found: {gpu_props.name}")

            # Verify compute capability is sufficient
            if compute_capability < 7.0:  # RTX 2080 Super has compute capability 7.5
                logging.warning(f"Compute capability might be insufficient: {compute_capability}")

            return True

        except Exception as e:
            logging.error(f"GPU capability verification failed: {str(e)}")
            return False

    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all GPU checks and return comprehensive results"""
        results = {
            "system_info": {
                "python_version": platform.python_version(),
                "os": platform.platform(),
                "pytorch_version": torch.__version__
            },
            "cuda_info": self.check_pytorch_cuda(),
            "gpu_info": self.get_gpu_info()
        }

        # Additional system checks
        if self.cuda_available:
            results["current_device"] = torch.cuda.current_device()
            results["default_device"] = torch.cuda.get_device_name(0)

        return results


def main():
    checker = GPUChecker()
    results = checker.run_comprehensive_check()

    # Print results in a readable format
    logging.info("\n=== GPU Check Results ===")

    logging.info("\nSystem Information:")
    for key, value in results["system_info"].items():
        logging.info(f"{key}: {value}")

    logging.info("\nCUDA Information:")
    for key, value in results["cuda_info"].items():
        logging.info(f"{key}: {value}")

    logging.info("\nGPU Information:")
    if "error" not in results["gpu_info"]:
        logging.info(f"GPU Count: {results['gpu_info']['gpu_count']}")
        for i, gpu in enumerate(results["gpu_info"]["gpus"]):
            logging.info(f"\nGPU {i}:")
            for key, value in gpu.items():
                logging.info(f"{key}: {value}")
    else:
        logging.error(f"Error getting GPU info: {results['gpu_info']['error']}")

    # Verify GPU capability
    if checker.verify_gpu_capability():
        logging.info("\nGPU capability verification: PASSED")
    else:
        logging.error("\nGPU capability verification: FAILED")


if __name__ == "__main__":
    main()