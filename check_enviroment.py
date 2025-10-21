"""
Environment Verification Script for PyTorch Deep Learning Setup
Checks package installation and GPU availability with error handling.
"""
import sys

def check_package(package_name, import_name=None):
    """
    Check if a package is installed and return its version.
    Handles import errors gracefully.
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)

        # Try different version attributes
        version_attrs = ['__version__', 'VERSION', 'version']
        version = "Unknown"

        for attr in version_attrs:
            if hasattr(module, attr):
                version = getattr(module, attr)
                # Handle cases where version might be a tuple
                if isinstance(version, (tuple,list)):
                    version = '.'.join(map(str, version))
                break
        return True, version
    
    except ImportError:
        return False, "Not Installed"
    except Exception as e:
        return False, f"Error: {str(e)}"
    
def check_pytorch_details():
    """
    Check PyTorch-specific details including GPU availability.
    """
    try:
        import torch

        details = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_names': []
        }

        if details['cuda_available']:
            details['cuda_version'] = torch.version.cuda
            details['gpu_count'] = torch.cuda.device_count()
            details['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(details['gpu_count'])]

        return True, details
    
    except ImportError:
        return False, "PyTorch is not installed."
    except Exception as e:
        return False, f"Error: {str(e)}"
    
def main():
    print("=" * 60)
    print("PYTORCH DEVELOPMENT ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # Python version
    print(f"Python Version: {sys.version}\n")

    # List of packages to check
    packages_to_check = [
        # Core PyTorch ecosystem
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("TorchAudio", "torchaudio"),
        
        # Data and numerical computing
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        
        # Visualization
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        
        # Image processing
        ("Pillow (PIL)", "PIL"),
        ("OpenCV", "cv2"),
        
        # Utilities
        ("tqdm", "tqdm"),
        ("Jupyter", "jupyter"),
    ]

    # Check all packages
    print("PACKAGE VERSIONS:")
    print("-" * 40)
    
    all_packages_ok = True
    for display_name, import_name in packages_to_check:
        is_installed, version = check_package(display_name, import_name)
        
        status = "✅" if is_installed else "❌"
        print(f"{status} {display_name:20} {version}")
        
        if not is_installed:
            all_packages_ok = False
    
    print("\n" + "=" * 60)
    print("PYTORCH GPU SUPPORT:")
    print("-" * 40)
    
    # Check PyTorch details
    pytorch_ok, pytorch_details = check_pytorch_details()
    
    if pytorch_ok:
        print(f"✅ PyTorch Version: {pytorch_details['version']}")
        print(f"✅ CUDA Available: {pytorch_details['cuda_available']}")
        
        if pytorch_details['cuda_available']:
            print(f"✅ CUDA Version: {pytorch_details['cuda_version']}")
            print(f"✅ GPU Count: {pytorch_details['gpu_count']}")
            
            for i, gpu_name in enumerate(pytorch_details['gpu_names']):
                print(f"   GPU {i}: {gpu_name}")
            
            # Test GPU tensor operations
            try:
                import torch
                # Test basic GPU operations
                if torch.cuda.is_available():
                    x = torch.randn(3, 3).cuda()
                    y = torch.randn(3, 3).cuda()
                    z = torch.matmul(x, y)
                    print("✅ GPU tensor operations: Working correctly")
                    
                    # Check GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
                
            except Exception as e:
                print(f"❌ GPU operations test failed: {e}")
        else:
            print("❌ No GPU detected - Training will use CPU only")
            print("💡 Tip: Consider installing CUDA-enabled PyTorch for faster training")
    
    else:
        print(f"❌ {pytorch_details}")
        all_packages_ok = False
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("-" * 40)
    
    if all_packages_ok and pytorch_ok:
        print("🎉 SUCCESS: All packages installed correctly!")
        if pytorch_details['cuda_available']:
            print("🚀 GPU acceleration is available for training!")
        else:
            print("⚠️  GPU not available - training will be slower on CPU")
    else:
        print("❌ Some packages failed to install.")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try: pip install --upgrade pip")
        print("3. For PyTorch GPU issues, visit: https://pytorch.org/get-started/locally/")
        print("4. Make sure you have compatible NVIDIA drivers for CUDA")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
