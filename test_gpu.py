import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Current Device Index: {torch.cuda.current_device()}")
    print("\n✅ GPU ENABLED - CUDA is ready!")
else:
    print("\n⚠️  GPU NOT DETECTED - CPU will be used")
