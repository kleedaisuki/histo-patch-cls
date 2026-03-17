import torch

print("torch version:", torch.__version__)
print("built with cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    print("matmul ok, shape =", z.shape)
