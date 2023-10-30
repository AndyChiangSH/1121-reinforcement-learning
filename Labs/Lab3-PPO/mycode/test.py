import torch

print("cuda" if torch.cuda.is_available() else "cpu")