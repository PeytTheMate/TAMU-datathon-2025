# finalize_model.py
import torch
from training.model import TronTransformer
from gymnasium import spaces

model = TronTransformer(spaces.Box(0,1,shape=(10,18,20)), features_dim=128)
model.load_state_dict(torch.load("models/interactive_final.pt", map_location="cpu"))
model.eval()
torch.save(model.state_dict(), "models/interactive_final.pt")
print("Model exported for Docker")