import torch
from transformers import HubertModel

model = HubertModel.from_pretrained("team-lucid/hubert-base-korean")

wav = torch.ones(1, 16000)
outputs = model(wav)
print(f"Input:   {wav.shape}")  # [1, 16000]
print(f"Output:  {outputs.last_hidden_state.shape}")  # [1, 49, 768]