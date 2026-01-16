# %%
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model, get_model_config, print_model_structure

# %%
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")
config = get_model_config(model)
print_model_structure(model)

# %%
# Test prompt
ctx = "Tom Hanks's favorite color is green. Olaf Scholz's favorite color is blue. Michael Jordan's favorite color is blue. What is Olaf Scholz's favorite color? Answer: his favorite color is"
print(f"Prompt: {ctx}")
print("Output: ", end='')

def my_print(s):
    print(s, end='', flush=True)

model.generate(ctx, token_count=20, callback=my_print)
print('\n')

# %%
# Direct forward pass example
out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())

out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())
print('\n')

# %%
# State structure
tot_layer = 0
tot_params = 0
avg_shape = None
for s in state:
    if len(s.shape) != 1:
        tot_layer += 1
        if avg_shape is None:
            avg_shape = s.shape
        tot_params += s.shape[0] * s.shape[1] * s.shape[2] 
print(f"Total layers: {tot_layer}")
print(f"Total parameters: {tot_params//1024//1024} MB")
print(f"Average shape: {avg_shape}")
# %%
