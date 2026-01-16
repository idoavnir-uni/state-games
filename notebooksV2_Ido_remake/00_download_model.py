# %%
# Cell 2 — download newest 0.1b ctx4096 checkpoint from BlinkDL/rwkv7-g1 (Xet-backed)
import re
from huggingface_hub import HfApi, hf_hub_download
import huggingface_hub, importlib.util

print("huggingface_hub:", huggingface_hub.__version__)
print("hf_xet installed:", importlib.util.find_spec("hf_xet") is not None)

repo_id = "BlinkDL/rwkv7-g1"
api = HfApi()
files = api.list_repo_files(repo_id)
print(files)

# %%
# pick newest file matching "*0.1b*ctx4096*.pth"
cands = [f for f in files if f.endswith(".pth") and "1.5b" in f and "ctx8192" in f]
assert cands, "No 1.5b ctx4096 .pth files found in repo."

def extract_date(fname: str):
    m = re.search(r"-(\d{8})-", fname)
    return int(m.group(1)) if m else -1

fname = max(cands, key=extract_date)
print("Selected checkpoint:", fname)

pth_path = hf_hub_download(repo_id=repo_id, filename=fname)
print("Downloaded to:", pth_path)

# %%
model_prefix = pth_path[:-4] if pth_path.endswith(".pth") else pth_path
model_prefix
# %%
files

# %%
