
import os
import sys

# Set environment variables BEFORE importing RWKV
os.environ["RWKV_V7_ON"] = '1'
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # Disable custom CUDA kernel for now to reduce variables

try:
    import rwkv
    print(f"RWKV package version: {getattr(rwkv, '__version__', 'unknown')}")
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
except ImportError as e:
    print(f"Error importing RWKV: {e}")
    sys.exit(1)

model_path = '/a/home/cc/students/cs/yuvalmilo/.cache/huggingface/hub/models--BlinkDL--rwkv7-g1/snapshots/f5fbc5646d0a8d5f8877ff3ff66145e1d07b2b75/rwkv7b-g1b-0.1b-20250822-ctx4096'

print(f"Loading model from: {model_path}")

try:
    # Try CPU FP32 first to rule out FP16/CUDA issues
    model = RWKV(model=model_path, strategy='cpu fp32')
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

ctx = "User: simulate SpaceX mars landing using python\n\nAssistant: "
print(f"Input context: {ctx}")

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.8, top_k = 100, # slightly higher temp to see if it's just stuck
                     alpha_frequency = 0.0,
                     alpha_presence = 0.0,
                     alpha_decay = 0.997,
                     token_ban = [],
                     token_stop = [],
                     chunk_len = 256)

print("Generating...")
def my_print(s):
    print(s, end='', flush=True)

pipeline.generate(ctx, token_count=20, args=args, callback=my_print)
print('\nGeneration complete.')

