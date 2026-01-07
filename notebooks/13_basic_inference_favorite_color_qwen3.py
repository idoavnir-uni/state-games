import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

prompt = """Q: Tom Hanks's favorite color is green. Olaf Scholz's favorite color is red. Michael Jordan's favorite color is blue. What is Olaf Scholz's favorite color?
A: The answer is red.
Q: Emma Watson's favorite color is purple. Bill Gates's favorite color is green. Taylor Swift's favorite color is blue. What is Bill Gates's favorite color?
A: The answer is green.
Q: Elon Musk's favorite color is red. Beyoncé's favorite color is gold. Leonardo DiCaprio's favorite color is blue. What is Beyoncé's favorite color?"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
max_new_tokens = 20

print("Top 5 predictions for each generated token:\n")

with torch.no_grad():
    for step in range(max_new_tokens):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        next_token_id = top_indices[0, 0].unsqueeze(0).unsqueeze(0)
        next_token = tokenizer.decode(next_token_id[0])
        
        print(f"Step {step + 1} (selected: '{next_token}'):")
        for i in range(5):
            token = tokenizer.decode(top_indices[0, i])
            prob = top_probs[0, i].item()
            print(f"  {i+1}. '{token}' - {prob:.4f}")
        print()
        
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=attention_mask.device)], dim=1)
        
        if next_token_id[0, 0] == tokenizer.eos_token_id:
            break

print("Full generated text:")
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))

