from transformers import AutoTokenizer, AutoModelForCausalLM

# This command downloads (if not already cached) the tokenizer and model from Hugging Face.
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

prompt = "Hello, I am a dataset"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
