import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def quick_test():
    """
    Quick test of the fine-tuned model
    """
    BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_PATH = "llama-docstring-generator-final"
    
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model without quantization for inference
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Test question
    question = "My hair is falling out and my nails are brittle. Is there a product that can help"
    prompt = f"<s>[INST] {question} [/INST] "
    
    print(f"Testing with question: {question}")
    print("Generating response...")
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "[/INST]" in response:
        answer = response.split("[/INST]")[-1].strip()
    else:
        answer = response
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print("\nTest completed!")

if __name__ == "__main__":
    quick_test()
