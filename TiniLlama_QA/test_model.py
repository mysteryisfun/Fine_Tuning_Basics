import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_fine_tuned_model(base_model_id, adapter_path):
    """
    Load the fine-tuned model with LoRA adapters
    """
    logger.info(f"Loading base model: {base_model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Load base model without quantization for inference
    logger.info("Loading base model for inference...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("Base model loaded successfully")
    
    # Load LoRA adapters
    logger.info(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    logger.info("LoRA adapters loaded successfully")
    
    # Optional: Merge adapters for faster inference (but uses more memory)
    # Uncomment the next two lines if you want to merge the adapters
    # logger.info("Merging LoRA adapters...")
    # model = model.merge_and_unload()
    
    # Set model to evaluation mode
    model.eval()
    
    logger.info("Model loaded and ready for inference!")
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=256, temperature=0.7):
    """
    Generate a response for a given question
    """
    # Format the prompt similar to training format
    prompt = f"<s>[INST] {question} [/INST] "
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after [/INST])
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    return response

def test_with_sample_questions():
    """
    Test the model with some sample questions
    """
    sample_questions = [
        "What is Duolife",
        "How does a vitamin helps up"
    ]
    
    return sample_questions

def interactive_testing(model, tokenizer):
    """
    Interactive testing mode
    """
    logger.info("Starting interactive testing mode. Type 'quit' to exit.")
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            logger.info("Exiting interactive mode...")
            break
        
        if not question:
            continue
        
        logger.info(f"Generating response for: {question}")
        response = generate_response(model, tokenizer, question)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {response}")
        print("-" * 50)

def batch_testing(model, tokenizer, questions):
    """
    Test multiple questions at once
    """
    logger.info(f"Testing {len(questions)} sample questions...")
    
    results = []
    for i, question in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}: {question}")
        response = generate_response(model, tokenizer, question)
        
        result = {
            "question": question,
            "answer": response
        }
        results.append(result)
        
        print(f"\nQ{i}: {question}")
        print(f"A{i}: {response}")
        print("-" * 50)
    
    return results

def save_test_results(results, filename="test_results.json"):
    """
    Save test results to a JSON file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Test results saved to {filename}")

def main():
    # Configuration
    BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_PATH = "llama-docstring-generator-final"  # Path to your trained adapters
    
    try:
        # Load the fine-tuned model
        model, tokenizer = load_fine_tuned_model(BASE_MODEL_ID, ADAPTER_PATH)
        
        # Get sample questions
        sample_questions = test_with_sample_questions()
        
        print("\nChoose testing mode:")
        print("1. Batch testing with sample questions")
        print("2. Interactive testing")
        print("3. Both")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            # Batch testing
            logger.info("Starting batch testing...")
            results = batch_testing(model, tokenizer, sample_questions)
            save_test_results(results)
        
        if choice in ['2', '3']:
            # Interactive testing
            interactive_testing(model, tokenizer)
        
        logger.info("Testing completed!")
        
    except FileNotFoundError:
        logger.error(f"Adapter path '{ADAPTER_PATH}' not found. Make sure you have completed training first.")
        logger.error("Expected structure: llama-docstring-generator-final/adapter_config.json")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    main()
