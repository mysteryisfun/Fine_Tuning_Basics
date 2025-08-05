import logging
import os
from datetime import datetime
from datasets import load_from_disk
from transformers import TrainingArguments

from trl import SFTTrainer

from model_loader import load_model_for_training

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting TinyLlama fine-tuning process")
    
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    PROCESSED_DATA_PATH = "processed_dataset"
    OUTPUT_DIR = "llama-docstring-generator"
    FINAL_MODEL_PATH = "llama-docstring-generator-final"

    logger.info(f"Model ID: {MODEL_ID}")
    logger.info(f"Dataset path: {PROCESSED_DATA_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Final model path: {FINAL_MODEL_PATH}")

    try:
        logger.info("Loading model, tokenizer, and LoRA configuration...")
        model, tokenizer, lora_config = load_model_for_training(MODEL_ID)
        logger.info("Model loading completed successfully")
        
        logger.info("Loading processed dataset...")
        train_dataset = load_from_disk(PROCESSED_DATA_PATH)
        logger.info(f"Dataset loaded with {len(train_dataset)} training examples")
        
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            num_train_epochs=30,
            max_steps=-1,
            save_steps=100,
            logging_dir=f"{OUTPUT_DIR}/logs",
            report_to=None,  # Disable wandb/tensorboard for now
        )
        
        logger.info("Training arguments configured:")
        logger.info(f"  - Batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        logger.info(f"  - Learning rate: {training_args.learning_rate}")
        logger.info(f"  - Number of epochs: {training_args.num_train_epochs}")
        logger.info(f"  - Logging steps: {training_args.logging_steps}")
        
        logger.info("Initializing SFT Trainer...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=lora_config,
            args=training_args
        )
        logger.info("Trainer initialized successfully")
        
        logger.info("Starting training process...")
        trainer.train()
        logger.info("Training completed successfully!")
        
        logger.info(f"Saving final adapter model to {FINAL_MODEL_PATH}...")
        trainer.model.save_pretrained(FINAL_MODEL_PATH)
        logger.info("Model saved successfully!")
        logger.info("Fine-tuning process completed!")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main()
    
