# TinyLlama Fine-Tuning for Question-Answering

A complete implementation of fine-tuning TinyLlama-1.1B-Chat using LoRA (Low-Rank Adaptation) for domain-specific question-answering tasks.

## Project Overview

This project demonstrates how to fine-tune a large language model (TinyLlama-1.1B-Chat-v1.0) on a custom dataset to create a specialized question-answering system. The implementation uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters to train the model efficiently while maintaining the base model's capabilities.

## Key Features

- **Efficient Training**: Uses LoRA adapters for parameter-efficient fine-tuning
- **4-bit Quantization**: Implements BitsAndBytesConfig for memory-efficient training
- **Custom Data Processing**: Formats data according to Llama's chat template
- **Comprehensive Logging**: Detailed logging throughout training and inference
- **Multiple Testing Modes**: Both quick testing and interactive evaluation

## Project Structure

```
TiniLlama_QA/
├── dataset.json                    # Raw question-answer dataset
├── dataprep.py                     # Data preprocessing script
├── processed_dataset/              # Processed training data
├── model_loader.py                 # Model loading and configuration
├── train.py                        # Main training script
├── test_model.py                   # Comprehensive testing script
├── quick_test.py                   # Quick model testing
├── llama-docstring-generator/      # Training checkpoints
├── llama-docstring-generator-final/ # Final LoRA adapters
├── logs/                           # Training logs
└── README.md                       # This file
```

## Implementation Details

### Data Preparation
- **Input Format**: JSON with `question` and `answer_keywords` fields
- **Output Format**: Llama chat template `<s>[INST] {question} [/INST] {answer}</s>`
- **Processing**: Converts keyword lists to concatenated strings for training

### Model Architecture
- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit with NF4 quantization type
- **Target Modules**: q_proj, k_proj, v_proj, o_proj

### Training Configuration
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **Learning Rate**: 2e-4
- **Batch Size**: 1 (with gradient accumulation of 4)
- **Epochs**: 5
- **Sequence Length**: 512 tokens

## Usage

### 1. Data Preparation
```bash
python dataprep.py
```
This script:
- Loads the raw dataset from `dataset.json`
- Formats questions and answers according to Llama's chat template
- Saves processed data to `processed_dataset/`

### 2. Model Training
```bash
python train.py
```
Training process:
- Loads TinyLlama base model with quantization
- Applies LoRA configuration
- Trains on processed dataset
- Saves adapters to `llama-docstring-generator-final/`

### 3. Model Testing

#### Quick Test
```bash
python quick_test.py
```
Tests the model with a single predefined question.

#### Comprehensive Testing
```bash
python test_model.py
```
Offers three testing modes:
1. **Batch Testing**: Tests with 5 sample questions, saves results to JSON
2. **Interactive Mode**: Ask custom questions in real-time
3. **Combined**: Both batch and interactive testing

## Technical Specifications

### Hardware Requirements
- **GPU**: CUDA-compatible GPU with at least 8GB VRAM
- **RAM**: 16GB+ system memory recommended
- **Storage**: 10GB+ for model files and checkpoints

### Software Dependencies
```
torch
transformers
datasets
peft
bitsandbytes
accelerate
trl
```

### Model Performance
- **Training Time**: ~30 minutes (depends on dataset size and hardware)
- **Memory Usage**: ~6-8GB GPU memory during training
- **Inference Speed**: ~2-5 seconds per response

## Key Implementation Decisions

1. **LoRA over Full Fine-tuning**: Chosen for memory efficiency and faster training
2. **4-bit Quantization**: Reduces memory requirements while maintaining performance
3. **Chat Template Format**: Maintains consistency with base model's training
4. **Gradient Accumulation**: Simulates larger batch sizes with limited memory

## Results and Evaluation

The fine-tuned model demonstrates:
- Domain-specific knowledge retention
- Improved response relevance for target topics
- Maintained general conversational abilities
- Efficient inference with LoRA adapters

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Import Errors**: Ensure all dependencies are installed in the correct environment
3. **Model Loading Failures**: Verify adapter files exist in the final model directory

### Performance Optimization
- Use `torch.compile()` for faster inference (PyTorch 2.0+)
- Enable gradient checkpointing for larger models
- Adjust LoRA rank based on available memory

## Future Improvements

- [ ] Implement evaluation metrics (BLEU, ROUGE)
- [ ] Add support for multi-turn conversations
- [ ] Integrate with Hugging Face Hub for model sharing
- [ ] Add gradual unfreezing for improved performance
- [ ] Implement knowledge distillation techniques

## License

This project is provided for educational and research purposes. Please ensure compliance with the respective model licenses when using for commercial applications.

## Acknowledgments

- TinyLlama team for the base model
- Hugging Face for the transformers library
- Microsoft for the LoRA implementation in PEFT
