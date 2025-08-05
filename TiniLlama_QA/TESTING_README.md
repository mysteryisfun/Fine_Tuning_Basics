# Testing Your Fine-Tuned TinyLlama Model

After training your model, you can test it using the provided testing scripts.

## Prerequisites

Make sure your training has completed and you have the adapter files in the `llama-docstring-generator-final` directory.

## Testing Options

### 1. Quick Test (`quick_test.py`)

For a simple, single-question test:

```bash
python quick_test.py
```

This will:
- Load your fine-tuned model
- Test it with a predefined question
- Show the generated answer

### 2. Comprehensive Testing (`test_model.py`)

For more extensive testing:

```bash
python test_model.py
```

This offers three modes:

#### Batch Testing (Option 1)
- Tests with 5 predefined sample questions
- Saves results to `test_results.json`
- Good for evaluating overall performance

#### Interactive Testing (Option 2)
- Allows you to ask custom questions
- Type your questions and get immediate responses
- Type 'quit' to exit

#### Both (Option 3)
- Runs batch testing first
- Then enters interactive mode

## Sample Questions

The batch testing includes these sample questions:
- "What is machine learning?"
- "How does a neural network work?"
- "What is the difference between supervised and unsupervised learning?"
- "Explain what is deep learning?"
- "What is natural language processing?"

## Expected Output Format

The model will respond in the format it was trained on:

```
Question: What is machine learning?
Answer: [Generated response based on your training data]
```

## Troubleshooting

### If you get "FileNotFoundError":
- Make sure training completed successfully
- Check that `llama-docstring-generator-final` directory exists
- Verify it contains `adapter_config.json` and `adapter_model.bin`

### If you get memory errors:
- The model uses `torch.float16` and `device_map="auto"` to optimize memory usage
- Close other GPU-intensive applications
- Consider using a smaller batch size or sequence length

### If responses are poor quality:
- Your model might need more training epochs
- Check if your training data was properly formatted
- Consider adjusting training parameters

## Customizing the Test

You can modify the test scripts to:
- Change the temperature (0.1 for more deterministic, 0.9 for more creative)
- Adjust max_length for longer/shorter responses
- Add your own test questions
- Change the prompt format if needed

## Next Steps

After testing:
1. If results are good, your model is ready to use
2. If results need improvement, consider:
   - More training epochs
   - Different hyperparameters
   - More or better quality training data
   - Adjusting the LoRA configuration
