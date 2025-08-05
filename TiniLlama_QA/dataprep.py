from datasets import load_dataset, Dataset
import os

def format_chat_prompt(example):
    """
    format the caht to Llama template
    """
    open_list=example['answer_keywords']
    string = ""
    for elem in open_list:
        string+=elem
    return {
        "text": f"<s>[INST] {example['question']} [/INST] {string}</s>"
    }
    
def main():
    input_json = "dataset.json"
    output_dataset= "processed_dataset"
    
    raw_dataset = load_dataset('json', data_files=input_json, split='train')
    formatted_data = raw_dataset.map(format_chat_prompt)
    
    formatted_data = formatted_data.remove_columns(['question','answer_keywords'])
    print("saving")
    formatted_data.save_to_disk(output_dataset)
    print(f"First example in the processed dataset:\n{formatted_data[0]['text']}")
if __name__ == "__main__":
    main()