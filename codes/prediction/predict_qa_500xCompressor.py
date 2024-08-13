import json
import time
import torch
import numpy as np
import torch.nn as nn
from peft import LoraConfig
from L3LoraL3 import L3LoraL3
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer

class TextDataset(Dataset):
    def __init__(self, filepath, start_line=0):
        """
        Load the QA dataset.

        Args:
            filepath (str): Path for extractive QA pairs (jsonl, lines of json).
            start_line (int): The line to start to be loaded.
        """
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()[start_line:]
            for line in lines:
                json_data = json.loads(line.strip())
                self.data.append(json_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    device = torch.device("cuda")
    
    # whether to clear the previous predictions
    clear = True
    # path for extractive QA pairs
    text_file_path = "<to be filled>"
    # root path to save results
    root = "<to be filled>"
    # path for generated answers
    output_file_path = root + "gen_results"
    # path for target answers
    target_file_path = root + "tar_results" 
    # path for the questions in the extractive QA pairs
    question_file_path = root + "question"
    # path for the contexts in the extractive QA pairs
    context_file_path = root + "context"
    # path for the time for compressing the context
    compress_time_path = root + "compress_time"
    # path for the time for generating the answer
    predict_time_path = root + "predict_time"
    # path for the length of the generated answer
    generated_token_length_path = root + "generated_token_length"
    # "regeneration" or "qa"
    mode = "qa" 
    # max number of text tokens in encoder input (truncation or padding)
    context_len = 500
    # max number of context tokens to be compressed
    max_length = 480
    # max number of new tokens to be generated
    max_new_tokens = 46

    # this file only supports batch size 1
    batch_size = 1
    # start line of the benchmark file
    # set clear=False and start_line for continued prediction
    start_line = 0
    dataset = TextDataset(text_file_path, start_line=start_line)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # number of texts to be processed
    num_texts = 1000

    # huggingface path for the LLM
    llama_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    # cache path to save or load the LLM
    cache_dir="<to be filled>"
    # huggingface token to use the LLaMA model
    use_auth_token="<to be filled>"
    # lora parameters for regeneration
    lora_path_regen = ""
    # lora parameters for QA
    lora_path_qa = "<to be filled>"
    # number of compressed tokens
    num_mem = 16

    # compress the text for regeneration or QA
    if mode == "regeneration":
        lora_path = lora_path_regen
    elif mode == "qa":
        lora_path = lora_path_qa
    else:
        print("""Please specify the mode: "regeneration" or "qa".""")

    # LoRA configurations
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # load the compression model: llama-3-lora + llama-3
    model = L3LoraL3(llama_path=llama_path, 
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                    max_length=max_length,
                    lora_path=lora_path,
                    lora_config=lora_config,
                    num_mem=num_mem,
                    device=device)
    model = model.to(device)

    # whether to clear the output file before prediction
    if clear == True:
        with open(output_file_path, 'w') as file:
            pass

        with open(target_file_path, 'w') as file:
            pass

        with open(question_file_path, 'w') as file:
            pass

        with open(context_file_path, 'w') as file:
            pass

        with open(compress_time_path, 'w') as file:
            pass

        with open(predict_time_path, 'w') as file:
            pass

        with open(generated_token_length_path, 'w') as file:
            pass

    # number of the data record
    i = 0

    for batch_texts in data_loader:
        print(f"Processed {i} batches.")

        # record the question in the extractive QA pair
        with open(question_file_path, 'a', encoding='utf-8') as file:
            file.write(batch_texts["question"][0] + '\n')

        # record the context in the extractive QA pair
        with open(context_file_path, 'a', encoding='utf-8') as file:
            file.write(batch_texts["context"][0] + '\n')
        
        # to store context tokens to be compressed
        back_tokens = torch.full((context_len,), model.tokenizer.eos_token_id, dtype=torch.long)

        start_time = time.time()

        # context tokens to be compressed
        text_tokens = model.tokenizer(batch_texts["context"][0], 
                                    truncation=True, 
                                    max_length=max_length, 
                                    return_tensors="pt",
                                    add_special_tokens=False).input_ids[0]

        # store the context tokens to be compressed
        back_tokens[0:0+text_tokens.shape[0]] = text_tokens

        # target answer
        target_text = batch_texts["answer"][0]
        with open(target_file_path, 'a', encoding='utf-8') as file:
            target_text = target_text.replace("\n", " ").strip()
            file.write(target_text + '\n')

        # compress the context
        compress_start = time.time()
        past_key_values = model.compress(text=batch_texts["context"][0], text_tokens=back_tokens.unsqueeze(0), output_path = None)
        compress_end = time.time()
        # record the time for compression
        compress_time = compress_end - compress_start
        print(f"Compressed in {compress_time:.2f} seconds.")
        with open(compress_time_path, 'a', encoding='utf-8') as file:
            file.write(str(compress_time) + '\n')

        # generate the answer based on the compressed context and the question
        question = batch_texts["question"][0]
        predict_start = time.time()
        predicted_text, end, generated_token_length = model.predict(past_key_values=past_key_values, 
                                        max_new_tokens=max_new_tokens, 
                                        prompt=f"Question: {question} Answer: ")
        predict_end = time.time()
        # record the time for prediction
        predict_time = predict_end - predict_start
        print(f"Predicted in {predict_time:.2f} seconds.")
        with open(predict_time_path, 'a', encoding='utf-8') as file:
            file.write(str(predict_time) + '\n')
        with open(generated_token_length_path, 'a', encoding='utf-8') as file:
            file.write(str(generated_token_length) + '\n')

        # record the generated answer
        with open(output_file_path, 'a', encoding='utf-8') as file:
            predicted_text = predicted_text.replace("\n", " ").strip()
            file.write(predicted_text + '\n')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processed in {elapsed_time:.2f} seconds.")

        i += 1

        # stop if reach the maximum number of processed data records
        if i == num_texts:
            break


