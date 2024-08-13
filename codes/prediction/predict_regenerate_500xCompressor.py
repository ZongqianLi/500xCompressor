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
        Collect texts.

        Args:
            filepath (str): Path for lines of texts.
            start_line (int): The line start to be loaded.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()[start_line:]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx].strip()


if __name__ == "__main__":
    device = torch.device("cuda")
    
    # whether to clear the previous predictions
    clear = True
    # path for the original texts, lines of texts
    text_file_path = "<to be filled>"
    # path for the output regenerated texts
    output_file_path = "<to be filled>"
    # path for the target original texts
    target_file_path = "<to be filled>"
    status_file_path = "<to be filled>"
    # "bos" for regeneration
    prompt = "bos"
    # "regeneration" or "qa"
    mode = "regeneration"
    # max number of text tokens in encoder input (truncation or padding)
    context_len = 500
    # max number of context tokens to be compressed
    max_length = 96 
    # max number of new tokens to be generated
    max_new_tokens = max_length

    # huggingface path for the LLM
    llama_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    # cache path to save or load the LLM
    cache_dir="<to be filled>"
    # huggingface token to use the LLaMA model
    use_auth_token="<to be filled>"
    # lora parameters for regeneration
    lora_path_regen = "<to be filled>"
    # lora parameters for QA
    lora_path_qa = ""
    # number of compressed tokens
    num_mem = 16

    # this file only supports batch size 1
    batch_size = 1
    # start line of the benchmark file
    # set clear=False and start_line for continued prediction
    start_line = 0
    dataset = TextDataset(text_file_path, start_line=start_line)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # number of texts to be processed
    num_texts = 500

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
    model = L3LoraL3(
        llama_path=llama_path, 
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        max_length=max_length,
        lora_path=lora_path,
        lora_config=lora_config,
        num_mem=num_mem,
        device=device
    )
    model = model.to(device)

    # whether to clear the output file before prediction
    if clear == True:
        with open(output_file_path, 'w') as file:
            pass

        with open(target_file_path, 'w') as file:
            pass

        with open(status_file_path, 'w') as file:
            pass

    # number of the data record
    i = 0
    # number of data records that stop automatically
    n = 0

    for batch_texts in data_loader:
        print(f"Processed {i} batches.")

        # to store context tokens to be compressed
        back_tokens = torch.full(
            (context_len,), 
            model.tokenizer.eos_token_id, 
            dtype=torch.long
        )   

        start_time = time.time()

        # context tokens to be compressed
        text_tokens = model.tokenizer(
            batch_texts, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        # store the context tokens to be compressed
        back_tokens[0:0+text_tokens.shape[0]] = text_tokens

        # target text (original context)
        target_text = model.tokenizer.decode(
            text_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        # record the target text (original context)
        with open(target_file_path, 'a', encoding='utf-8') as file:
            target_text = target_text.replace("\n", " ").strip()
            file.write(target_text + '\n')

        # compress the context to compressed tokens (K V values)
        past_key_values = model.compress(
            text=batch_texts, 
            text_tokens=back_tokens.unsqueeze(0), 
            output_path=None
        )

        # regenerate the context based on the compressed tokens (K V values)
        predicted_text, end = model.predict(
            past_key_values=past_key_values, 
            max_new_tokens=max_new_tokens, 
            prompt=prompt
        )
        
        # whether the regeneration stops automatically
        if end == True:
            n += 1

        # record the regenerated text
        with open(output_file_path, 'a', encoding='utf-8') as file:
            predicted_text = predicted_text.replace("\n", " ").strip()
            file.write(predicted_text + '\n')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{n}/{i+1} Processed in {elapsed_time:.2f} seconds.")

        i += 1

        # stop if reach the maximum number of processed data records
        if i == num_texts:
            break

    # record the number of data records that stop automatically
    with open(status_file_path, "a") as file:
        file.write(f"{n}")

