import json
import time
import torch
import numpy as np
import torch.nn as nn
from ICAEL3 import ICAEL3
from peft import LoraConfig
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer

class TextDataset(Dataset):
    def __init__(self, filepath, start_line=0):
        with open(filepath, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()[start_line:]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx].strip()


if __name__ == "__main__":
    device = torch.device("cuda")
    
    clear = True
    text_file_path = '<to be filled>'
    output_file_path = '<to be filled>'
    target_file_path = "<to be filled>"
    status_file_path = "<to be filled>" 
    prompt = "ae"
    mode = "regeneration"
    context_len = 500
    max_length = 384
    max_new_tokens = max_length

    llama_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    cache_dir="<to be filled>"
    use_auth_token="<to be filled>"
    lora_path_regen = "<to be filled>"
    lora_path_qa = ""
    num_mem = 16

    batch_size = 1
    start_line = 0
    dataset = TextDataset(text_file_path, start_line=start_line)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_texts = 100

    if mode == "regeneration":
        lora_path = lora_path_regen
    elif mode == "qa":
        lora_path = lora_path_qa
    else:
        print("""Please specify the mode: "regeneration" or "qa".""")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = ICAEL3(
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

    if clear == True:
        with open(output_file_path, 'w') as file:
            pass

        with open(target_file_path, 'w') as file:
            pass

        with open(status_file_path, 'w') as file:
            pass

    i = 0
    n = 0

    for batch_texts in data_loader:
        print(f"Processed {i} batches.")

        back_tokens = torch.full(
            (context_len,), 
            model.tokenizer.eos_token_id, 
            dtype=torch.long
        )

        start_time = time.time()

        text_tokens = model.tokenizer(
            batch_texts, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        back_tokens[0:0+text_tokens.shape[0]] = text_tokens

        target_text = model.tokenizer.decode(
            text_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        with open(target_file_path, 'a', encoding='utf-8') as file:
            target_text = target_text.replace("\n", " ").strip()
            file.write(target_text + '\n')

        mem_vec = model.compress(
            text=batch_texts, 
            text_tokens=back_tokens.unsqueeze(0), 
            output_path=None
        )

        predicted_text, end = model.predict(
            mem_vec=mem_vec, 
            max_new_tokens=max_new_tokens, 
            prompt=prompt
        )

        if end == True:
            n += 1

        with open(output_file_path, 'a', encoding='utf-8') as file:
            predicted_text = predicted_text.replace("\n", " ").strip()
            file.write(predicted_text + '\n')

        i += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{n}/{i} auto ends. Processed in {elapsed_time:.2f} seconds.")

        if i == num_texts:
            break

    with open(status_file_path, "a") as file:
        file.write(f"{n}\n")


