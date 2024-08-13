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
    
    clear = True
    text_file_path = "<to be filled>"
    root = "<to be filled>"
    output_file_path = root + "gen_answer"
    target_file_path = root + "trg_answer" 
    question_file_path = root + "question"
    context_file_path = root + "context"
    compress_time_path = root + "compress_time"
    predict_time_path = root + "predict_time"
    generated_token_len_path = root + "generated_token_len"
    mode = "qa"
    context_len = 500
    max_length = 480
    max_new_tokens = 46

    llama_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    cache_dir="<to be filled>"
    use_auth_token="<to be filled>"
    lora_path_regen = ""
    lora_path_qa = "<to be filled>"
    num_mem = 16

    batch_size = 1
    start_line = 0
    dataset = TextDataset(text_file_path, start_line=start_line)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_texts = 1000

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

        with open(question_file_path, 'w') as file:
            pass

        with open(context_file_path, 'w') as file:
            pass

        with open(compress_time_path, 'w') as file:
            pass

        with open(predict_time_path, 'w') as file:
            pass

        with open(generated_token_len_path, 'w') as file:
            pass

    i = 0

    for batch_texts in data_loader:
        with open(question_file_path, 'a', encoding='utf-8') as file:
            file.write(batch_texts["question"][0] + '\n')

        with open(context_file_path, 'a', encoding='utf-8') as file:
            file.write(batch_texts["context"][0] + '\n')

        back_tokens = torch.full(
            (context_len,), 
            model.tokenizer.eos_token_id, 
            dtype=torch.long
        )
        
        print(f"Processed {i} batches.")

        i += 1

        start_time = time.time()

        text_tokens = model.tokenizer(
            batch_texts["context"][0], 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        back_tokens[0:0+text_tokens.shape[0]] = text_tokens

        target_text = batch_texts["answer"][0]

        with open(target_file_path, 'a', encoding='utf-8') as file:
            target_text = target_text.replace("\n", " ").strip()
            file.write(target_text + '\n')

        compress_start = time.time()
        mem_vec = model.compress(
            text=batch_texts, 
            text_tokens=back_tokens.unsqueeze(0), 
            output_path = None
        )
        compress_end = time.time()
        compress_time = compress_end - compress_start
        print(f"Compressed in {compress_time:.2f} seconds.")
        with open(compress_time_path, 'a', encoding='utf-8') as file:
            file.write(str(compress_time) + '\n')

        question = batch_texts["question"][0]
        predict_start = time.time()
        predicted_text, end, generated_token_len = model.predict(
            mem_vec=mem_vec, 
            max_new_tokens=max_new_tokens, 
            prompt=f"Question: {question} Answer: "
        )
        predict_end = time.time()
        predict_time = predict_end - predict_start
        print(f"Predicted in {predict_time:.2f} seconds.")
        with open(predict_time_path, 'a', encoding='utf-8') as file:
            file.write(str(predict_time) + '\n')
        with open(generated_token_len_path, 'a', encoding='utf-8') as file:
            file.write(str(generated_token_len) + '\n')
        with open(output_file_path, 'a', encoding='utf-8') as file:
            predicted_text = predicted_text.replace("\n", " ").strip()
            file.write(predicted_text + '\n')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processed in {elapsed_time:.2f} seconds.")

        if i == num_texts:
            break


