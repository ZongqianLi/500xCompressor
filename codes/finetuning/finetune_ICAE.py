import json
import wandb
import torch
import numpy as np
import torch.nn as nn
from rouge import Rouge
import torch.optim as optim
from peft import LoraConfig
from ICAEL3QA import ICAEL3QA
from safetensors.torch import load_model
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data.append(json_data)
    return data

class TextDataset(Dataset):
    def __init__(self, text_file, llama_path, max_context_length, max_qa_len, num_mem):
        self.text = read_jsonl_file(text_file)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token="<to be filled>")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_context_length = max_context_length
        self.max_qa_len = max_qa_len
        self.num_mem = num_mem
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        c_tokens = self.tokenizer(
            self.text[idx]["context"], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_context_length, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.squeeze()

        question = self.text[idx]["question"]
        q_tokens = self.tokenizer(
            f"Question: {question} Answer: ", 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.squeeze()

        a_tokens = self.tokenizer(
            self.text[idx]["answer"], 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.squeeze()
        if a_tokens.shape == torch.Size([]):
            a_tokens = self.tokenizer(
                self.text[idx]["answer"], 
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.reshape(1)
        
        input_ids = torch.full((self.max_context_length+self.max_qa_len,), 128001, dtype=torch.long)
        input_ids[:self.max_context_length] = c_tokens
        input_ids[self.max_context_length:self.max_context_length+len(q_tokens)+len(a_tokens)] = torch.cat((q_tokens, a_tokens), dim=0)  

        target_tokens = torch.full((self.num_mem+self.max_qa_len,), -100, dtype=torch.long)
        target_tokens[self.num_mem+len(q_tokens)-1:self.num_mem+len(q_tokens)-1+len(a_tokens)+1] = torch.cat((a_tokens, torch.tensor([128001])), dim=0)

        return {"input_ids": input_ids, "labels": target_tokens}


if __name__ == "__main__":
    device = torch.device(f"cuda")

    # ====================
    # Training parameters
    # ====================
    project_name = "<to be filled>"
    train_text_path = "<to be filled>"
    test_text_path = "<to be filled>"
    lora_path="<to be filled>"
    resume_from_checkpoint = None
    num_mem = 1
    max_length = 500
    max_qa_len = 46 
    llama_path="meta-llama/Meta-Llama-3-8B-Instruct"

    output_dir = "<to be filled>"
    deepspeed_config = "<to be filled>"
    logging_dir = "<to be filled>"
    num_train_epochs = 10
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 48
    save_strategy = "steps"
    save_steps = 100
    evaluation_strategy = "steps"
    eval_steps = 500
    eval_accumulation_steps = 4
    logging_steps = 1
    learning_rate = 5e-5
    save_total_limit = 3
    lr_scheduler_type = "constant_with_warmup"
    warmup_steps = 300

    train_dataset = TextDataset(train_text_path, llama_path, max_length, max_qa_len, num_mem)
    test_dataset = TextDataset(test_text_path, llama_path, max_length, max_qa_len, num_mem)
    print("Dataset created.")
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    wandb.init(project=project_name)

    # ====================
    # Compression model
    # ====================
    print("Loading llama + lora + llama ...")
    model = ICAEL3QA(
        llama_path=llama_path,
        max_context_length=max_length,
        lora_path=lora_path,
        lora_config=lora_config,
        num_mem=num_mem,
        device=device
    )
    print("Number of trainable parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model is on CUDA device:", torch.cuda.current_device())
    model.config = model.llama.config
    print("model.llama.config: ", model.llama.config)
    print("llama + lora + llama loaded successfully.")

    # ====================
    # Training
    # ====================
    torch.autograd.set_detect_anomaly(True)

    training_args = TrainingArguments(
        output_dir=output_dir,          
        overwrite_output_dir=False,      
        num_train_epochs=num_train_epochs,              
        per_device_train_batch_size=per_device_train_batch_size,   
        per_device_eval_batch_size=per_device_eval_batch_size, 
        save_strategy=save_strategy,
        save_steps=save_steps,      
        evaluation_strategy=evaluation_strategy,    
        eval_steps=eval_steps, 
        eval_accumulation_steps=eval_accumulation_steps,
        logging_dir=logging_dir,    
        logging_steps=logging_steps,
        deepspeed=deepspeed_config,
        learning_rate=learning_rate,
        save_total_limit=save_total_limit,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    if resume_from_checkpoint == None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    evaluation_results = trainer.evaluate()
    print("evaluation_results: ", evaluation_results)

