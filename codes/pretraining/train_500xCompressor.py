import json
import torch
import wandb
import numpy as np
import torch.nn as nn
from rouge import Rouge
import torch.optim as optim
from peft import LoraConfig
from L3LoraL3 import L3LoraL3
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer

def read_text_from_file(file_path):
    """
    Load lines of texts.

    Args:
        file_path (str): Path for lines of texts.

    Returns:
        (List[str]): List of texts.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

class TextDataset(Dataset):
    def __init__(self, text_file, llama_path, max_length):
        """
        Create the training or evaluation dataset.

        Args:
            text_file (str): Path for lines of texts.
            llama_path (str): Path for the base LLM.
            max_length (int): Max number of tokens to be compressed.
        """
        self.text = read_text_from_file(text_file)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token="<to be filled>")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text_tokens = self.tokenizer(
            self.text[idx], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids
        input_ids = text_tokens.squeeze()
        # input text tokens + EOS token
        target_tokens = torch.full((len(input_ids)+1,), -100, dtype=torch.long)
        text_eos_tokens = input_ids.tolist()
        text_eos_tokens.append(128001)
        text_eos_tokens_len = len(text_eos_tokens)
        target_tokens[0:0+text_eos_tokens_len] = torch.tensor(text_eos_tokens, dtype=torch.long, device=device)
        return {"input_ids": input_ids, "labels": target_tokens}


if __name__ == "__main__":
    device = torch.device(f"cuda")

    # ====================
    # training configurations
    # ====================
    project_name = "<to be filled>"
    # training corpus (lines of texts)
    train_text_path = "<to be filled>"
    test_text_path = "<to be filled>"
    # path to save the results
    output_dir = "<to be filled>"
    # number of compressed tokens
    num_mem = 1
    # max number of tokens to be compressed
    max_length = 500
    # resume training from specific checkpoint
    resume_from_checkpoint = None
    # (huggingface) path for the base LLM
    llama_path="meta-llama/Meta-Llama-3-8B-Instruct"

    # path for the deepspeed configuration
    deepspeed_config = "<to be filled>"
    logging_dir = "<to be filled>"
    num_train_epochs = 3
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 48
    save_strategy = "steps"
    save_steps = 300
    evaluation_strategy = "steps"
    eval_steps = 100
    eval_accumulation_steps = 4
    logging_steps = 1
    learning_rate = 1e-4
    save_total_limit = 1
    lr_scheduler_type = "constant_with_warmup"
    warmup_steps = 300

    # create the training and evaluation datasets
    train_dataset = TextDataset(train_text_path, llama_path, max_length)
    test_dataset = TextDataset(test_text_path, llama_path, max_length)
    print("Dataset created.")

    # lora configurations
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    wandb.init(project=project_name)

    # ====================
    # compression model
    # ====================
    print("Loading llama + lora + llama ...")
    model = L3LoraL3(
        llama_path=llama_path,
        max_length=max_length,
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
    # give the detailed information for the error
    torch.autograd.set_detect_anomaly(True)

    # training parameters
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

    # If the resume path is not none
    # continue from the provided checkpoint
    if resume_from_checkpoint == None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    evaluation_results = trainer.evaluate()
    print("evaluation_results: ", evaluation_results)


