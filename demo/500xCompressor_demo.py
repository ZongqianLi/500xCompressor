import json
import time
import torch
import numpy as np
import torch.nn as nn
from peft import LoraConfig
from L3LoraL3 import L3LoraL3
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer

if __name__ == "__main__":
    device = torch.device("cuda")

    # base LLM name in huggingface
    llama_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    # cache path to save the LLM
    cache_dir = None
    # huggingface token to use llama model
    use_auth_token="<to be filled>"
    # llama lora parameters for regeneration
    lora_path_regen = "<to be filled>"
    # llama lora parameters for question-answering
    lora_path_qa = "<to be filled>"
    # number of tokens used for compression
    num_mem = 4 
    # "regeneration" or "qa"
    mode = "qa"

    # max number of input tokens
    context_len = 500
    # max number of input tokens to be compressed
    max_length = 96
    # max number of new tokens to be generated
    max_new_tokens = 96

    context = """We show that every reciprocity sheaf gives rise to a cycle (pre)module in the sense of Rost over a perfect field. Over a perfect field of positive characteristic, we show that the first cohomology group of a logarithmic de Rham-Witt sheaf has a partial cycle module structure. As a consequence, we show that Kato complexes of logarithmic de Rham-Witt sheaves satisfy functoriality properties similar to Rost's cycle complexes."""
    question = "Over what type of field do we show that Kato complexes satisfy functoriality properties?"

    # identify the lora path according to the mode
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

    # load llama-3-lora + llama-3
    model = L3LoraL3(llama_path=llama_path, 
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                    max_length=max_length,
                    lora_path=lora_path,
                    lora_config=lora_config,
                    num_mem=num_mem,
                    device=device)
    model = model.to(device)
        
    back_tokens = torch.full((context_len,), model.tokenizer.eos_token_id, dtype=torch.long)

    text_tokens = model.tokenizer(context, 
                                truncation=True, 
                                max_length=max_length, 
                                return_tensors="pt",
                                add_special_tokens=False).input_ids[0]

    back_tokens[0:0+text_tokens.shape[0]] = text_tokens

    # compress the text
    past_key_values = model.compress(text=context, text_tokens=back_tokens.unsqueeze(0), output_path = None)
    for i, layer in enumerate(past_key_values):
        key_shape, value_shape = layer[0].shape, layer[1].shape
        break

    # identify the prompt according to the mode
    if mode == "regeneration":
        prompt = "bos"
    elif mode == "qa":
        prompt = f"Question: {question} Answer: "

    # regenerate the compressed text or do QA based on the compressed tokens
    predicted_text = model.predict(past_key_values=past_key_values, 
                                    max_new_tokens=max_new_tokens, 
                                    prompt=prompt)

    print("Predicted text: " + predicted_text)


