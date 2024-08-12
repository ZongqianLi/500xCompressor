import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_lora_parameters(model, lora_params_path):
    """
    Integrate LoRA parameters into the base LLM.

    Args:
        model (torch.nn.Module): The base LLM with randomly initialized LoRA parameters.
        lora_params_path (str): The path to the saved LoRA parameters.
    """
    # load the saved LoRA parameters
    lora_params = torch.load(lora_params_path)

    # integrate LoRA parameters into the LLM
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_params: 
                # update the compressed token or LoRA parameters or the token for regeneration
                if 'lora' in name or 'memory_embeddings' in name or "ae_embedding" in name:
                    param.copy_(lora_params[name])
                    if 'memory_embeddings' in name:
                        print("Find memory_embeddings!")
                    if "ae_embedding" in name:
                        print("Find ae_embedding!")
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:
                print(f"No saved parameter for {name}")


class L3ICAE(nn.Module):
    def __init__(self,
                llama_path,
                cache_dir,
                use_auth_token,
                max_length,
                lora_path,
                lora_config,
                num_mem,
                device
                ):
        """
        Load LLaMA and LoRA.

        Args:
            llama_path (str): Name of the original LLaMA model.
            cache_dir (str): Path to the cache of the LLaMA model.
            use_auth_token (str): Huggingface token for using the LLaMA model.
            max_length (int): Max length of the text to be compressed.
            lora_path (str): Path to the LoRA parameters.
            lora_config (dict): Configuration parameters for LoRA.
            num_mem (int): Number of compressed tokens.
            device (torch.device): Use cpu or gpu.
        """
        super(L3ICAE, self).__init__()
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_path, 
            cache_dir=cache_dir, 
            use_auth_token=use_auth_token,
            torch_dtype=torch.bfloat16,
        )
        print("LLaMA loaded.")
        self.llama = get_peft_model(self.llama, lora_config)
        print("LoRA added.")
        self.llama.eval()
        for param in self.llama.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            llama_path, 
            cache_dir=cache_dir, 
            use_auth_token=use_auth_token
        )
        print("LLaMA tokenizer loaded.")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_mem = num_mem
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 4096, dtype=torch.bfloat16).to(device))
        self.ae_embedding = nn.Parameter(torch.randn(1, 1, 4096, dtype=torch.bfloat16).to(device))
        load_lora_parameters(self, lora_path)
        print("LoRA updated.")
        self.device = device

    def compress(self, text, text_tokens=None, output_path=None):
        """
        Compress the text into a small number of compressed tokens.

        Args:
            text (List[str]): Input text.
            text tokens (torch.Tensor): (batch size, sequence length), input text tokens.
        
        Returns:
            mem_vec (torch.tensor): output compressed tokens.
        """
        # If the input is not text token
        # use the tokenizer to tokenize the text
        if text_tokens is None:
            text_tokens = self.tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_length, 
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
        else:
            text_tokens = text_tokens.to(self.device)
        text_tok_embeddings = self.llama.get_input_embeddings()(text_tokens)
        # initialize compressed tokens
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        # encoder input: original text + compressed tokens
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.llama(inputs_embeds=encoder_input_embeddings, output_hidden_states=True).hidden_states[-1]
        # get the output compressed tokens
        mem_vec = encoder_output[:, -self.num_mem:, :]

        return mem_vec

    def predict(self, mem_vec, max_new_tokens, prompt):
        """
        Regenerate the compressed text or do QA based on the compressed tokens.

        Args:
            mem_vec (torch.Tensor): compressed query vector.
            max_new_tokens (int): maximum number of new tokens to generate.
            prompt (str): prompt text.

        Returns:
            generated_text (str): Generated text. 
        """
        if prompt == "ae":
            prompt_embedding = self.ae_embedding
        else:
            prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            prompt_embedding = self.llama.get_input_embeddings()(prompt_tokens).to(self.device)

        input_embeddings = torch.cat((mem_vec, prompt_embedding), dim=1)

        generated_text = []
        for i in range(max_new_tokens):
            # predict the next new token by the original LLM
            with self.llama.disable_adapter():
                if i == 0:
                    output = self.llama(inputs_embeds=input_embeddings)
                else:
                    output = self.llama(inputs_embeds=input_embeddings, past_key_values=past_key_values)
            logits = output.logits
            past_key_values = output.past_key_values

            # choose the token id with the highest probability
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # stop generating new tokens when meet end tokens
            if next_token == torch.tensor([128001], device='cuda:0') or next_token == self.tokenizer.eos_token_id:
                break

            # add the new token to generated tokens
            generated_text.append(next_token.item())
        
            # update the input token
            input_tokens = next_token.unsqueeze(0)
            input_embeddings = self.llama.get_input_embeddings()(input_tokens)

        generated_text = self.tokenizer.decode(generated_text, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generated_text


