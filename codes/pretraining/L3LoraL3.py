import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model

class L3LoraL3(nn.Module):
    def __init__(
        self,
        llama_path,
        max_length,
        lora_config,
        num_mem,
        device
    ):
        """
        Create the compression model: LLM-LoRA + LLM.

        Args:
            llama_path (str): Path for the base LLM.
            max_length (int): Max number of tokens to be compressed.
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(L3LoraL3, self).__init__()
        # load the original base LLaMA model
        llama = AutoModelForCausalLM.from_pretrained(
            llama_path, 
            # cache path to save the LLaMA model
            cache_dir="<to be filled>", 
            # huggingface token to use the LLaMA model
            use_auth_token="<to be filled>",
            torch_dtype=torch.bfloat16,
        )
        # add LoRA parameters to the LLM
        self.llama = get_peft_model(llama, lora_config)
        # only LoRA parameters are trainable
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
            if 'lora' in name:
                param.requires_grad = True
        print(f"Total parameters of llama: {sum(p.numel() for p in self.llama.parameters())}")
        # load the LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token="<to be filled>")
        print("llama tokenizer loaded.")
        # set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # max number of tokens to be compressed
        self.max_length = max_length
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # number of compressed tokens
        self.num_mem = num_mem
        # compressed token
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 4096, dtype=torch.bfloat16).to(device))
        self.memory_embeddings.requires_grad = True
        self.device = device

    def forward(self, input_ids, labels):
        ####################
        # Encoder - llama+lora
        ####################
        # input text tokens to be compressed
        text_tokens = input_ids
        # target tokens: input text tokens + EOS token
        target_tokens = labels
        text_tok_embeddings = self.llama.get_input_embeddings()(text_tokens).to(self.device)
        # compressed tokens
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        # encoder input: text tokens + compressed tokens
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.llama(inputs_embeds=encoder_input_embeddings)
        # get the K V values for the encoder output
        past_key_values = encoder_output.past_key_values
        # get the K V values for the compressed tokens
        trimmed_past_key_values = tuple(
            (layer_key[:, :, -self.num_mem:, :], layer_value[:, :, -self.num_mem:, :]) 
            for layer_key, layer_value in past_key_values
        )

        ####################
        # Decoder - llama
        ####################
        # BOS token
        prompt_tokens = [self.tokenizer.bos_token_id]
        prompt_tokens = torch.tensor(prompt_tokens, device=self.device)
        prompt_tok_embeddings = self.llama.get_input_embeddings()(prompt_tokens)
        prompt_tok_embeddings = prompt_tok_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1)

        # decoder input: BOS token + text tokens
        decoder_input_embeddings = torch.cat((prompt_tok_embeddings, text_tok_embeddings), dim=1)
        # use the original LLM without LoRA parameters
        with self.llama.disable_adapter():
            decoder_output = self.llama(inputs_embeds=decoder_input_embeddings, past_key_values=trimmed_past_key_values)
        # logits for the decoder output
        all_logits = decoder_output.logits

        # target output: text tokens + EOS token
        # calculate the cross entropy
        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))

        return {'loss': loss, 'logits': all_logits}


