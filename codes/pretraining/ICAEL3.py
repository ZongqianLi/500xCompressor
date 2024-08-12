import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

class ICAEL3(nn.Module):
    def __init__(
        self,
        llama_path,
        max_length,
        lora_config,
        num_mem,
        device
    ):
        super(ICAEL3, self).__init__()
        llama = AutoModelForCausalLM.from_pretrained(
            llama_path, 
            cache_dir="<to be filled>", 
            use_auth_token="<to be filled>",
            torch_dtype=torch.bfloat16,
        )
        self.llama = get_peft_model(llama, lora_config)
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
            if 'lora' in name:
                param.requires_grad = True
        print(f"Total parameters of llama: {sum(p.numel() for p in self.llama.parameters())}")
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token="<to be filled>")
        print("llama tokenizer loaded.")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_mem = num_mem
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 4096, dtype=torch.bfloat16).to(device))
        self.memory_embeddings.requires_grad = True
        self.ae_embedding = nn.Parameter(torch.randn(1, 1, 4096, dtype=torch.bfloat16).to(device))
        self.ae_embedding.requires_grad = True
        self.device = device

    def forward(self, input_ids, labels):
        ####################
        # Encoder - llama+lora
        ####################
        text_tokens = input_ids
        target_tokens = labels
        text_tok_embeddings = self.llama.get_input_embeddings()(text_tokens).to(self.device)
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.llama(inputs_embeds=encoder_input_embeddings, output_hidden_states=True).hidden_states[-1]
        encoder_output = encoder_output[:, -self.num_mem:, :]

        ####################
        # Decoder - llama
        ####################
        ae_tok_embedding = self.ae_embedding.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        prompt_tok_embeddings = ae_tok_embedding

        decoder_input_embeddings = torch.cat((encoder_output, prompt_tok_embeddings, text_tok_embeddings), dim=1)
        with self.llama.disable_adapter():
            decoder_output = self.llama(inputs_embeds=decoder_input_embeddings)
        all_logits = decoder_output.logits

        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))

        return {'loss': loss, 'logits': all_logits}


