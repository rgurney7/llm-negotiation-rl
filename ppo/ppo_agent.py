import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.distributions.categorical import Categorical

class NegotiationAgent(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.llm = self.model.to(self.device)

        lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

        for param in self.llm.parameters():
            param.requires_grad = False

        self.llm = get_peft_model(self.llm, lora_config)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "right"

        self.critic = nn.Sequential(
            nn.LayerNorm(self.llm.config.hidden_size),
            nn.Linear(self.llm.config.hidden_size, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)
        ).to(self.device)

    def generate(self, obs_text, seller_prompt=""):
        messages = []
        if seller_prompt:
            messages.append({"role": "system", "content": seller_prompt})
        messages.append({"role": "user", "content": obs_text})
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        context_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.llm.generate(context_ids, max_new_tokens=100, do_sample=True)

        generated_ids = output_ids[:, context_ids.shape[1]:]
        return context_ids, generated_ids

    def evaluate(self, context_ids, generated_ids):
        input_len = context_ids.shape[1]
        num_gen = generated_ids.shape[1]
        full_ids = torch.cat([context_ids, generated_ids], dim=1)

        outputs = self.llm(full_ids, output_hidden_states=True)

        # Shift left by one: logit at position i predicts token at i+1.
        logits = outputs.logits[:, input_len - 1 : input_len - 1 + num_gen, :]
        hidden = outputs.hidden_states[-1][:, input_len - 1 : input_len - 1 + num_gen, :].float()

        dist = Categorical(logits=logits.float())
        log_probs = dist.log_prob(generated_ids)
        entropy = dist.entropy()
        value = self.critic(hidden[0, -1, :]).squeeze()

        return log_probs[0], entropy[0], value
