import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.distributions.categorical import Categorical

class GrpoAgent(nn.Module):
    def __init__(self):
        super().__init__()

        model_id = "Qwen/Qwen3.5-4B"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
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

    def generate(self, obs_text: str):
        messages = [{"role": "user", "content": obs_text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt")
        context_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            output_ids = self.llm.generate(context_ids, attention_mask=attention_mask, max_new_tokens=100, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)

        generated_ids = output_ids[:, context_ids.shape[1]:]
        return context_ids, generated_ids

    def evaluate(self, context_ids, generated_ids):
        input_len = context_ids.shape[1]
        num_gen = generated_ids.shape[1]
        full_ids = torch.cat([context_ids, generated_ids], dim=1)

        outputs = self.llm(full_ids)
        # Shift left by one: logit at position i predicts token at i+1.
        logits = outputs.logits[:, input_len - 1 : input_len - 1 + num_gen, :]

        dist = Categorical(logits=logits.float())
        log_probs = dist.log_prob(generated_ids)
        entropy = dist.entropy()

        return log_probs[0], entropy[0]

    def evaluate_ref(self, context_ids, generated_ids):
        # Log probs under the frozen base model (LoRA disabled).
        self.llm.disable_adapter_layers()
        with torch.no_grad():
            input_len = context_ids.shape[1]
            num_gen = generated_ids.shape[1]
            full_ids = torch.cat([context_ids, generated_ids], dim=1)
            outputs = self.llm(full_ids)
            logits = outputs.logits[:, input_len - 1 : input_len - 1 + num_gen, :]
            dist = Categorical(logits=logits.float())
            ref_log_probs = dist.log_prob(generated_ids)
        self.llm.enable_adapter_layers()
        return ref_log_probs[0]

    def save(self, path):
        self.llm.save_pretrained(path)
