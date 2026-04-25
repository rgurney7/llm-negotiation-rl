import random
import torch


class LocalBuyer:
    def __init__(self, agent):
        self.agent = agent
        self.name = "Qwen3.5-4B-base"

    def chat(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        tokenizer = self.agent.tokenizer
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(self.agent.device)

        self.agent.llm.disable_adapter_layers()
        with torch.no_grad():
            output_ids = self.agent.llm.generate(
                input_ids, max_new_tokens=150, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        self.agent.llm.enable_adapter_layers()
        return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def pick_buyer(local_buyer, get_api_buyer, local_prob=0.05):
    if random.random() < local_prob:
        return local_buyer
    return get_api_buyer()
