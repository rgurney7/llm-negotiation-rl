import sys
import os
import gymnasium as gym
from gymnasium import spaces
from dotenv import load_dotenv, find_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

load_dotenv(find_dotenv())

from shared.rollout_config import get_eval_model
from shared.reward import price_reward

eval_model = get_eval_model()

MAX_BUYER_CHARS = 500


class NegotiationEnv(gym.Env):
    def __init__(self, buyer, craigslist_df, max_steps=8):
        super().__init__()
        self.observation_space = spaces.Text(max_length=10000)
        self.action_space = spaces.Text(max_length=2000)
        self.max_steps = max_steps
        self.buyer = buyer
        self.craigslist_df = craigslist_df
        self.current_step = 0
        self.transcript = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.transcript = []

        row = self.craigslist_df.sample(1).iloc[0]
        self.listing_price = float(row["listing_price"])
        self.seller_prompt = row["seller_prompt"]
        self.buyer_prompt = row["buyer_prompt"]
        self.item_title = row.get("item_title", "Unknown")

        opening = self.buyer.chat(
            self.buyer_prompt,
            "You're reaching out about this item. Start the negotiation with a brief message.",
        )
        if not opening:
            opening = "Hi, I'm interested in this item. What's your best price?"
        if len(opening) > MAX_BUYER_CHARS:
            opening = opening[:MAX_BUYER_CHARS].rsplit(" ", 1)[0]
        self.transcript.append(("Buyer", opening))

        return self._get_agent_obs(), {}

    def get_seller_prompt(self):
        return self.seller_prompt

    def step(self, agent_text):
        self.current_step += 1
        self.transcript.append(("Seller", agent_text))

        buyer_user = f"Transcript:\n{self._build_transcript()}\n\nYour next reply:"
        buyer_reply = self.buyer.chat(self.buyer_prompt, buyer_user)
        if not buyer_reply:
            buyer_reply = "I need to think about it."
        if len(buyer_reply) > MAX_BUYER_CHARS:
            buyer_reply = buyer_reply[:MAX_BUYER_CHARS].rsplit(" ", 1)[0]
        self.transcript.append(("Buyer", buyer_reply))

        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.current_step >= self.max_steps:
            truncated = True
            agreed_price = eval_model.extract_price(self._build_transcript())
            reward = price_reward(agreed_price, 0.7 * self.listing_price, self.listing_price)
            info = {
                "agreed_price": agreed_price,
                "scenario": self.item_title,
                "listing_price": self.listing_price,
            }

        return self._get_agent_obs(), reward, terminated, truncated, info

    def _build_transcript(self):
        return "\n".join(f"{spk}: {txt}" for spk, txt in self.transcript)

    def _get_agent_obs(self):
        lines = [f"[{spk}]: {txt}" for spk, txt in self.transcript]
        return "Negotiation so far:\n" + "\n".join(lines) + "\n\n[Your Turn]:"
