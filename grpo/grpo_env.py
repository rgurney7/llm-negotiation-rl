import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

from shared.rollout_config import get_grpo_buyer, get_eval_model
from shared.reward import price_reward

MAX_BUYER_CHARS = 500


class NegotiationEnv:
    def __init__(self):
        self.eval_model = get_eval_model()

    def buyer_reply(self, buyer, buyer_prompt, transcript, seller_msg):
        full_transcript = f"{transcript}\nSeller: {seller_msg}"
        buyer_user = f"Transcript:\n{full_transcript}\n\nYour next reply:"
        reply = buyer.chat(buyer_prompt, buyer_user)
        if len(reply) > MAX_BUYER_CHARS:
            reply = reply[:MAX_BUYER_CHARS].rsplit(" ", 1)[0]
        return reply, full_transcript + f"\nBuyer: {reply}"

    def judge(self, transcript, listing_price):
        agreed_price = self.eval_model.extract_price(transcript)
        return price_reward(agreed_price, 0.7 * listing_price, listing_price)

    def buyer_replies_concurrent(self, buyer, buyer_prompt, transcripts, seller_msgs):
        replies = [None] * len(seller_msgs)
        updated = [None] * len(seller_msgs)
        with ThreadPoolExecutor(max_workers=len(seller_msgs)) as pool:
            futures = {
                pool.submit(self.buyer_reply, buyer, buyer_prompt, transcripts[i], seller_msgs[i]): i
                for i in range(len(seller_msgs))
            }
            for future in as_completed(futures):
                i = futures[future]
                replies[i], updated[i] = future.result()
        return replies, updated

    def judge_concurrent(self, transcripts, listing_price):
        rewards = [None] * len(transcripts)
        with ThreadPoolExecutor(max_workers=len(transcripts)) as pool:
            futures = {
                pool.submit(self.judge, transcripts[i], listing_price): i
                for i in range(len(transcripts))
            }
            for future in as_completed(futures):
                i = futures[future]
                rewards[i] = future.result()
        return rewards
