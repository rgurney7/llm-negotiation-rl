# Download and parse the Stanford CraigslistBargain dataset into a flat CSV.
# Run: python -m negotiation.data.craigslist

import os
import csv
import json
import urllib.request

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "craigslist_parsed.csv")

FIELDNAMES = [
    "split",            # train / validation / test
    "uuid",             # dialogue ID
    "category",         # electronics, cars, housing, etc.
    "item_title",       # listing title
    "description",      # Craigslist listing description
    "listing_price",    # original Craigslist listing price
    "buyer_target",     # buyer's private target price
    "seller_target",    # seller's private target price
    "num_turns",        # total number of message events
    "final_price",      # agreed price (None if no deal)
    "deal_reached",     # True / False
    "seller_utterances", # all seller messages joined by " ||| "
    "buyer_utterances",  # all buyer messages joined by " ||| "
    "transcript",       # full transcript: "Seller: ... \n Buyer: ..."
]

MIN_TURNS = 3

# CodaLab raw JSON URLs
DATA_URLS = {
    "train": "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json",
    "validation": "https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json",
    "test": "https://worksheets.codalab.org/rest/bundles/0x54d325bbcfb2463583995725ed8ca42b/contents/blob/parsed.json",
}


def download_split(url, name):
    print(f"  Downloading {name}...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(f"  Got {len(data)} dialogues", flush=True)
    return data


def parse_dialogue(row, split_name):
    kbs = row["kbs"]
    listing_price = None
    item_title = None
    description = ""
    category = None

    for kb in kbs:
        listing_price = kb["item"]["Price"]
        item_title = kb["item"]["Title"]
        category = kb["item"]["Category"]
        desc_parts = kb["item"].get("Description", [])
        if desc_parts:
            description = " ".join(desc_parts).strip()

    # kbs[i] is agent i's knowledge base; figure out which agent is the buyer.
    buyer_agent = 0 if kbs[0]["personal"]["Role"] == "buyer" else 1
    seller_agent = 1 - buyer_agent
    buyer_target = kbs[buyer_agent]["personal"].get("Target")
    seller_target = kbs[seller_agent]["personal"].get("Target")

    events = row["events"]
    seller_msgs = []
    buyer_msgs = []
    transcript_lines = []
    num_message_turns = 0

    for event in events:
        action = event["action"]
        agent = event["agent"]
        data = event["data"]

        if action == "message" and data:
            num_message_turns += 1
            label = "Seller" if agent == seller_agent else "Buyer"
            transcript_lines.append(f"{label}: {data}")
            if agent == seller_agent:
                seller_msgs.append(data)
            else:
                buyer_msgs.append(data)
        elif action == "offer" and isinstance(data, dict):
            label = "Seller" if agent == seller_agent else "Buyer"
            transcript_lines.append(f"{label}: [Offers ${data.get('price')}]")
        elif action in ("accept", "reject", "quit"):
            label = "Seller" if agent == seller_agent else "Buyer"
            transcript_lines.append(f"{label}: [{action.capitalize()}]")

    outcome = row.get("outcome", {}) or {}
    deal_reached = outcome.get("reward", 0) == 1
    final_price = None
    if deal_reached and outcome.get("offer"):
        final_price = outcome["offer"].get("price")

    return {
        "split": split_name,
        "uuid": row.get("uuid", ""),
        "category": category,
        "item_title": item_title,
        "description": description,
        "listing_price": listing_price,
        "buyer_target": buyer_target,
        "seller_target": seller_target,
        "num_turns": num_message_turns,
        "final_price": final_price,
        "deal_reached": deal_reached,
        "seller_utterances": " ||| ".join(seller_msgs),
        "buyer_utterances": " ||| ".join(buyer_msgs),
        "transcript": "\n".join(transcript_lines),
    }


def main():
    print("Downloading CraigslistBargain from CodaLab...", flush=True)

    total = 0
    kept = 0
    skipped_short = 0
    skipped_no_price = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for split_name, url in DATA_URLS.items():
            dialogues = download_split(url, split_name)
            print(f"Processing {split_name} ({len(dialogues)} dialogues)...", flush=True)

            for row in dialogues:
                total += 1

                scenario = row.get("scenario", {})
                parsed_row = {
                    "uuid": row.get("uuid", ""),
                    "kbs": scenario.get("kbs", []),
                    "events": row.get("events", []),
                    "outcome": row.get("outcome"),
                }

                try:
                    parsed = parse_dialogue(parsed_row, split_name)
                except Exception:
                    continue

                if parsed["num_turns"] < MIN_TURNS:
                    skipped_short += 1
                    continue

                has_price = (parsed["final_price"] is not None or
                             any(c.isdigit() for c in parsed["transcript"] if c != '\n'))
                if not has_price:
                    skipped_no_price += 1
                    continue

                writer.writerow(parsed)
                kept += 1

    print(f"\nDone. Total: {total}, Kept: {kept}, "
          f"Skipped (short): {skipped_short}, Skipped (no price): {skipped_no_price}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
