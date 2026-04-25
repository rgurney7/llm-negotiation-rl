SELLER_TACTICS = [
    {"name": "extreme_anchor",       "instruction": "Start with a price at least 2x your reservation price to anchor high."},
    {"name": "comparable_sales",     "instruction": "Justify your price by citing comparable sales or market rates."},
    {"name": "deadline_pressure",    "instruction": "Create urgency by mentioning another interested buyer or a deadline."},
    {"name": "bundle_unbundle",      "instruction": "Reframe the value by breaking down or bundling what's included."},
    {"name": "concede_slowly",       "instruction": "Make small concessions with clear rationale for each step down."},
    {"name": "silence",              "instruction": "After stating your price, stay quiet and let the buyer fill the gap."},
    {"name": "flinch",               "instruction": "React with surprise or disbelief at the buyer's first offer."},
]

BUYER_TACTICS = [
    {"name": "lowball_anchor",       "instruction": "Open with an offer around 30% of the listing price."},
    {"name": "competing_offer",      "instruction": "Claim you have a similar offer from another seller at a lower price."},
    {"name": "walkaway_threat",      "instruction": "Threaten to walk away and buy from a competitor instead."},
    {"name": "nitpick_flaws",        "instruction": "Point out every flaw, defect, or risk to justify a lower price."},
    {"name": "emotional_appeal",     "instruction": "Use an emotional story or reason to appeal for a lower price."},
    {"name": "budget_constraint",    "instruction": "Claim you have a hard budget limit and can't go higher."},
    {"name": "request_extras",       "instruction": "Ask for extras (free delivery, warranty, accessories) to close the deal."},
]


PERSONAS = [
    {
        "name": "normal",
        "description": "A reasonable buyer who negotiates fairly, makes logical counteroffers, and responds to good arguments.",
        "rigidity": "medium",
        "tactics": ["walkaway_threat", "budget_constraint"],
    },
    {
        "name": "aggressive",
        "description": "Pushy, impatient, uses pressure tactics. Lowballs hard, points out every flaw, makes the seller feel lucky to get any offer.",
        "rigidity": "high",
        "tactics": ["lowball_anchor", "nitpick_flaws"],
    },
    {
        "name": "passive",
        "description": "Wants a good deal but avoids conflict. Tends to accept the seller's framing, rarely pushes back, often caves when pressured.",
        "rigidity": "low",
        "tactics": ["emotional_appeal", "budget_constraint"],
    },
    {
        "name": "irrational",
        "description": "Inconsistent decision-making. Anchors on irrelevant details, changes priorities mid-negotiation, makes offers that don't make logical sense.",
        "rigidity": "low",
        "tactics": ["emotional_appeal", "lowball_anchor"],
    },
    {
        "name": "methodical",
        "description": "Negotiates systematically. Researches comparable prices, asks specific questions about specs and condition, builds a case for each offer with concrete reasoning.",
        "rigidity": "medium",
        "tactics": ["comparable_sales", "nitpick_flaws"],  # reuses seller tactic name for comparable_sales — buyer version
    },
    {
        "name": "budget_constrained",
        "description": "Genuinely limited funds. Every dollar matters. Will walk away if the price exceeds what they can afford, not as a tactic but out of necessity.",
        "rigidity": "low",
        "tactics": ["budget_constraint", "emotional_appeal"],
    },
    {
        "name": "expert_informed",
        "description": "Deeply knowledgeable about the product category. Cites specific specs, model years, market data. Hard to bluff or upsell.",
        "rigidity": "high",
        "tactics": ["nitpick_flaws", "competing_offer"],
    },
    {
        "name": "emotional",
        "description": "Decisions driven by feelings — excitement, nostalgia, guilt, anxiety. Can be swayed by stories and rapport, but also prone to buyer's remorse threats.",
        "rigidity": "low",
        "tactics": ["emotional_appeal", "walkaway_threat"],
    },
    {
        "name": "time_pressured",
        "description": "Needs to close the deal quickly. Willing to pay more for speed but will walk if the negotiation drags on. Impatient with back-and-forth.",
        "rigidity": "low",
        "tactics": ["walkaway_threat", "budget_constraint"],
    },
    {
        "name": "delegated_authority",
        "description": "Negotiating on behalf of someone else. Frequently says 'I need to check with my boss/partner.' Uses this to create delay and extract concessions.",
        "rigidity": "medium",
        "tactics": ["budget_constraint", "competing_offer"],
    },
]


RIGIDITY_TEMPLATES = {
    "high":   "You absolutely will not pay more than ${soft_max}, period. This is your hard line.",
    "medium": "You're targeting around ${soft_min}-${soft_max}.",
    "low":    "You're thinking somewhere around ${soft_min}-${soft_max}, but you're not really sure what these go for.",
}


SCENARIOS = [
    # Micro tier ($50–$500)
    {"item": "used laptop",                      "seller_context": "The laptop is 2 years old and in good working condition.",                                                                                          "tier": "micro", "soft_min": 600,   "soft_max": 1000,  "seller_reserve": 800,   "buyer_max": 1000},
    {"item": "vintage vinyl record collection",  "seller_context": "A collection of 50 records from the 1960s-80s, mostly classic rock, all in playable condition with original sleeves.",                              "tier": "micro", "soft_min": 80,    "soft_max": 180,   "seller_reserve": 100,   "buyer_max": 180},
    {"item": "used mountain bike",               "seller_context": "The bike is a 2021 hardtail with hydraulic disc brakes, lightly used for trail riding.",                                                            "tier": "micro", "soft_min": 150,   "soft_max": 260,   "seller_reserve": 200,   "buyer_max": 260},
    {"item": "gaming console bundle",            "seller_context": "A PS5 with two controllers and 5 games, all in original packaging. Bought 18 months ago.",                                                          "tier": "micro", "soft_min": 250,   "soft_max": 420,   "seller_reserve": 300,   "buyer_max": 420},
    {"item": "standing desk",                    "seller_context": "An electric sit-stand desk, 60 inches wide, with programmable height presets. Light scratches on the surface.",                                      "tier": "micro", "soft_min": 180,   "soft_max": 350,   "seller_reserve": 220,   "buyer_max": 350},
    {"item": "espresso machine",                 "seller_context": "A semi-automatic espresso machine with built-in grinder, purchased 1 year ago. Includes extra portafilter baskets.",                                 "tier": "micro", "soft_min": 200,   "soft_max": 400,   "seller_reserve": 250,   "buyer_max": 400},
    {"item": "DSLR camera body",                 "seller_context": "A Canon EOS 90D body only, 15,000 shutter actuations. No lens included. Comes with original box and charger.",                                      "tier": "micro", "soft_min": 350,   "soft_max": 500,   "seller_reserve": 400,   "buyer_max": 500},
    {"item": "electric guitar",                  "seller_context": "A Fender Player Stratocaster in sunburst finish. Minor fret wear, electronics work perfectly. Includes gig bag.",                                    "tier": "micro", "soft_min": 300,   "soft_max": 480,   "seller_reserve": 350,   "buyer_max": 480},
    {"item": "leather sectional couch",          "seller_context": "A 3-piece leather sectional, dark brown, 2 years old. One small tear on the back cushion, not visible when seated.",                                 "tier": "micro", "soft_min": 250,   "soft_max": 450,   "seller_reserve": 300,   "buyer_max": 450},
    {"item": "drone with camera",                "seller_context": "A DJI Mini 3 with carrying case and 3 batteries. 4K video, under 250g. Flown about 20 times.",                                                     "tier": "micro", "soft_min": 250,   "soft_max": 400,   "seller_reserve": 300,   "buyer_max": 400},

    # Small tier ($500–$5K)
    {"item": "freelance web development project", "seller_context": "The project involves building a small e-commerce site with a 4-week timeline. The scope is fixed — no additional features, pages, or extended timelines.", "tier": "small", "soft_min": 2500,  "soft_max": 5000,  "seller_reserve": 3000,  "buyer_max": 5000},
    {"item": "photography session",              "seller_context": "A half-day shoot with edited photos delivered within one week. The scope is fixed — no extended hours, additional locations, or extra deliverables.",  "tier": "small", "soft_min": 350,   "soft_max": 700,   "seller_reserve": 400,   "buyer_max": 700},
    {"item": "logo design package",              "seller_context": "The package includes three initial concepts, two revision rounds, and final files in vector and raster formats. The scope is fixed — no additional concepts, brand guidelines, or collateral design.", "tier": "small", "soft_min": 300, "soft_max": 550, "seller_reserve": 350, "buyer_max": 550},
    {"item": "used riding lawn mower",           "seller_context": "The mower is a 2020 zero-turn with a 42-inch deck, serviced annually, approximately 120 hours of use.",                                              "tier": "small", "soft_min": 1100,  "soft_max": 2000,  "seller_reserve": 1400,  "buyer_max": 2000},
    {"item": "catering for a private dinner party", "seller_context": "A three-course plated dinner for 20 guests including appetizers, mains, and dessert with all tableware provided. The scope is fixed — no additional courses, guests, or bar service.", "tier": "small", "soft_min": 1800, "soft_max": 2700, "seller_reserve": 2200, "buyer_max": 2700},
    {"item": "wedding DJ package",               "seller_context": "A five-hour reception set with professional sound equipment, a wireless microphone for toasts, and a pre-event planning call. The scope is fixed — no additional hours, lighting rigs, or emcee duties.", "tier": "small", "soft_min": 1200, "soft_max": 2800, "seller_reserve": 1500, "buyer_max": 2800},
    {"item": "interior house painting",          "seller_context": "Painting three standard-size rooms including walls and trim, with all materials and prep work included. The scope is fixed — no additional rooms, ceiling work, or wallpaper removal.", "tier": "small", "soft_min": 2800, "soft_max": 4500, "seller_reserve": 3500, "buyer_max": 4500},
    {"item": "personal training package",        "seller_context": "20 one-hour sessions over 10 weeks with a certified trainer, including an initial fitness assessment and customized workout plan.",                    "tier": "small", "soft_min": 800,   "soft_max": 1600,  "seller_reserve": 1000,  "buyer_max": 1600},
    {"item": "used hot tub",                     "seller_context": "A 6-person hot tub, 3 years old, recently resealed. Buyer responsible for pickup and installation. Includes cover and steps.",                        "tier": "small", "soft_min": 1500,  "soft_max": 3000,  "seller_reserve": 2000,  "buyer_max": 3000},
    {"item": "professional video editing project", "seller_context": "Editing a 10-minute corporate video from raw footage, including color grading, titles, and two rounds of revisions. Delivery within 2 weeks.",      "tier": "small", "soft_min": 1200,  "soft_max": 2500,  "seller_reserve": 1500,  "buyer_max": 2500},

    # Medium tier ($5K–$20K)
    {"item": "used car",                         "seller_context": "The car is a 2019 sedan with 45,000 miles, one previous owner, clean title.",                                                                        "tier": "medium", "soft_min": 10000, "soft_max": 16000, "seller_reserve": 12000, "buyer_max": 16000},
    {"item": "used fishing boat with trailer",   "seller_context": "A 16-foot aluminum fishing boat with a 40 HP outboard motor, galvanized trailer, and fish finder. The motor was rebuilt last season.",                "tier": "medium", "soft_min": 7000,  "soft_max": 11000, "seller_reserve": 9000,  "buyer_max": 11000},
    {"item": "kitchen remodel labor",            "seller_context": "Labor only for a standard kitchen remodel: demolition, cabinet installation, countertop templating, backsplash tiling, and plumbing hookups. Materials supplied by homeowner. 3-week timeline.", "tier": "medium", "soft_min": 6000, "soft_max": 12000, "seller_reserve": 8000, "buyer_max": 12000},
    {"item": "used travel trailer",              "seller_context": "A 2020 travel trailer, 24 feet, sleeps 4, fully self-contained with AC and awning. 12,000 miles towed. Minor cosmetic wear inside.",                  "tier": "medium", "soft_min": 10000, "soft_max": 18000, "seller_reserve": 13000, "buyer_max": 18000},
    {"item": "website redesign project",         "seller_context": "Full redesign of an existing 20-page business website including responsive design, CMS migration, and SEO audit. 6-week timeline, fixed scope.",      "tier": "medium", "soft_min": 6000,  "soft_max": 15000, "seller_reserve": 8000,  "buyer_max": 15000},
    {"item": "used grand piano",                 "seller_context": "A Yamaha C3 grand piano, 15 years old, regularly tuned. Buyer responsible for moving. Bench included.",                                               "tier": "medium", "soft_min": 8000,  "soft_max": 14000, "seller_reserve": 10000, "buyer_max": 14000},
    {"item": "commercial photography contract",  "seller_context": "A 3-day product photography shoot for an e-commerce catalog: 200 SKUs, white background, 3 angles each. Includes editing and retouching.",            "tier": "medium", "soft_min": 5000,  "soft_max": 10000, "seller_reserve": 6500,  "buyer_max": 10000},
    {"item": "HVAC system replacement",          "seller_context": "Full replacement of a residential central AC and furnace system including ductwork inspection, permits, and installation. 2-day job.",                 "tier": "medium", "soft_min": 6000,  "soft_max": 12000, "seller_reserve": 8000,  "buyer_max": 12000},
    {"item": "used side-by-side UTV",            "seller_context": "A 2021 Polaris RZR 900, 1,200 miles, aftermarket bumper and light bar. Garage-kept, no accidents.",                                                   "tier": "medium", "soft_min": 8000,  "soft_max": 14000, "seller_reserve": 10000, "buyer_max": 14000},
    {"item": "custom software integration",      "seller_context": "Integrating an existing CRM with the client's ERP system via REST APIs. Includes data mapping, testing, and documentation. Fixed scope, 4-week timeline.", "tier": "medium", "soft_min": 7000, "soft_max": 15000, "seller_reserve": 9000, "buyer_max": 15000},

    # Large tier ($20K–$100K)
    {"item": "backyard deck construction",       "seller_context": "A 300 square foot pressure-treated wood deck with railing and a single set of stairs, including permits and materials. The scope is fixed — no composite upgrades, built-in seating, or electrical work.", "tier": "large", "soft_min": 18000, "soft_max": 35000, "seller_reserve": 22000, "buyer_max": 35000},
    {"item": "commercial kitchen equipment package", "seller_context": "A complete used commercial kitchen setup: 6-burner range, convection oven, reach-in fridge and freezer, prep tables, and exhaust hood. All NSF-certified, 3 years old.", "tier": "large", "soft_min": 18000, "soft_max": 35000, "seller_reserve": 22000, "buyer_max": 35000},
    {"item": "basement finishing project",       "seller_context": "Finishing a 1,000 sq ft basement including framing, drywall, flooring, egress window, and electrical. Permits included. No bathroom or wet bar.",       "tier": "large", "soft_min": 20000, "soft_max": 40000, "seller_reserve": 25000, "buyer_max": 40000},
    {"item": "used box truck",                   "seller_context": "A 2018 Isuzu NPR 16-foot box truck, 85,000 miles, diesel, liftgate. Fleet-maintained with full service records.",                                     "tier": "large", "soft_min": 20000, "soft_max": 35000, "seller_reserve": 24000, "buyer_max": 35000},
    {"item": "annual IT consulting retainer",    "seller_context": "A 12-month IT consulting retainer: 20 hours/month of on-call support, quarterly security audits, and priority response SLA. Fixed scope.",             "tier": "large", "soft_min": 25000, "soft_max": 50000, "seller_reserve": 30000, "buyer_max": 50000},
    {"item": "home solar panel installation",    "seller_context": "A 10kW residential solar panel system including panels, inverter, mounting, permits, and grid interconnection. 25-year panel warranty.",                "tier": "large", "soft_min": 18000, "soft_max": 32000, "seller_reserve": 22000, "buyer_max": 32000},
    {"item": "used excavator",                   "seller_context": "A 2019 Kubota KX040 mini excavator, 2,400 hours, with thumb attachment and 3 bucket sizes. Recent track replacement.",                                "tier": "large", "soft_min": 25000, "soft_max": 45000, "seller_reserve": 30000, "buyer_max": 45000},
    {"item": "commercial office buildout",       "seller_context": "Tenant improvement buildout for a 2,000 sq ft office: framing, drywall, flooring, lighting, and HVAC tie-in. Permits included. No furniture.",         "tier": "large", "soft_min": 30000, "soft_max": 60000, "seller_reserve": 38000, "buyer_max": 60000},
    {"item": "professional marketing campaign",  "seller_context": "A 6-month digital marketing campaign: strategy, content creation, paid ad management, and monthly reporting. Fixed deliverables, no media spend included.", "tier": "large", "soft_min": 20000, "soft_max": 45000, "seller_reserve": 28000, "buyer_max": 45000},
    {"item": "used food truck",                  "seller_context": "A 2020 custom food truck with commercial kitchen equipment, generator, and serving window. Health department certified. 30,000 miles.",                "tier": "large", "soft_min": 35000, "soft_max": 65000, "seller_reserve": 42000, "buyer_max": 65000},

    # Enterprise tier ($100K+)
    {"item": "commercial building lease (annual)", "seller_context": "A 5,000 sq ft retail space in a mid-traffic shopping center. Triple-net lease, 3-year minimum term. Tenant responsible for buildout.",                "tier": "enterprise", "soft_min": 80000,  "soft_max": 150000, "seller_reserve": 100000, "buyer_max": 150000},
    {"item": "fleet vehicle purchase (10 vans)",   "seller_context": "Ten 2022 Ford Transit cargo vans, fleet-maintained, average 40,000 miles each. Sold as a lot only, not individually.",                               "tier": "enterprise", "soft_min": 180000, "soft_max": 300000, "seller_reserve": 220000, "buyer_max": 300000},
    {"item": "commercial construction project",    "seller_context": "Ground-up construction of a 3,000 sq ft commercial building: foundation, framing, roofing, electrical, plumbing, and finish work. Permits and plans included.", "tier": "enterprise", "soft_min": 250000, "soft_max": 450000, "seller_reserve": 300000, "buyer_max": 450000},
    {"item": "enterprise software license deal",   "seller_context": "A 3-year enterprise license for a warehouse management system: 50 user seats, implementation, training, and first-year support included.",           "tier": "enterprise", "soft_min": 120000, "soft_max": 250000, "seller_reserve": 150000, "buyer_max": 250000},
    {"item": "industrial equipment package",       "seller_context": "A complete CNC machining cell: 3-axis CNC mill, CNC lathe, tooling package, and installation. 2 years old, under 3,000 hours each.",                 "tier": "enterprise", "soft_min": 150000, "soft_max": 280000, "seller_reserve": 180000, "buyer_max": 280000},
    {"item": "warehouse lease (annual)",           "seller_context": "A 15,000 sq ft warehouse with loading dock, drive-in door, and climate-controlled office space. 2-year minimum term.",                                "tier": "enterprise", "soft_min": 100000, "soft_max": 180000, "seller_reserve": 120000, "buyer_max": 180000},
    {"item": "commercial solar installation",      "seller_context": "A 100kW commercial rooftop solar array including panels, inverters, racking, permits, and grid interconnection for a warehouse facility.",            "tier": "enterprise", "soft_min": 130000, "soft_max": 220000, "seller_reserve": 160000, "buyer_max": 220000},
    {"item": "restaurant franchise buildout",      "seller_context": "Complete buildout of a fast-casual restaurant franchise location: kitchen equipment, dining area, signage, permits, and contractor management. Franchise fee not included.", "tier": "enterprise", "soft_min": 200000, "soft_max": 380000, "seller_reserve": 250000, "buyer_max": 380000},
    {"item": "managed IT infrastructure contract", "seller_context": "A 3-year managed IT contract: 24/7 monitoring, helpdesk for 100 employees, server management, cybersecurity, and disaster recovery. Fixed scope.",    "tier": "enterprise", "soft_min": 150000, "soft_max": 300000, "seller_reserve": 200000, "buyer_max": 300000},
    {"item": "commercial HVAC retrofit",           "seller_context": "Full HVAC retrofit for a 20,000 sq ft commercial building: rooftop units, ductwork, controls, and building automation integration. Permits included.", "tier": "enterprise", "soft_min": 120000, "soft_max": 250000, "seller_reserve": 160000, "buyer_max": 250000},
]

SCENARIOS_BY_TIER = {}
for s in SCENARIOS:
    SCENARIOS_BY_TIER.setdefault(s["tier"], []).append(s)

TIERS = ["micro", "small", "medium", "large", "enterprise"]
PERSONAS_BY_NAME = {p["name"]: p for p in PERSONAS}


PERSONA_SUFFIX = (
    "\n\nYou ARE this buyer. Respond only with what you would actually say to the seller — "
    "a single short message of natural spoken dialogue. "
    "NO bullet points, headers, options, analysis, tactics commentary, or meta-text of any kind. "
    "Just the words you say."
)


def make_seller_prompt(scenario, tactics=None):
    base = (
        f"You are Seller A. Your goal is to maximize sale price while still closing the deal.\n\n"
        f"Your reservation price is ${scenario['seller_reserve']}. Do not reveal it.\n\n"
        f"Scenario:\nYou are negotiating the sale of a {scenario['item']}.\n"
        f"{scenario['seller_context']}\n\n"
        f"You are only authorized to sell the {scenario['item']} described above. "
        f"Do not offer gift cards, credits, bonuses, free items, or anything other than the {scenario['item']} itself. "
        f"Do not offer add-ons, extras, or concessions of any kind other than adjusting the price. "
        f"Negotiate on price only.\n\n"
    )
    if tactics:
        tactic_block = "Tactics to use in this negotiation:\n"
        for t in tactics:
            tactic_block += f"- {t['instruction']}\n"
        base += tactic_block + "\n"
    base += (
        f"Write your next message only. One to three sentences of natural dialogue. "
        f"Do not start your message with 'You:', 'Seller:', 'Buyer:', or any other label or prefix. "
        f"Do not write the buyer's response. Stop after your own message ends."
    )
    return base


def make_buyer_prompt(persona, scenario, tactics=None):
    template = RIGIDITY_TEMPLATES[persona["rigidity"]]
    price_guidance = template.format(soft_min=scenario["soft_min"], soft_max=scenario["soft_max"])

    prompt = (
        f"You are Buyer B negotiating to buy a {scenario['item']}. "
        f"{persona['description']} "
        f"{price_guidance}"
    )
    if tactics:
        prompt += "\n\nTactics to use in this negotiation:\n"
        for t in tactics:
            prompt += f"- {t['instruction']}\n"
    prompt += PERSONA_SUFFIX
    return prompt
