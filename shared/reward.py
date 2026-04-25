def price_reward(agreed_price, reserve, ceiling):
    if agreed_price is None:
        return 0.0
    if agreed_price < reserve:
        return -1.0
    return min(1.0, (agreed_price - reserve) / (ceiling - reserve))
