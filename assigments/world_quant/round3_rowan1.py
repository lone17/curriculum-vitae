# Definition:
#   A "drawdown" occurs when a stock price falls, and it equals the percent drop in price.
#   If start_price > end_price, then drawdown = (start_price - end_price) / start_price.
#   If price does not fall, then drawdown = 0.
#
# Problem:
#   Suppose you are given a sequence of stock prices (1 stock, prices in chronological order).
#   Write a function that computes the largest drawdown.
#
# Examples:
# >>> max_drawdown([99, 100])
# 0
# >>> max_drawdown([99, 100, 98])
# 0.02
# >>> max_drawdown([99, 100, 98, 99, 97])
# 0.03

import math
import multiprocessing as mp
import time

import numpy as np


def max_drawdown(prices):
    n = len(prices)
    tail_min = []
    cur_min = math.inf
    ans = 0
    for i in range(n - 1, -1, -1):
        if prices[i] < cur_min:
            cur_min = prices[i]
        tail_min.append(cur_min)

    tail_min = list(reversed(tail_min))

    ans = 0
    for i, p in enumerate(prices):
        if p == 0:
            continue
        ans = max(ans, (p - tail_min[i]) / p)

    return ans


def validate_prices(prices):
    for p in prices:
        if p <= 0:
            raise ValueError("Price value cannot be <= 0.")


def max_drawdown_2(prices):
    validate_prices(prices)

    max_idx = 0
    min_idx = 0
    ans = 0
    for i, p in enumerate(prices):
        if p > prices[max_idx]:
            max_idx = i
        if p < prices[min_idx]:
            min_idx = i
        if min_idx > max_idx:
            ans = max(ans, (prices[max_idx] - prices[min_idx]) / prices[max_idx])

    return ans


print(max_drawdown_2([99, 100, 500, 250]))
# 0
# print(max_drawdown_2([99, 100]))
# # 0
# print(max_drawdown_2([99, 100, 98]))
# # 0.02
# print(max_drawdown_2([99, 100, 98, 99, 97]))
# # 0.03
# print(max_drawdown_2([99, 100, 500, 250]))
# # 0.5

# Follow up: suppose you are given prices for multiple stocks over the same time range.
# How would you compute the largest drawdown per stock?

# n stocks and d dates


def max_drawdown_single(prices):
    running_max = 0
    ans = 0
    for p in prices:
        if p > running_max:
            running_max = p
        else:
            ans = max(ans, (running_max - p) / running_max)

    return ans


def max_drawdown_mp(prices):
    with mp.Pool(4) as pool:
        ans = pool.map(max_drawdown_single, prices)

    return ans


def max_drawdown_np(prices):
    running_max = np.maximum.accumulate(prices, axis=1)

    ans = (running_max - prices) / running_max
    ans = ans.max(axis=1)

    return ans


if __name__ == "__main__":
    elapses_mp = []
    elapses_np = []
    for _ in range(100):
        prices = np.random.randint(1, 1000, size=(32, 10**6))

        start_time = time.time()
        np_results = max_drawdown_np(prices)
        elapses_np.append(time.time() - start_time)

        start_time = time.time()
        mp_results = max_drawdown_mp(prices)
        elapses_mp.append(time.time() - start_time)

    print(f"np: avg - {np.mean(elapses_np):.4f}, std - {np.std(elapses_np):.4f}")
    # np: avg - 0.3066, std - 0.0249
    print(f"mp: avg - {np.mean(elapses_mp):.4f}, std - {np.std(elapses_mp):.4f}")
    # mp: avg - 4.7190, std - 0.4964

    # print(mp_results == list(np_results))
