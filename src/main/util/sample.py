"""采样，得到候选点集"""
import random

def get_candidates(nodes, N: int=100, L: int=100) -> list:
    candidates = []
    for i in range(N):
        candidates.append(random.sample(nodes, L))
    return candidates
    