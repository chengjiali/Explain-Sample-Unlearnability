import torch
import numpy as np
import pandas as pd
from collections import Counter


methods = ["GradDiff", "NPO", "SimNPO", "RMU", "UNDIAL"]
size = 50
min_appearence = 3


r = []
forget_pct = 'forget10'
for method in methods:
    path = f'open-unlearning/saves/sample_difficulty/tofu/Llama-3.2-1B-Instruct/{forget_pct}/{method}/loss.pt'
    loss = torch.load(path, weights_only=True)
    res = pd.DataFrame({'idx': range(len(loss)), 'loss': loss.tolist(), 'df': forget_pct, 'method': method})

    r.append(res)

r = pd.concat(r, ignore_index=True)

order = r.groupby('method')['loss'].apply(lambda x: np.argsort(x)).droplevel(1).reset_index(name='order')
order.to_csv('sample_difficulty/tofu/loss.csv', index=None)


# Common easy and hard
order_mat = order['order'].values.reshape(-1, 400)

unique_values, counts = np.unique(order_mat[:, -size:], return_counts=True)
common_easy = unique_values[counts >= min_appearence]

unique_values, counts = np.unique(order_mat[:, :size], return_counts=True)
common_hard = unique_values[counts >= min_appearence]


with open(f'sample_difficulty/tofu/common_easy_{size}.txt', 'w') as f:
    f.write(','.join([str(i) for i in common_easy]))

with open(f'sample_difficulty/tofu/common_hard_{size}.txt', 'w') as f:
    f.write(','.join([str(i) for i in common_hard]))