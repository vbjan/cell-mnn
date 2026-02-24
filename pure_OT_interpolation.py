from lib.metrics import compute_wasserstein
from lib.metrics.emd import compute_ot_coupling
from lib.data.data_preprocessing import get_data
from lib.utils import fix_seed

import numpy as np

SEED = 43
MAX_OT_ITER = 1_000_000
fix_seed(SEED)

data_dir = get_data(ds_name="embryoid", val_prop=0.0)
X_train = data_dir["X_train"]
days = data_dir["t_train"]

skip_days = days[1: -1]
assert len(skip_days) == len(days) - 2, "There should be one less skip day than total days."
print(f"days: {days} \n skip_days: {skip_days}")

w1_values = []

for i, day in enumerate(days):
    if day in skip_days:
        print(f"Processing skip day: {day}")
        prev_day = days[i - 1]
        next_day = days[i + 1]
        x_prev_day = X_train[i - 1]
        x_skip_day = X_train[i]
        x_next_day = X_train[i + 1]

        print(f"x_prev_day shape: {x_prev_day.shape}, x_next_day shape: {x_next_day.shape}")
        
        paired_next = compute_ot_coupling(x_prev_day, x_next_day, num_itermax=MAX_OT_ITER)

        print(f"paired_next shape: {paired_next.shape}")

        # Linear interpolation between previous and next day
        linear_interpolated = x_prev_day + (day - prev_day) / (next_day - prev_day) * (paired_next - x_prev_day)

        w1 = compute_wasserstein(x_skip_day, linear_interpolated, num_iter_max=MAX_OT_ITER)
        w1_values.append(w1)
        print(f"Wasserstein distance between x_skip_day and linear_interpolated: {w1}")

print(f"Average Wasserstein distance over all skip days: {np.mean(w1_values)} +- {np.std(w1_values)}")

