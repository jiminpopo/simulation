import numpy as np
import pandas as pd
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_simulation_boxplot_data(n_per_group=100, seed=42, mode='random', save_path=None):
    """
    Generate simulation data with cat2 determined by the selected mode.

    Parameters:
    - n_per_group (int): Samples per cat1 level
    - seed (int): Random seed for reproducibility
    - mode (str): 'random' | 'rule' | 'logistic'
    - save_path (str or None): Path to save the generated CSV file. If None, auto-generates filename.

    Returns:
    - pd.DataFrame: Simulated data with 5 columns: num1, num2, num3, cat1, cat2
    """
    np.random.seed(seed)

    levels_cat1 = ['A', 'B', 'C', 'D']
    data = []

    for cat1 in levels_cat1:
        mean_num1 = {'A': 10, 'B': 20, 'C': 30, 'D': 40}[cat1]
        mean_num2 = {'A': 40, 'B': 30, 'C': 20, 'D': 10}[cat1]
        mean_num3 = {'A': 35, 'B': 25, 'C': 15, 'D': 30}[cat1]

        for _ in range(n_per_group):
            num1 = np.random.normal(loc=mean_num1, scale=5)
            num2 = np.random.normal(loc=mean_num2, scale=5)
            num3 = np.random.normal(loc=mean_num3, scale=4)

            # === cat2 generation based on selected mode ===
            if mode == 'random':
                cat2 = np.random.choice(['a', 'b'])

            elif mode == 'rule':
                cat2 = 'a' if num1 > num2 else 'b'

            elif mode == 'logistic':
                beta_0 = -5
                beta_1 = 0.08
                beta_2 = -0.06
                beta_3 = 0.04
                logit = beta_0 + beta_1 * num1 + beta_2 * num2 + beta_3 * num3
                p_a = sigmoid(logit)
                cat2 = np.random.choice(['a', 'b'], p=[p_a, 1 - p_a])

            else:
                raise ValueError("Invalid mode. Choose from ['random', 'rule', 'logistic'].")

            data.append([num1, num2, num3, cat1, cat2])

    df = pd.DataFrame(data, columns=['num1', 'num2', 'num3', 'cat1', 'cat2'])

    # Auto-generate save path if not provided
    if save_path is None:
        save_path = f"data/s3_{mode}.csv"

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    df.to_csv(save_path, index=False)

    return df

if __name__ == "__main__":
    generate_simulation_boxplot_data(mode='random')
