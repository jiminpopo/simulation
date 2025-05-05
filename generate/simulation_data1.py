import numpy as np
import pandas as pd

def generate_simulation_data(n_samples=1000, ratio=0.3, seed=42, save_path="data/s1.csv"):
    """
    Generate simulation data with two categories (A and B) from multivariate normal distributions.

    Parameters:
    - n_samples (int): Total number of samples.
    - ratio (float): Proportion of category A (0 < ratio < 1).
    - seed (int): Random seed for reproducibility.
    - save_path (str): Path to save the generated CSV file.
    """

    # Set random seed
    np.random.seed(seed)

    # Number of samples for each category
    n_A = int(n_samples * ratio)
    n_B = n_samples - n_A

    # Means and covariances for each group
    mean_A = [30, 10]
    mean_B = [50, 40]
    cov_A = [[10, 3], [3, 10]]
    cov_B = [[10, -2], [-2, 10]]

    # Sample from multivariate normal distributions
    num_data_A = np.random.multivariate_normal(mean_A, cov_A, n_A)
    num_data_B = np.random.multivariate_normal(mean_B, cov_B, n_B)

    # Category labels
    category_A = ['A'] * n_A
    category_B = ['B'] * n_B

    # Combine data
    num_data = np.vstack([num_data_A, num_data_B])
    categories = category_A + category_B

    # Create DataFrame
    df = pd.DataFrame(num_data, columns=['num1', 'num2'])
    df['category'] = categories

    # Save to CSV
    df.to_csv(save_path, index=False)

    return df

if __name__ == "__main__":
    generate_simulation_data()
