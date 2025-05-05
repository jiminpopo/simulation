import numpy as np
import pandas as pd
from scipy.stats import norm, expon, beta, t
import os

def generate_simulation_data2(n=1000, seed=42, save_path="data/s2.csv"):
    """
    Generate simulation data with correlated Gaussian inputs and transformed marginals.
    
    Parameters:
    - n (int): Number of samples.
    - seed (int): Random seed for reproducibility.
    - save_path (str): Output path for the generated CSV.
    
    Returns:
    - df_sim (pd.DataFrame): Simulated data with 5 variables.
    """
    np.random.seed(seed)

    # Define mean and covariance
    mean = np.zeros(5)
    cov = [
        [1.0,  0.8, -0.4,  0.2,  0.0],
        [0.8,  1.0, -0.2,  0.4,  0.1],
        [-0.4, -0.2, 1.0, -0.5, -0.3],
        [0.2,  0.4, -0.5, 1.0,  0.6],
        [0.0,  0.1, -0.3, 0.6,  1.0]
    ]

    # Generate multivariate normal data
    Z = np.random.multivariate_normal(mean, cov, size=n)

    # Transform marginals
    x1 = Z[:, 0]  # Standard normal
    x2_u = norm.cdf(Z[:, 1])  # Convert to uniform for mode-based mapping
    x2 = np.select(
        [x2_u < 0.1, x2_u < 0.3, x2_u < 0.6, x2_u < 0.85, x2_u <= 1.0],
        [np.random.normal(5, 1, n),
         np.random.normal(15, 1.5, n),
         np.random.normal(25, 1, n),
         np.random.normal(35, 1.2, n),
         np.random.normal(45, 1, n)]
    )

    x3 = expon.ppf(norm.cdf(Z[:, 2]), scale=2.0)
    x4 = beta.ppf(norm.cdf(Z[:, 3]), a=2, b=5)
    x5 = t.ppf(norm.cdf(Z[:, 4]), df=2)

    # Combine into DataFrame
    df_sim = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

    # Ensure data folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to CSV
    df_sim.to_csv(save_path, index=False)

    return df_sim

if __name__ == "__main__":
    generate_simulation_data2()
