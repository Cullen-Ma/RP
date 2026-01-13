# Bayesian 2TCM Kinetic Modeling (MCMC)

This repository implements a Bayesian inference framework for voxel-wise kinetic modeling of **$^{18}$F-FDG PET** data. It uses the **Metropolis-Hastings** algorithm to estimate physiological parameters based on the **2-Tissue Compartment Model (2TCM)** with an irreversible trapping constraint ($k_4=0$).

The statistical framework is strictly adapted from the methodology described in *Kinetic parameter estimation of hepatocellular carcinoma on 18F-FDG PET/CT based on Bayesian method*.

## Key Features

*   **Physiological Model:** 2TCM with $k_4$ strictly constrained to 0 (Irreversible).
*   **Statistical Framework:**
    *   **Algorithm:** Metropolis-Hastings MCMC.
    *   **Iterations:** 10,000 samples (800 burn-in), strictly following the reference paper.
    *   **Prior:** Gaussian distribution with standard deviations derived from the **population statistics** of the initial NLLS inputs (not hardcoded).
    *   **Likelihood:** Gaussian noise model with $\sigma$ estimated dynamically from the **residuals** of the initial fit for each voxel.
*   **Performance:** Implements `multiprocessing` for parallel voxel-wise computation.

## Dependencies

*   Python 3.x
*   numpy
*   scipy
*   matplotlib
*   tqdm

## Usage

1.  **Data Preparation:** Ensure your input data (images, segmentation, input function, and initial NLLS parameters) are located at the paths specified in the `__main__` block (e.g., `/share/home/kma/...`).
2.  **Execution:**
    ```bash
    python main.py
    ```
3.  **Output:**
    *   Posterior samples are saved as `.npy` files in the `samples_strict_paper/` directory.
    *   Files are named `sample_batch_{id}.npy`.

## Notes on Configuration

*   **Noise Estimation (INDICATOR):** The code automatically calculates the noise level ($\sigma$) for the likelihood function based on the Residual Sum of Squares (RSS) of the initial parameters.
*   **Prior Width (INDICATOR):** The width of the prior distribution is automatically calculated based on the standard deviation of the input parameter group.
