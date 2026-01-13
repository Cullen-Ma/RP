# Image-Domain Bootstrapping for PET Data

This repository contains a PyTorch/NumPy implementation of the method described in **"A Generalized Linear modeling approach to bootstrapping multi-frame PET image data"**.

The method allows for the generation of bootstrap replicates of dynamic PET data directly in the image domain, avoiding the computationally expensive reconstruction of list-mode data.

## Prerequisites

*   Python 3.8+
*   Numpy
*   Pandas
*   Scipy
*   Scikit-learn
*   PyTorch
*   Tqdm

Install dependencies:
```bash
pip install numpy pandas scipy scikit-learn torch tqdm

Run in sequence:
1. cluster.py
2. elastic_regression_simulation_final_bootstrap.py
3. iterate_sigma_phi_h.py
4. bootstrap.py
