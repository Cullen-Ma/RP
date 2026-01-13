import torch
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import r2_score
from utils import check_dir, save_torch_npy

class ElasticNetRegression:
    def __init__(self, alpha=0.1, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.intercept_ = 0

    def fit(self, X, y):
        # X: [n_features, n_samples] (Transposed in logic below), y: [n_samples]
        # Objective: minimize RMSE
        n_features = X.shape[0] 
        
        def fun(w):
            pred = np.dot(X.T, w) 
            error = pred - y
            rmse_loss = np.sqrt(np.mean(error ** 2))
            
            # Simple ElasticNet Penalty (optional, depends on implementation needs)
            # l1_reg = self.alpha * self.l1_ratio * np.sum(np.abs(w))
            # l2_reg = self.alpha * (1 - self.l1_ratio) * np.sum(np.square(w))
            return rmse_loss 

        w0 = np.ones(n_features)
        # BFGS is standard for unconstrained smooth optimization
        res = minimize(fun, w0, method='BFGS', options={'maxiter': 500})
        self.coef_ = res.x
        return res.x

def run_regression(data_path, mask_path, avg_tac_path, output_dir, device_name):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Load Data
    img_full = np.load(data_path)
    if mask_path:
        mask = np.load(mask_path)
        img_masked = img_full[:, mask == 1].T # [Voxels, Frames]
    else:
        img_masked = img_full
    
    avg_tac = np.load(avg_tac_path) # [Clusters, Frames]
    
    num_voxels, num_frames = img_masked.shape
    num_clusters = avg_tac.shape[0]

    # Convert to standard numpy for scipy optimization
    z_ij_estimate = np.zeros((num_voxels, num_frames))
    alpha_coeffs = np.zeros((num_voxels, num_clusters))
    
    print("Fitting GLM mean (Regression)...")
    for j in tqdm(range(num_voxels)):
        y_vec = img_masked[j, :]
        
        model = ElasticNetRegression()
        coefs = model.fit(avg_tac, y_vec) # Fits y ~ avg_tac.T * w
        
        z_est = np.dot(avg_tac.T, coefs)
        
        z_ij_estimate[j, :] = z_est
        alpha_coeffs[j, :] = coefs

    # Compute Residuals
    residual = img_masked - z_ij_estimate
    
    check_dir(output_dir)
    np.save(os.path.join(output_dir, 'z_ij_estimate.npy'), z_ij_estimate)
    np.save(os.path.join(output_dir, 'alpha_coeffs.npy'), alpha_coeffs)
    np.save(os.path.join(output_dir, 'residuals_raw.npy'), residual)
    print("Regression complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--avg_tac', type=str, required=True, help='Path to avg_tac.npy from Step 1')
    parser.add_argument('--out', type=str, default='./results/step2')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    run_regression(args.data, args.mask, args.avg_tac, args.out, args.device)
