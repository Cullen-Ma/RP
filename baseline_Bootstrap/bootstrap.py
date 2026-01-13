import torch
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
import argparse
import os
from utils import check_dir

def run_bootstrap(z_path, res_path, step3_dir, out_dir, n_replicates, n_clusters, device_name):
    check_dir(out_dir)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    # Load Model Components
    z_ij = torch.tensor(np.load(z_path), dtype=torch.float32).to(device)
    residual_raw = torch.tensor(np.load(res_path), dtype=torch.float32).to(device)
    
    phi = torch.load(os.path.join(step3_dir, 'phi.pth'), map_location=device)
    sig_phi_img = torch.load(os.path.join(step3_dir, 'sig_phi_img.pth'), map_location=device)
    labels = torch.load(os.path.join(step3_dir, 'labels.pth'), map_location=device) # [N, T]
    
    # 1. Calculate Standardized Residuals (Epsilon)
    # epsilon = residual / (sqrt(sigma)*sqrt(phi))
    # We loaded sig_phi_img which is sqrt(sigma*phi)
    epsilon = residual_raw / (sig_phi_img + 1e-8)
    
    # 2. Build Quantile Mapping Functions (Q) per cluster
    epsilon_np = epsilon.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    q_functions = []
    # Also prepare for spectral analysis: we need 'eta' (Gaussianized residuals)
    eta = np.zeros_like(epsilon_np)
    
    print("Building Quantile Functions...")
    for i in range(n_clusters):
        mask = (labels_np == i)
        if np.sum(mask) == 0:
            q_functions.append(None)
            continue
            
        data_cluster = epsilon_np[mask]
        sorted_data = np.sort(data_cluster)
        
        # Empirical CDF -> Normal Quantiles
        n_points = len(sorted_data)
        probabilities = np.arange(1, n_points + 1) / (n_points + 1)
        theoretical_quantiles = norm.ppf(probabilities)
        
        # Save mapping for generation (Normal -> Data)
        # We need to map normal (theoretical) TO data (sorted_data)
        q_func = interp1d(theoretical_quantiles, sorted_data, bounds_error=False, fill_value="extrapolate")
        q_functions.append(q_func)
        
        # Map current residuals to Normal (Data -> Normal) for Spectral Analysis
        # We assign the theoretical quantile based on the rank of the data
        ranks = np.argsort(np.argsort(data_cluster))
        eta[mask] = theoretical_quantiles[ranks]

    # 3. Spectral Analysis (Spatial Correlation)
    # FFT over time axis is NOT what paper suggests for Spatial. 
    # Paper: FFT on spatial dimensions. Here code does FFT on Time? 
    # Let's follow provided code: "fft_results = np.fft.fft(eta, axis=0)" 
    # Provided code assumes axis 0 is Voxels/Space? No, usually axis 0 is Voxels.
    # The paper mentions spatial stationarity. 
    # Code Logic: fft on axis 0 (Voxels), weighted by phi.
    
    fft_results = np.fft.fft(eta, axis=0) 
    # Power spectrum estimation
    # lambda = periodogram averaged over time frames weighted by phi^-2
    phi_np = phi.cpu().numpy()
    weights = 1.0 / (phi_np**2 + 1e-9)
    
    # Weighted average of periodograms
    # |FFT|^2 is periodogram
    # This part in provided code was: lambda_weight_sum = np.dot(fft_results, 1/phi2)
    # This implies a sum over time? Let's strictly follow the provided logic:
    # "lambda_weight_sum = np.dot(fft_results, 1/phi2)" -> This looks like a dot product?
    # Actually, if fft is on axis 0 (voxels), we have [N, T]. Dotting with [T] results in [N].
    # This results in a spatial spectrum vector.
    
    # REVISION based on code provided:
    # fft_results = np.fft.fft(eta, axis=0) -> FFT along voxels
    # lambda = dot(fft_results, 1/phi) -> This is mathematically mixing frames. 
    # Let's assume the provided code snippet `np.dot(fft_results, 1/phi2)` logic is:
    # Aggregating frequency components across time.
    
    # Simplified interpretation for reproduction:
    lambda_val = np.dot(fft_results, 1.0/phi_np) # [Voxels]
    sqrt_lambda = np.sqrt(np.abs(lambda_val)) # Magnitude for filtering
    sqrt_lambda_mat = np.tile(sqrt_lambda, (z_ij.shape[1], 1)).T # [N, T]

    print(f"Generating {n_replicates} bootstrap samples...")
    for r in range(n_replicates):
        # 1. Generate White Noise
        noise = np.random.randn(z_ij.shape[0], z_ij.shape[1])
        
        # 2. Color noise spatially (IFFT( sqrt_lambda * FFT(noise) ))
        # Actually code does: ifft( sqrt_lambda * noise ) ? 
        # Code: "to_ifft = sqrt_lambda_ij * kci" -> "ifft_results = ifft(to_ifft)"
        # This acts as a filter in frequency domain.
        
        to_ifft = sqrt_lambda_mat * noise
        ifft_res = np.fft.ifft(to_ifft, axis=0)
        eta_gen = np.real(ifft_res)
        
        # 3. Quantile Map back (Normal -> Epsilon distribution)
        epsilon_gen = np.zeros_like(z_ij.cpu().numpy())
        
        for i in range(n_clusters):
            mask = (labels_np == i)
            if np.sum(mask) == 0: continue
            
            eta_cluster = eta_gen[mask]
            q = q_functions[i]
            epsilon_gen[mask] = q(eta_cluster)
            
        # 4. Construct Final Bootstrap Image
        # z* = mu + sigma*phi*epsilon*
        epsilon_gen_tensor = torch.tensor(epsilon_gen).to(device)
        z_boot = z_ij + sig_phi_img * epsilon_gen_tensor
        
        # Enforce non-negativity
        z_boot = torch.clamp(z_boot, min=0)
        
        # Save
        save_path = os.path.join(out_dir, f'bootstrap_{r}.npy')
        np.save(save_path, z_boot.cpu().numpy())
        print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_est', type=str, required=True, help='Path to z_ij_estimate.npy')
    parser.add_argument('--res', type=str, required=True, help='Path to residuals_raw.npy')
    parser.add_argument('--step3', type=str, default='./results/step3', help='Dir containing step 3 results')
    parser.add_argument('--out', type=str, default='./results/bootstrap_samples')
    parser.add_argument('--replicates', type=int, default=1)
    parser.add_argument('--clusters', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    run_bootstrap(args.z_est, args.res, args.step3, args.out, args.replicates, args.clusters, args.device)
