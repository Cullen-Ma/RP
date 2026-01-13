import torch
import numpy as np
import argparse
import os
from utils import check_dir

def iteration_step(z_ij, residual, sigma, phi, h, n_clusters, device):
    # Ensure all inputs are tensors on device
    
    # 1. Update Phi (Temporal Variance)
    # w_{ij} = 1 / (sigma_i * h)
    # phi_j = sum_i (w * r^2) / N
    sigma_exp = sigma.unsqueeze(1).repeat(1, h.shape[1]) # [N, T]
    h_exp = h # [N, T] assuming h varies by cluster per voxel
    
    w_phi = (sigma_exp.pow(-1)) * (h_exp.pow(-1))
    # Avoid div by zero
    w_phi = torch.nan_to_num(w_phi, posinf=0, neginf=0)
    
    phi_num = torch.sum(w_phi * (residual ** 2), dim=0)
    phi_new = phi_num / residual.shape[0]
    
    # Normalize Phi so sum(phi) = T
    phi_new = phi_new / torch.sum(phi_new) * residual.shape[1]
    
    # 2. Update Sigma (Spatial Variance)
    # w_{ij} = 1 / (phi_j * h)
    phi_exp = phi_new.unsqueeze(0).repeat(h.shape[0], 1)
    w_sigma = (phi_exp.pow(-1)) * (h_exp.pow(-1))
    w_sigma = torch.nan_to_num(w_sigma, posinf=0, neginf=0)

    sigma_new = torch.sum(w_sigma * (residual ** 2), dim=1) / residual.shape[1]
    
    # 3. Update Clustering of Residuals (K) and H
    # k_{ij} = z_{ij} / sqrt(sigma_i * phi_j)
    sigma_col = sigma_new.unsqueeze(1)
    phi_row = phi_new.unsqueeze(0)
    denom = torch.sqrt(sigma_col * phi_row)
    k_ij = z_ij / (denom + 1e-8) # Avoid div 0

    # Sort k to find quantile bins
    flat_k = k_ij.flatten()
    sorted_k, _ = torch.sort(flat_k)
    num_elements = flat_k.numel()
    
    # Map elements to cluster labels (1 to n_clusters)
    # Using simple quantile binning
    bin_size = num_elements // n_clusters
    
    # We assign labels based on value ranking
    # To do this efficiently on GPU:
    ranks = torch.argsort(torch.argsort(flat_k)) # get rank of each element
    labels_flat = (ranks // bin_size).clamp(0, n_clusters - 1)
    labels_ij = labels_flat.view(k_ij.shape)
    
    # 4. Update H (Variance scale per cluster)
    h_new_vals = torch.zeros(n_clusters, device=device)
    
    # Weights for H update: 1 / (sigma * phi)
    w_h = (sigma_col.pow(-1)) * (phi_row.pow(-1))
    
    for c in range(n_clusters):
        mask = (labels_ij == c)
        if mask.sum() > 0:
            nom = torch.sum(w_h[mask] * (residual[mask] ** 2))
            den = mask.sum()
            h_new_vals[c] = nom / den
            
    # Normalize H
    # Global H normalization logic from paper/code
    # Assign H back to image
    h_img = torch.zeros_like(z_ij)
    for c in range(n_clusters):
        h_img[labels_ij == c] = h_new_vals[c]
        
    # Re-normalize h_img to maintain total variance scale if needed
    current_sum = torch.sum(h_img)
    target_sum = residual.numel() # or similar constant
    h_img = h_img / current_sum * target_sum

    return sigma_new, phi_new, h_img, labels_ij

def run_variance_estimation(z_path, res_path, out_dir, n_clusters, device_name):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Variance estimation on {device}")
    
    z_ij = torch.tensor(np.load(z_path), dtype=torch.float32).to(device)
    residual = torch.tensor(np.load(res_path), dtype=torch.float32).to(device)
    
    N, T = z_ij.shape
    
    # Initialization
    phi = torch.rand(T, device=device)
    phi = phi / torch.sum(phi) * T
    
    # Init sigma roughly from residuals
    phi_row = phi.unsqueeze(0).repeat(N, 1)
    sigma = torch.mean((residual**2)/phi_row, dim=1)
    
    h = torch.ones_like(z_ij, device=device)
    
    print("Iterating...")
    for i in range(20): # 20-50 iterations usually enough
        sigma, phi, h, labels = iteration_step(z_ij, residual, sigma, phi, h, n_clusters, device)
        
        if i % 5 == 0:
            print(f"Iter {i}: Phi mean={phi.mean():.4f}, Sigma mean={sigma.mean():.4f}")

    # Compute final scale factor (sig * phi)
    sigma_col = sigma.unsqueeze(1)
    phi_row = phi.unsqueeze(0)
    sig_phi_img = torch.sqrt(sigma_col * phi_row)

    check_dir(out_dir)
    # Save results
    torch.save(sigma, os.path.join(out_dir, 'sigma.pth'))
    torch.save(phi, os.path.join(out_dir, 'phi.pth'))
    torch.save(h, os.path.join(out_dir, 'h_img.pth'))
    torch.save(labels, os.path.join(out_dir, 'labels.pth'))
    torch.save(sig_phi_img, os.path.join(out_dir, 'sig_phi_img.pth'))
    print("Variance estimation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_est', type=str, required=True, help='z_ij_estimate.npy path')
    parser.add_argument('--res', type=str, required=True, help='residuals_raw.npy path')
    parser.add_argument('--out', type=str, default='./results/step3')
    parser.add_argument('--clusters', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    run_variance_estimation(args.z_est, args.res, args.out, args.clusters, args.device)
