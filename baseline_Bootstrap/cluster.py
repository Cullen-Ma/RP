import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
import os
from utils import check_dir

def run_clustering(data_path, mask_path, output_dir, n_clusters):
    print(f"Loading data from {data_path}...")
    # Assume input is (Time, Z, Y, X) or (Time, Voxels)
    img = np.load(data_path) 
    
    # Flatten data to (Voxels, Time) based on mask if provided, else just flatten
    if mask_path:
        mask = np.load(mask_path)
        # Assuming img is [Time, Z, Y, X] and mask is [Z, Y, X]
        time_points = img.shape[0]
        masked_img = img[:, mask == 1] # Result: [Time, N_Voxels]
        time_curves = masked_img.T     # Result: [N_Voxels, Time]
    else:
        # Assuming input is already [N_Voxels, Time] or reshaped
        if img.ndim > 2:
            time_curves = img.reshape(img.shape[0], -1).T
        else:
            time_curves = img
            
    # Handle NaNs
    df = pd.DataFrame(time_curves)
    df = df.fillna(method='ffill').fillna(method='bfill')
    time_curves = df.values

    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, max_iter=600, random_state=0)
    kmeans.fit(time_curves)
    cluster_labels = kmeans.labels_

    # Calculate Average TACs (Basis Functions X)
    frames = time_curves.shape[1]
    avg_tac = np.zeros((n_clusters, frames))
    
    for i in range(n_clusters):
        positions = np.where(cluster_labels == i)[0]
        if len(positions) > 0:
            avg_tac[i, :] = np.mean(time_curves[positions, :], axis=0)
            
    check_dir(output_dir)
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), cluster_labels)
    np.save(os.path.join(output_dir, 'avg_tac.npy'), avg_tac)
    print("Clustering complete. Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to 4D PET npy file')
    parser.add_argument('--mask', type=str, default=None, help='Path to binary mask npy file')
    parser.add_argument('--out', type=str, default='./results/step1', help='Output directory')
    parser.add_argument('--clusters', type=int, default=12, help='Number of clusters')
    args = parser.parse_args()
    
    run_clustering(args.data, args.mask, args.out, args.clusters)
