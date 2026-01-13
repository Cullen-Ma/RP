import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import multiprocessing as mpc
import warnings
import os
import math
import time
from tqdm import tqdm

# Environment settings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# =============================================================================
# Helper Functions: Data Generation & Physics Model
# =============================================================================

def t_data_generator(delta_frames):
    """Generates time midpoints for frames."""
    frame_split = np.cumsum(np.concatenate([np.zeros(shape=(1,)), delta_frames], axis=0))
    t_midpoint = np.zeros(shape=(frame_split.shape[0] - 1))
    for i in range(t_midpoint.shape[0]):
        t_midpoint[i] = (frame_split[i] + frame_split[i + 1]) / 2
    return frame_split / 60, t_midpoint / 60  # unit conversion to min

def C_0(t, interpolated_C0):
    """Input function sampler."""
    sampled_C0_values = interpolated_C0(t)
    return sampled_C0_values

def TAC_generator(interpolated_C0, t, params, step, t_sample_matrix, t_sample_matrix_r):
    """
    Generates the Time-Activity Curve (TAC) based on the 2-Tissue Compartment Model.
    
    Structure of params: [fv, K1, k2, k3, k4]
    """
    fv = params[0]
    K1, k2, k3, k4 = params[1:5]
    
    # Solve 2TCM differential equations analytically
    k2_k3_k4 = k2 + k3 + k4
    k2_k4 = k2 * k4
    
    # Calculate eigenvalues (alpha1, alpha2)
    # Using epsilon to avoid sqrt of negative numbers in extreme random walks
    discriminant = k2_k3_k4 ** 2 - 4 * k2_k4
    if discriminant < 0:
        discriminant = 0
        
    a1 = 0.5 * (k2_k3_k4 - (discriminant) ** 0.5)
    a2 = 0.5 * (k2_k3_k4 + (discriminant) ** 0.5)
    
    # Convolve Input Function with Impulse Response
    C_0_vec = interpolated_C0(t)
    
    # Numerical Convolution (Integration)
    # Note: Vectorized for performance
    C_0_matrix_1 = np.repeat(C_0_vec.reshape(1, -1), t.shape[0], axis=0)
    # Trapezoidal rule approximation factor
    integration_factor = t_sample_matrix * (np.ones(t_sample_matrix.shape[0]) - 0.5 * np.eye(t_sample_matrix.shape[0]))
    C_0_matrix = C_0_matrix_1 * integration_factor
    
    response_function_1_r = np.exp(-a1 * t_sample_matrix_r)
    response_function_2_r = np.exp(-a2 * t_sample_matrix_r)
    
    # Convolution operations
    _ = C_0_matrix * response_function_1_r
    conv_1 = (np.sum(_, axis=1) * step)

    _ = C_0_matrix * response_function_2_r
    conv_2 = (np.sum(_, axis=1) * step)

    C_0_sample = C_0(t, interpolated_C0)
    
    # Calculate C_tissue components
    # Avoid division by zero
    denominator = a2 - a1
    if abs(denominator) < 1e-9:
        denominator = 1e-9
        
    C_1_sample = K1 / denominator * ((k4 - a1) * conv_1 + (a2 - k4) * conv_2)
    C_2_sample = K1 * k3 / denominator * (conv_1 - conv_2)
    
    # Total Concentration: Blood fraction + Tissue
    C_T_sample = fv * C_0_sample + (1 - fv) * (C_1_sample + C_2_sample)

    return C_0_sample, C_T_sample

# =============================================================================
# Statistical Functions: Likelihood & Prior
# =============================================================================

def likelihood(interpolated_C0, zij, params, t, step, t_sample_matrix, t_sample_matrix_r, dura_seconds, estimated_sigma):
    """
    Calculates the Log-Likelihood assuming Gaussian noise.
    
    Args:
        zij: Measured PET data (frames).
        estimated_sigma: The standard deviation of noise, estimated from residuals.
    """
    # Generate high-resolution TAC (1 second resolution usually)
    _, C_T_high_res = TAC_generator(interpolated_C0, t, params, step, t_sample_matrix, t_sample_matrix_r)
    
    # Downsample high-res TAC to match PET frames
    # Optimization: Use reduceat instead of loop for speed
    # dura_seconds example: [120, 120, ..., 300]
    
    # Calculate indices for slicing. 
    # Provided t_solution is 1 sec, so indices map directly to accumulated seconds.
    cumsum_dura = np.cumsum(np.concatenate(([0], dura_seconds))).astype(int)
    # Ensure we don't exceed the generated curve length
    cumsum_dura = np.clip(cumsum_dura, 0, len(C_T_high_res))
    
    # Sum within frames
    # Note: cumsum_dura[:-1] defines starts, cumsum_dura[1:] defines ends
    # We use np.add.reduceat. However, reduceat is tricky with 0. 
    # Alternative reliable loop for variable frame length (vectorized is hard if frames vary wildly per patient, but here they are fixed):
    
    frame_means = []
    for i in range(len(dura_seconds)):
        start = cumsum_dura[i]
        end = cumsum_dura[i+1]
        if start == end:
            frame_means.append(0)
        else:
            frame_means.append(np.mean(C_T_high_res[start:end]))
    
    C_T_frames = np.array(frame_means)
    
    # Calculate Residuals
    rij = C_T_frames - zij
    
    # Log-Likelihood for Gaussian Distribution
    # Log(L) = - (N/2)*ln(2*pi*sigma^2) - (1/(2*sigma^2)) * sum(residuals^2)
    # We focus on the proportional part for MCMC
    
    # INDICATOR: "estimated_sigma" comes from the initial fit residuals
    sigma = estimated_sigma 
    
    # Avoid numerical overflow
    if sigma < 1e-6: sigma = 1e-6
        
    term1 = -len(zij) * math.log(sigma) # Simplified log term
    term2 = -np.sum(rij**2) / (2 * sigma**2)
    
    log_prob = term1 + term2
    return log_prob

def prior(initial_theta, theta, prior_stds):
    """
    Calculates the Prior probability based on Population Statistics.
    
    Args:
        initial_theta: The NLLS result for this specific voxel (Mean of Prior).
        theta: The current proposed parameters.
        prior_stds: The standard deviations derived from the whole population (Std of Prior).
    """
    # params structure: [fv, K1, k2, k3, k4]
    
    # INDICATOR: Using population statistics for prior width as per Paper
    prob = 1.0
    for i in range(5):
        # Calculate PDF for Gaussian: N(mean=initial, std=population_std)
        p = norm.pdf(theta[i], loc=initial_theta[i], scale=prior_stds[i])
        prob *= p
        
    return prob

# =============================================================================
# MCMC Core
# =============================================================================

def proposal_distribution(current_theta):
    """Random Walk Proposal with k4 constraint."""
    # INDICATOR: Step size (sigma for proposal). Can be tuned.
    # 0.01 is a reasonable starting point for normalized parameters, 
    # but might need tuning for k2/k3 if they are large.
    step_sigma = 0.01 
    
    new_theta = np.random.normal(current_theta, step_sigma)
    
    # INDICATOR: Strictly enforcing 2TCM Irreversible Model -> k4 = 0
    new_theta[4] = 0 
    
    # Constraint: Parameters must be non-negative (Physiological constraint)
    # If negative, reflect or clamp? Rejection is better for detailed balance, 
    # but clamping is common in simple implementations. Here we take absolute.
    new_theta = np.abs(new_theta)
    new_theta[4] = 0 # Ensure 0 again after abs
    
    return new_theta

def metropolis_hastings(batch_id, interpolated_C0, n_iterations, burn_in, 
                        data_batch, initial_theta_batch, 
                        step, t_sample, t_sample_matrix, t_sample_matrix_r,
                        dura_seconds, population_prior_stds, estimated_sigmas_batch):
    
    n_samples = n_iterations - burn_in
    length = len(data_batch)
    
    # To store results
    all_samples = np.zeros((length, n_samples, 5))
    
    for idx in range(length):
        data = data_batch[idx]
        current_theta = initial_theta_batch[idx].copy()
        current_theta[4] = 0 # Ensure start with k4=0
        
        # Get the pre-estimated noise sigma for this specific voxel
        sigma_lik = estimated_sigmas_batch[idx]
        
        # Calculate initial Log-Posterior
        current_log_lik = likelihood(interpolated_C0, data, current_theta, t_sample, step, 
                                     t_sample_matrix, t_sample_matrix_r, dura_seconds, sigma_lik)
        # We work with Log-Prior to avoid underflow
        # prior returns probability density, log it.
        # Add small epsilon to avoid log(0)
        current_prior_val = prior(initial_theta_batch[idx], current_theta, population_prior_stds)
        current_log_prior = np.log(current_prior_val + 1e-300) 
        
        current_log_post = current_log_lik + current_log_prior
        
        accepted_count = 0
        
        # MCMC Loop
        # INDICATOR: n_iterations set to 10,000 in main
        for i in range(n_iterations):
            proposed_theta = proposal_distribution(current_theta)
            
            # Calculate Proposed Log-Posterior
            prop_log_lik = likelihood(interpolated_C0, data, proposed_theta, t_sample, step, 
                                      t_sample_matrix, t_sample_matrix_r, dura_seconds, sigma_lik)
            prop_prior_val = prior(initial_theta_batch[idx], proposed_theta, population_prior_stds)
            prop_log_prior = np.log(prop_prior_val + 1e-300)
            
            prop_log_post = prop_log_lik + prop_log_prior
            
            # Metropolis Acceptance Criterion (Log domain)
            # alpha = min(1, P_new / P_old) -> log_alpha = min(0, log_new - log_old)
            log_alpha = prop_log_post - current_log_post
            
            # Accept check
            if np.log(np.random.rand()) < log_alpha:
                current_theta = proposed_theta
                current_log_post = prop_log_post
                if i >= burn_in:
                    accepted_count += 1
            
            # Store after burn-in
            if i >= burn_in:
                all_samples[idx, i - burn_in, :] = current_theta

    # Save batch results
    output_dir = 'samples_strict_paper'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/sample_batch_{batch_id}.npy', all_samples)
    return

# =============================================================================
# Main Execution
# =============================================================================

def generate_t_sample_matrix(end, t_solution):
    step = t_solution / 60
    n_samples = int(end / (t_solution / 60)) + 1
    t_sample = np.linspace(start=0, stop=end, num=n_samples)
    t_sample_matrix = np.zeros(shape=(n_samples, n_samples))
    t_sample_matrix_r = np.zeros(shape=(n_samples, n_samples))
    for i in range(n_samples):
        t_sample_matrix[i, 0: i + 1] = t_sample[0: i + 1]
        t_sample_matrix_r[i, 0: i + 1] = np.flip(t_sample[0: i + 1], axis=[0])
    return step, t_sample, t_sample_matrix, t_sample_matrix_r

if __name__=='__main__':
    mpc.set_start_method('spawn')
    
    print('Loading Data...', time.strftime('%Y-%m-%d %H:%M:%S'))
    
    # 1. Load Data
    # -------------------------------------------------------------------------
    # Using the paths from your provided code
    imgs = np.load('/share/home/kma/metropolis_hastings/data2/img_slicer_cut.npy')
    segment = np.load('/share/home/kma/metropolis_hastings/data2/segment_slicer_cut.npy')
    z, y = np.where(segment == 1)
    data_group = imgs[:, z, y] # Shape: (Time, Voxels) -> Transpose to (Voxels, Time) usually better
    data_group = data_group.T  # Shape: (N_voxels, N_frames)
    
    # Initial parameters from NLLS (Shape: N_voxels, 5) [fv, K1, k2, k3, k4]
    initial_theta_group_full = np.load('/share/home/kma/metropolis_hastings/data2/parameter.npy')
    initial_theta_group = initial_theta_group_full[z, y, :5]
    
    # C0 Input Function
    discrete_C0_values = np.load('/share/home/kma/metropolis_hastings/data2/c0.npy')
    
    # 2. Time & Frame Setup
    # -------------------------------------------------------------------------
    t_total_min = 60
    t_solution_sec = 1
    
    # Frame duration in seconds (Matches your provided 'dura')
    dura_seconds = np.concatenate([np.ones(shape=(30,)) * 2,   # 1 min (30x2s)
                                   np.ones(shape=(12,)) * 10,  # 2 min (12x10s) -> total 3min
                                   np.ones(shape=(6,)) * 30,   # 3 min (6x30s) -> total 6min
                                   np.ones(shape=(12,)) * 120, # 24 min -> total 30min
                                   np.ones(shape=(6,)) * 300], # 30 min -> total 60min
                                  axis=0)
    
    # Generate high-res time matrix for convolution
    step, t_sample, t_sample_matrix, t_sample_matrix_r = generate_t_sample_matrix(end=t_total_min, t_solution=t_solution_sec)
    
    # Interpolate C0
    # Assuming t_vec for C0 corresponds to frame midpoints roughly or raw data timing
    # Re-using your logic for t_vec
    t_vec_frames = np.cumsum(dura_seconds) / 60 # ends in min
    t_vec_mid = (np.concatenate(([0], t_vec_frames[:-1])) + t_vec_frames) / 2
    
    # Ensure C0 interpolation covers 0 to 60
    extended_t_vec = np.hstack([[0], t_vec_mid, [60]])
    if extended_t_vec[1] > 0:
        extended_C0_values = np.hstack([[discrete_C0_values[0]], discrete_C0_values, [discrete_C0_values[-1]]])
    else:
        extended_C0_values = discrete_C0_values
    
    interpolated_C0 = interp1d(extended_t_vec, extended_C0_values, kind='cubic', fill_value="extrapolate")

    # 3. Scientific Configuration (Strict adherence to Paper)
    # -------------------------------------------------------------------------
    print('Configuring Statistical Model...')

    # A. Prior Standard Deviation (Population Statistics)
    # INDICATOR: Instead of 0.01, we calculate the Std Dev of the NLLS results across all voxels.

    population_prior_stds = np.std(initial_theta_group, axis=0) 
    # Safety: ensure no zero std if all initial values are identical
    population_prior_stds[population_prior_stds == 0] = 0.01 
    print(f"Calculated Population Prior Stds: {population_prior_stds}")

    # B. Likelihood Sigma Estimation (Residual Based)
    # INDICATOR: We estimate 'sigma' for each voxel based on the RSS of the NLLS fit.
    # This replaces the hardcoded '100'.
    print('Estimating Noise Sigma from NLLS residuals...')
    estimated_sigmas = []
    
    # We need to simulate the curve for every voxel once to get the residual
    # Since this is slow in Python, we do a quick check or use multiprocessing if dataset is huge.
    # Here is a sequential implementation for clarity (can be parallelized if needed).
    
    # Define a quick function for initial residual calc
    def get_residual_sigma(idx):
        params = initial_theta_group[idx]
        params[4] = 0 # Enforce k4=0 for the model check
        zij = data_group[idx]
        
        # Generate curve
        _, C_T_est = TAC_generator(interpolated_C0, t_sample, params, step, t_sample_matrix, t_sample_matrix_r)
        
        # Downsample (Simplified logic matching likelihood)
        cumsum_dura = np.cumsum(np.concatenate(([0], dura_seconds))).astype(int)
        cumsum_dura = np.clip(cumsum_dura, 0, len(C_T_est))
        frame_means = []
        for i in range(len(dura_seconds)):
            start, end = cumsum_dura[i], cumsum_dura[i+1]
            val = np.mean(C_T_est[start:end]) if start != end else 0
            frame_means.append(val)
        
        residuals = np.array(frame_means) - zij
        rss = np.sum(residuals**2)
        # Sigma estimation: sqrt(RSS / (N - p)) or just RMSE. 
        # Paper implies Gaussian error variance. RMSE is a robust estimator.
        sigma_est = np.sqrt(rss / len(zij))
        return sigma_est

    # Use a small pool to calculate initial Sigmas to save time
    # Note: If this takes too long, you can implement a simpler heuristic, 
    # e.g., sigma = 0.05 * max_uptake (INDICATOR).
    # But strictly speaking, residual based is better.
    with mpc.Pool(10) as pool:
        estimated_sigmas = list(tqdm(pool.imap(get_residual_sigma, range(len(data_group))), 
                                     total=len(data_group), desc="Calculating Initial Sigmas"))
    estimated_sigmas = np.array(estimated_sigmas)
    
    # Handle cases where sigma might be 0 (perfect fit)
    estimated_sigmas[estimated_sigmas < 1e-6] = 1.0

    # C. MCMC Parameters
    # INDICATOR: Strict Paper settings
    n_iterations = 10000 
    burn_in = 800

    # 4. Parallel Processing Execution
    # -------------------------------------------------------------------------
    print(f'Starting MCMC Sampling with {n_iterations} iterations...')
    
    n_cpus = 10 # Adjust based on your machine
    total_voxels = len(data_group)
    chunk_size = int(np.ceil(total_voxels / n_cpus))
    
    # Create batches explicitly to ensure no data loss
    batches = []
    for i in range(0, total_voxels, chunk_size):
        end_idx = min(i + chunk_size, total_voxels)
        batch_dict = {
            'id': i // chunk_size,
            'data': data_group[i:end_idx],
            'theta': initial_theta_group[i:end_idx],
            'sigmas': estimated_sigmas[i:end_idx]
        }
        batches.append(batch_dict)

    pool = mpc.Pool(n_cpus)
    results = []

    for batch in batches:
        # Pass all necessary static args + batch specific args
        pool.apply_async(func=metropolis_hastings,
                         args=(batch['id'], interpolated_C0, n_iterations, burn_in, 
                               batch['data'], batch['theta'], 
                               step, t_sample, t_sample_matrix, t_sample_matrix_r,
                               dura_seconds, population_prior_stds, batch['sigmas']))

    pool.close()
    pool.join()
    
    print('Processing Complete.', time.strftime('%Y-%m-%d %H:%M:%S'))
