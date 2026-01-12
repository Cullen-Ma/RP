import torch
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import multiprocessing as mpc
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import pandas as pd

class ElasticNetRegression:
    def __init__(self, alpha=100, l1_ratio=0.2):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.intercept_ = None
        self.losses = []

    def fit(self, X, y, weight, delta=50):
        delta = np.std(y)
        n_samples, n_features = X.shape
        self.coef_ = torch.ones(n_features).numpy()
        self.intercept_ = 0

        '''def fun(w):
            pred = np.dot(X, w)
            error = pred - y
            abs_error = np.abs(error)

            quadratic = np.minimum(abs_error, delta)
            linear = abs_error - quadratic
            huber_loss = np.sum((0.5 * quadratic ** 2 + delta * linear)) / n_samples

            # 正则化项
            l1_reg = self.alpha * self.l1_ratio * np.sum(np.abs(w))
            l2_reg = self.alpha * (1 - self.l1_ratio) * np.sum(np.square(w))

            #loss = huber_loss + l1_reg + l2_reg
            loss = huber_loss
            self.losses.append(loss.item())
            return loss'''
        def fun(w):
            pred = np.dot(X, w) + self.intercept_  # 别忘了加上截距项
            error = pred - y
            
            # 计算平方根误差(RMSE)
            rmse_loss = np.sqrt(np.mean(error ** 2))
            loss = rmse_loss
            self.losses.append(loss)  # 注意：这里不再调用`.item()`，因为不再是tensor
            return loss

        w = np.ones(n_features)
        options = {'maxiter': 1000}
        result = minimize(fun, w, method='BFGS', options=options)

        self.coef_ = result.x
        #print(f'lalala{result.x.shape}')
        #pdb.set_trace()
        return result.x

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def plot_losses(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Iterations')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
def divide_images(imgs, standard_len, n_gpus):

    vex, time = imgs.shape

 
    n_list = vex // standard_len

    n_reminder = vex - n_list * standard_len

    if n_reminder != 0:
        n_list += 1

    C_Ts_list_list = []
    for i in range(n_list):
        start = i * standard_len
        end = start + standard_len
        if i == n_list - 1:
            end = vex
        C_Ts_list = imgs[start:end, :]
        C_Ts_list_list.append(C_Ts_list)
    return C_Ts_list_list


def regression(standard_len, img, avg, weights, device, name):
    #print(f'avg={avg.shape}')
    print(f'img.shape[0]={img.shape[0]}')
    x = np.load('/share/home/kma/metabolism/data_simulation_code/data/t_midpoint.npy')
    alpha = torch.zeros((standard_len,30)).to(device)
    z_ij_estimate = torch.zeros((standard_len,66)).to(device)
    residual = torch.zeros((standard_len,66)).to(device)
    for j in tqdm(range(img.shape[0])):
        img_np = img[j,:].cpu().numpy()
        #print(img_np)
        
        avg_np = avg.cpu().numpy()
        #weights_np = weights.cpu().numpy()
        weights_np = np.ones(66)
        #print(f'weight{weights_np}')
        model = ElasticNetRegression(alpha=0.1, l1_ratio=0.5)
        model.fit(avg_np.T, img_np, weights_np)
        #model.plot_losses(f"wash_data/loss_{name}_{j}.png")
        params = model.coef_
        #print(f'params={params}')
        z_ij_np = np.dot(params.T, avg_np)
        
        '''plt.figure()
        plt.plot(x, img_np, color='green', label='img')
        plt.plot(x, z_ij_np, color='red', label='z_ij')
        
        plt.legend()
        plt.savefig(f"wash_data/plot_{name}_{j}.png")'''
        #print(f'z={z_ij_np}')
        
        r2 = r2_score(img_np, z_ij_np)
        print(f'R = {r2}')
        params = torch.from_numpy(params).to(device)
        alpha[j:] = params
        z_ij_estimate[j, :] = torch.matmul(params.T, avg)
        #print(f'z_ij={z_ij_estimate[j, :]}')
        residual[j, :] = img[j,:] - z_ij_estimate[j,:]
        #print(f'residual={residual[j, :]}')
        #print(f'img ={img[j,:]}')
    folder_result = 'simulation_data/result/parameters_estimate_cluster30kmeans/rmse_alpha0.10'
    if not os.path.exists(folder_result):  
      os.makedirs(folder_result)
    folder_result_1 = 'simulation_data/result/parameters_estimate_cluster30kmeans/rmse_z_ij_estimate0.10'
    if not os.path.exists(folder_result_1):  
      os.makedirs(folder_result_1)
    folder_result_2 = 'simulation_data/result/parameters_estimate_cluster30kmeans/rmse_residual0.10'
    if not os.path.exists(folder_result_2):  
      os.makedirs(folder_result_2)
    torch.save(alpha, 'simulation_data/result/parameters_estimate_cluster30kmeans/rmse_alpha0.10/alpha_all%s.pth' % (name))
    print('alpha_save')
    torch.save(z_ij_estimate,'simulation_data/result/parameters_estimate_cluster30kmeans/rmse_z_ij_estimate0.10/z_ij_estimate_all%s.pth' % (name))
    print('z_save')
    torch.save(residual, 'simulation_data/result/parameters_estimate_cluster30kmeans/rmse_residual0.10/residual_all%s.pth'% (name))
    print('residual')

def weight_func(img, time, duration):
    T = img.shape[1] 
    miu_avg = torch.mean(img, dim=0)
    miu_max = torch.max(img, dim=0)[0]
    miu = torch.where(miu_max * 0.1 > miu_avg, miu_avg, miu_max * 0.1)
    miu = miu / torch.max(miu)
    weight = torch.zeros(T)
    for i in range(len(miu)):
        weight[i] = torch.exp(-time[i] * 0.063) * duration[i] / miu[i]
        if weight[i] == float('inf') or weight[i] == 0:  
            weight[i] = 1
    return weight  

def main():
  
  #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
  mpc.set_start_method('spawn')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("begin")
  img = torch.load('simulation_data/C_t/noise/generalmid(i_C_0=0,noise_level=0.10).pth').to(device)
  img = img.T
  img = img[200:,:]
  avg = np.load('simulation_data/result/cluster30_for_bootstrap/avg_tac_kmeans_noise0.10.npy')
  avg = torch.from_numpy(avg).to(device)
  dura = torch.zeros(66, device=device)
  dura[:30] = 2/60
  dura[30:42] = 10/60
  dura[42:48] = 0.5
  dura[48:60] = 2
  dura[60:66] = 5
  
  time = np.load('/share/home/kma/metabolism/data_simulation_code/data/t_midpoint.npy')
  time = torch.from_numpy(time).to(device)
  weights = weight_func(img, time, dura)
  weights_min = torch.min(abs(weights))
  weights[:6] = weights_min
  weights = weights.to(device)
  z_ij_estimate = torch.zeros((img.shape[0], img.shape[1]), device=device)
  residual = torch.zeros((img.shape[0], img.shape[1]), device=device)
  cluster_label = 30
  alpha = torch.zeros((img.shape[0], cluster_label), device=device)


  
  gpus_list = [0,1,2,3,4,5,6]  
  n_gpus = len(gpus_list)
  standard_len= 100
  ####################################
  C_Ts_list_list = divide_images(img, standard_len, n_gpus)
  C_Ts_list_batch = []
  batch_index = 0
  for C_Ts_list in tqdm(C_Ts_list_list):
      C_Ts_list_batch.append(C_Ts_list)
      if len(C_Ts_list_batch) == n_gpus:
          pool = mpc.Pool(n_gpus)
          for i, gpu_id in enumerate(gpus_list): 
              pool.apply_async(func=regression,
                               args=(standard_len, C_Ts_list_batch[i], avg, weights,device,'%s_%s' % (str(batch_index), str(i))))
  
          pool.close()
          pool.join()
          batch_index += 1
          C_Ts_list_batch = []
if __name__ == '__main__':
  # freeze_support()
  main()
