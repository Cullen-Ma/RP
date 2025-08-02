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
import pdb
class ElasticNetRegression:
    def __init__(self, alpha=100, l1_ratio=0.2):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.intercept_ = None
        self.losses = []
    def fit(self, X, y, weight, delta=50):
        delta =  np.std(y)
        n_samples, n_features = X.shape
        self.coef_ = np.ones(n_features)  # 替换掉 torch 初始化
        self.intercept_ = 0

        def fun(w):
            pred = np.dot(X, w)
            error = pred - y
            abs_error = np.abs(error)

            # Huber Loss 计算
            quadratic = np.minimum(abs_error, delta)
            linear = abs_error - quadratic
            huber_loss = np.sum((0.5 * quadratic ** 2 + delta * linear)) / n_samples

            # 正则化项
            l1_reg = self.alpha * self.l1_ratio * np.sum(np.abs(w))
            l2_reg = self.alpha * (1 - self.l1_ratio) * np.sum(np.square(w))

            #loss = huber_loss + l1_reg + l2_reg
            loss = huber_loss
            self.losses.append(loss.item())
            return loss

        w = np.ones(n_features)
        options = {'maxiter': 10000000}
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
'''class ElasticNetRegression:
    def __init__(self, alpha=100, l1_ratio=0.2):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.intercept_ = None
        self.losses = []

    def fit(self, X, y, weight):
        n_samples, n_features = X.shape
        self.coef_ = torch.ones(n_features).numpy()
        self.intercept_ = 0

        def fun(w, beta):
            epsilon = 1e-10
            y_prob = y / (np.sum(y) + epsilon)
            Xw_prob = np.dot(X, w) / (np.sum(np.dot(X, w)) + epsilon)
            log = np.log(abs(y_prob / Xw_prob) + epsilon)
            log_modified = np.nan_to_num(log)
            KL_Divergence = np.sum(weight * y_prob * log_modified)
            Euclidean_Loss = np.sqrt(np.sum(weight * np.square((y - np.dot(X, w)))) / n_features)
            l1_reg = self.alpha * self.l1_ratio * np.sum(np.abs(w))
            l2_reg = self.alpha * (1 - self.l1_ratio) * np.sum(np.square(w))
            
            #kl_weight =  5 ** int(np.floor(np.log10(abs(Euclidean_Loss))))
            loss = l1_reg + l2_reg + KL_Divergence
            #loss = Euclidean_Loss + l1_reg + l2_reg + 1000*KL_Divergence
            #print(f'euclidean={Euclidean_Loss} and kl={kl_weight*KL_Divergence}')
            #pdb.set_trace()
            #loss = Euclidean_Loss + l1_reg + l2_reg
            #print(KL_Divergence)
            self.losses.append(loss.item())

            return loss

        w = np.ones(n_features)
        beta = 1
        options = {'maxiter':400}
        result = minimize(fun, w, args=(beta,), method='BFGS', options=options)

        self.coef_ = result.x
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
            plt.show()'''
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
    x = np.load('/share/home/kma/metabolism/data/tmidpoint.npy')
    #x = torch.load('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1/t_mid.pth')
    #x = x.numpy()
    alpha = torch.zeros((standard_len,12)).to(device)
    frame =66
    z_ij_estimate = torch.zeros((standard_len,frame)).to(device)
    residual = torch.zeros((standard_len,frame)).to(device)
    for j in tqdm(range(img.shape[0])):
        img_np = img[j,:].cpu().numpy()
        #print(f'img:{img_np}')
        
        avg_np = avg.cpu().numpy()
        #weights_np = weights.cpu().numpy()
        ####change!
        weights_np = np.ones(frame)
        #print(f'weight{weights_np}')
        model = ElasticNetRegression(alpha=0.1, l1_ratio=0.8)
        ###########################
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(avg_np.T)
        ###################################
        model.fit(X_scaled, img_np, weights_np)
        model.plot_losses(f"wash_data/img/loss_{name}_{j}.png")
        params = model.coef_
        #print(f'params={params}')
        z_ij_np = np.dot(params.T, avg_np)
        
        plt.figure()
        plt.plot(x, z_ij_np, color='red', label='z_ij')
        plt.plot(x, img_np, color='green', label='img')
        plt.legend()
        plt.savefig(f"wash_data/img/plot_10kl_{name}_{j}.png")
        pdb.set_trace()
        r2 = r2_score(img_np, z_ij_np)
        print(f'R = {r2}')
        params = torch.from_numpy(params).to(device)
        alpha[j:] = params
        z_ij_estimate[j, :] = torch.matmul(params.T, avg)
        #print(f'z_ij={z_ij_estimate[j, :]}')
        residual[j, :] = img[j,:] - z_ij_estimate[j,:]
        #print(f'residual={residual[j, :]}')
        #print(f'img ={img[j,:]}')
    folder_result = '/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/parameters_estimate_cluster12kmeans/alpha_62_wash'
    if not os.path.exists(folder_result):  
      os.makedirs(folder_result)
    folder_result_1 = '/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/parameters_estimate_cluster12kmeans/z_ij_estimate_62_wash'
    if not os.path.exists(folder_result_1):  
      os.makedirs(folder_result_1)
    folder_result_2 = '/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/parameters_estimate_cluster12kmeans/residual_62_wash'
    if not os.path.exists(folder_result_2):  
      os.makedirs(folder_result_2)
    torch.save(alpha, '/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/parameters_estimate_cluster12kmeans/alpha_62_wash/alpha_all%s.pth' % (name))
    print('alpha_save')
    torch.save(z_ij_estimate,'/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/parameters_estimate_cluster12kmeans/z_ij_estimate_62_wash/z_ij_estimate_all%s.pth' % (name))
    print('z_save')
    torch.save(residual, '/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/parameters_estimate_cluste12kmeans/residual_62_wash/residual_all%s.pth'% (name))
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
    img_tensor = torch.load('/share/home/kma/metabolism/data_guc_cut/UCD-GUC-2.0.0(ks_1=1).pth').to(device)
    img = img_tensor
    #img = img_tensor[:, 85:265, :]
    #img_np = np.load('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1_ks3/img_cut_slicer.npy')
    #img = torch.tensor(img_np).to(device)
    
    seg = np.load('/share/home/kma/metabolism/process_data/segment_slicer_cut.npy')
    #seg = np.load('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1/seg_pkc_1.npy')
    #seg = seg.astype(np.int64)  # 将 uint16 转换为 int64
    seg = torch.from_numpy(seg).to(device)
    z,y = torch.where(seg == 1)
    time_curves = img[:, z,y]
    img = time_curves.T
    #pdb.set_trace()
    #index  = np.load("wash_data/index_beta5.npy")
    #selected_rows = img[index]
    #img = selected_rows

    ##################################################################
    #img = np.load('new.npy')
    #img = torch.from_numpy(img).to(device)
    #####################################################################
    print(img.shape)
    avg = np.load('/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/slice62_cluster60/avg_tac_kmeans_12.npy')
    avg = torch.from_numpy(avg).to(device)
    dura = torch.zeros(66, device=device)
    dura[:30] = 2/60
    dura[30:42] = 10/60
    dura[42:48] = 0.5
    dura[48:60] = 2
    dura[60:66] = 5
    #pdb.set_trace()
    #dura = torch.load('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1/delta_frames.pth').to(device)
    #dura = dura/60
    #time = torch.load('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1/t_mid.pth').to(device)
    time = np.load('/share/home/kma/metabolism/data/tmidpoint.npy')
    #print(time)
    time = torch.from_numpy(time).to(device)
    weights = weight_func(img, time, dura)
    weights_min = torch.min(abs(weights))
    weights[:6] = weights_min
    weights = weights.to(device)
    z_ij_estimate = torch.zeros((img.shape[0], img.shape[1]), device=device)
    residual = torch.zeros((img.shape[0], img.shape[1]), device=device)
    cluster_label = 12
    alpha = torch.zeros((img.shape[0], cluster_label), device=device)



    #gpus_list = [0, 1,2,3,4,5,6,7,8,9]  
    gpus_list = [0]
    n_gpus = len(gpus_list)
    standard_len= 1500
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
