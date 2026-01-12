import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics import r2_score
import torch.distributions.gamma as gamma_dist
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pdb
def r2_change(img, z, kind):
    sum = 0
    count = 0
    for i in range(kind):
        r2 = r2_score(img[i, :], z[i, :])
        if r2<0:
          count=count+1
          #print(r2)
        sum = sum +r2
    print(count)
    return sum/kind
def t_data_generator(delta_frames):
    frame_split = np.cumsum(np.concatenate([np.zeros(shape=(1,)), delta_frames], axis=0))
    t_midpoint = np.zeros(shape=(frame_split.shape[0] - 1))
    for i in range(t_midpoint.shape[0]):
        t_midpoint[i] = (frame_split[i] + frame_split[i + 1]) / 2
    return frame_split / 60, t_midpoint / 60  # unit conversion to min
def generate_t_sample_matrix(end, t_solution):

    # t_solution must be divisible by 1
    # unit of t_solution: second
    # unit of end: min

    step = t_solution / 60
    n_samples = int(end / (t_solution / 60)) + 1
    t_sample = np.linspace(start=0, stop=end, num=n_samples)
    t_sample_matrix = np.zeros(shape=(n_samples, n_samples))
    t_sample_matrix_r = np.zeros(shape=(n_samples, n_samples))
    for i in range(n_samples):
        t_sample_matrix[i, 0: i + 1] = t_sample[0: i + 1]
        t_sample_matrix_r[i, 0: i + 1] = np.flip(t_sample[0: i + 1], axis=[0])
    return step, t_sample, t_sample_matrix, t_sample_matrix_r
label_ij = np.load("/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/slice62_cluster60/label_cluster_tac_kmeans_12.npy")
#print(label.shape)
#print(np.max(label_ij))
#print(np.min(label_ij))
z_ij_estimate = np.zeros((15000,66))
for gpu in range(10):
  z2 = torch.load('/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/parameters_estimate_cluster12kmeans/z_ij_estimate_62_wash/z_ij_estimate_all0_%d.pth' % ( gpu), map_location=torch.device('cpu')).cpu().numpy()
  z_ij_estimate[gpu*1500:(gpu+1)*1500,:] = z2
z_ij_estimate = z_ij_estimate[:14813,:]
#img_tensor = torch.load('/share/home/kma/metabolism/data_guc_cut/UCD-GUC-2.0.0(ks_1=3).pth')
#img = img_tensor[:, 85:265, :]

img_tensor = torch.load('/share/home/kma/metabolism/data_guc_cut/UCD-GUC-2.0.0(ks_1=1).pth')
img = img_tensor.numpy()
seg = np.load('/share/home/kma/metabolism/process_data/segment_slicer_cut.npy')
#seg = seg.astype(np.int64)  # 将 uint16 转换为 int64
z,y = np.where(seg == 1)
#import pdb
#pdb.set_trace()
#############################
'''x = np.linspace(1, 66, 66)
zij_2d = np.zeros((66,180,150))
zij_2d[:,z,y] = z_ij_estimate.T
plt.figure()
plt.plot(x, zij_2d[:,91,87], color='red', label='z_ij')
plt.plot(x, img[:,91,87], color='green', label='img')
plt.legend()
plt.savefig(f"plot11.png")'''
############################
##########################
#和生成15个样本无关
delta_frames = np.concatenate([np.ones(shape=(30,)) * 2,  # 1 min
                                   np.ones(shape=(12,)) * 10,  # 3 min
                                   np.ones(shape=(6,)) * 30,  # 6 min
                                   np.ones(shape=(12,)) * 115,  # 30 min
                                   np.ones(shape=(6,)) * 300], axis=0)
print(f'delta_frame{delta_frames}')
#dura = torch.load('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1/delta_frames.pth')
#print(f'dura{dura}')
#pdb.set_trace()
#dura = dura/60
#delta_frames = dura.numpy()
frame_split, t_vec = t_data_generator(delta_frames)
step, t_sample, t_sample_matrix, t_sample_matrix_r = generate_t_sample_matrix(end=60, t_solution=1)
extended_t_vec = np.hstack([[0], t_vec, [60]])
_ = img

#######################
time_curves = img[:, z,y]
img = time_curves.T
residual = img - z_ij_estimate
cluster_kind = 12
lung_tac_mean_list = []
#直接交换残差
for j in range(20):
    for i in range(cluster_kind):
        # 找到label中等于1的位置的索引
        index = np.where(label_ij == i)[0]
        index = list(set(index))
        # 提取res中对应位置的数值
        values_shuffle = residual[index,:]
        # 按行打乱数组
        permutation = np.random.permutation(values_shuffle.shape[0])
        values_shuffle = values_shuffle[permutation]
        # 将打乱后的值放回res中的对应位置
        residual[index] = values_shuffle
    #########################
    #draw curves of tac to show uncertainty
    
    ##############################
    img_simulation = z_ij_estimate + residual

    #注意这里生成的是二维数组
    r2 = r2_change(img, img_simulation,600)
    print(f'average={r2}')

    np.save('/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/sample_real/img_simulation_rp20_%d.npy'%j,img_simulation)
##############
#draw curves of tac to show uncertainty

