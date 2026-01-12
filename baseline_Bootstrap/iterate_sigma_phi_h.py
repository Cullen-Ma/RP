import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
def r2_change(img, z, kind):
    sum = 0
    count = 0
    for i in range(kind):
        r2 = r2_score(img[i, :], z[i, :])
        if r2<0:
          count=count+1
          #print(r2)
        sum = sum +r2
    #print(count)
    return sum/kind

def iteration(z_ij_estimate, sigma, phi, h, cluster_label):
    # 统一转换为 float32
    z_ij_estimate = z_ij_estimate.float()
    sigma = sigma.float()
    phi = phi.float()
    h = h.float()

    h = torch.where(h == 0, torch.tensor(0.01, device=h.device), h)
    sigma = torch.where(sigma == 0, torch.tensor(0.001, device=sigma.device), sigma)
    h_new = torch.ones(cluster_label)
    #######nbnbnb
    sigma = sigma.unsqueeze(0).repeat(h.shape[1], 1)
    sigma = sigma.T
    
    # 计算 w_ij_sigma
    w_ij_sigma = (sigma.type(torch.float)) ** -1 * (h.type(torch.float)) ** -1
    #print(np.isnan(w_ij_sigma))
    phi_new_row = torch.sum(w_ij_sigma * (residual ** 2), dim=0)/residual.shape[0]
    #print(np.isnan(phi_new_row))
    phi_new1 = phi_new_row/torch.sum(phi_new_row)*residual.shape[1]
    #print(torch.sum(phi_new))
    #print(phi_new)
    #phi_new = torch.where(phi_new==0, 0.001, phi_new)
    phi_new = phi_new1.unsqueeze(0).repeat(h.shape[0], 1)
    #############################
    ##############修改
    #print(torch.sum(phi))
    phi = phi.unsqueeze(0).repeat(h.shape[0], 1)
    w_ij_phi = (phi.type(torch.float)) ** -1 * (h.type(torch.float)) ** -1

    sigma_new_line = torch.sum(w_ij_phi * (residual ** 2), dim=1) / residual.shape[1]
    #sigma_new_line = sigma_new_line/torch.sum(sigma_new_line)*residual.shape[0]
    sigma_new = sigma_new_line.unsqueeze(1).repeat(1, h.shape[1])
    #sigma_new = torch.where(sigma_new==0, 0.001, sigma_new)
    ################################


    # 生成K,将k_ij排序，进行分位数映射，映射到l个上，l是聚类的类别
    # 注意，文章中说cluster的类别为50-100，但是我是6，所以这个要再完善
    #for i in range(residual.shape[0]):
        #for j in range(residual.shape[1]):
            #k_ij[i][j] = z_ij_estimate[i][j] / sigma_new[i] / phi_new[j]
    # 利用numpy的广播机制计算每个元素的值,降低复杂度
    k_ij = z_ij_estimate / torch.sqrt(sigma_new * phi_new)

    # 排序k
    array_1d = k_ij.flatten()  # 将二维数组展平为一维数组
    array_sorted = torch.sort(array_1d)[0]  # 对一维数组进行排序
    h_label_ij = torch.zeros_like(k_ij)  # 定义标记数组，用于存储分类结果
    # 对元素进行分类，并将分类结果标记回二维数组
    num_elements = array_sorted.numel()
    cluster_step = num_elements // cluster_label +1 # 注意使用整数除法
    labels = (torch.arange(num_elements) // cluster_step) + 1
    # 创建一个映射，用于快速查找每个排序元素的标签
    element_to_label_map = {element.item(): label for element, label in zip(array_sorted, labels)}
    # 更新 k_ij 中的标签
    h_label_1 = torch.tensor([element_to_label_map[element.item()] for element in array_1d])
    h_label_ij = h_label_1.reshape(k_ij.shape)
    torch.save(h_label_ij, '/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/h_label_ij_cluster7.pth')
    '''
    for i, element in enumerate(array_sorted):
        # 计算元素应该属于的类别
        # !!!!!!!!!!!!!!!!!!!70是cluster的种类
        # 获取张量的元素数量
        num_elements = array_sorted.numel()
        label = int(i / (num_elements /cluster_label)) + 1
        # 将分类结果标记回二维数组的对应位置上
        h_label_ij[k_ij == element] = label
    '''

    #label_ij = torch.zeros_like(k_ij)
    label_ij = h_label_ij
    for item in range(cluster_label):
        mask = (h_label_ij == item + 1)
        sigma_masked = sigma[mask]
        phi_masked = phi[mask]
        residual_masked = residual[mask]

        # Calculate the weights
        #weights = torch.where((sigma_masked != 0) & (phi_masked != 0),sigma_masked.pow(-1) * phi_masked.pow(-1), 1)
        weights = sigma_masked.pow(-1) * phi_masked.pow(-1)
        # Compute the weighted sum of residuals squared
        h_sum = (weights * residual_masked.pow(2)).sum()
        # Calculate the average
        h_new[item] = h_sum / mask.sum().float()
    
    '''
    # 得到h_l
    for item in range(cluster_label):
        h_sum = 0
        indices = (h_label_ij == item + 1).nonzero()  # 获取满足条件的索引
        if indices.size(0) > 0:
            for i in range(indices.size(0)):
                i_idx = indices[i, 0]
                j_idx = indices[i, 1]
                if sigma_new[i_idx, j_idx] != 0 and phi_new[i_idx, j_idx] != 0:
                    w_ij_sigma_phi = float(sigma_new[i_idx, j_idx]) ** -2 * float(phi_new[i_idx, j_idx]) ** -2
                else:
                    w_ij_sigma_phi = 1

                h_sum = h_sum + w_ij_sigma_phi * residual[i_idx, j_idx] * residual[i_idx, j_idx]
            h_new[item] = h_sum / indices.size(0)
    '''

    #70是聚类的类别
    h_new_ij = h_label_ij.float()
    h_new = h_new*cluster_label/ torch.sum(h_new)
    #print(f'h={h_new}')
    #print(torch.sum(h_new))
    unique_labels = torch.unique(h_label_ij)
    for item in unique_labels:
        indices2 = (h_label_ij == item).nonzero(as_tuple=False)
        if indices2.numel() > 0:  
            h_new_ij[indices2[:, 0], indices2[:, 1]] = h_new[item-1]
    h_new_ij_1 = h_new_ij / torch.sum(h_new_ij) *h.shape[1] * h.shape[0]
    return sigma_new_line, phi_new1, h_new_ij_1, label_ij, k_ij


z_ij_estimate = torch.zeros((15000,66))
num = 14813
for gpu in range(10):
  z2 = torch.load('/share/home/kma/metabolism/data_guc_cut/UCD_GUC_2.0.0(ks_1=1)_result/parameters_estimate_cluster12kmeans/z_ij_estimate_62_wash/z_ij_estimate_all0_%d.pth' % ( gpu), map_location=torch.device('cpu'))
  z_ij_estimate[gpu*1500:(gpu+1)*1500,:] = z2
zij = z_ij_estimate[:14813,:]

img = torch.load('/share/home/kma/metabolism/data_guc_cut/UCD-GUC-2.0.0(ks_1=1).pth')
seg = np.load('/share/home/kma/metabolism/process_data/segment_slicer_cut.npy')
seg = seg.astype(np.int64)  # 将 uint16 转换为 int64
seg = torch.from_numpy(seg)
z,y = torch.where(seg == 1)
time_curves = img[:, z,y]
img_62 = time_curves.T
zij = zij.float()
img_62 = img_62.float()
residual = img_62 - zij

print(residual)
#residual = abs(residual)

cluster_label = 12
# 原来可能用了 .double()
init_phi_nor = torch.rand(66)  # 默认就是 float32
print(torch.max(init_phi_nor))
init_phi = init_phi_nor

init_phi2 = init_phi * init_phi
init_phi2 = init_phi2 / torch.sum(init_phi2) * residual.shape[1]

init_phi_ij = init_phi2.unsqueeze(0).repeat(residual.shape[0], 1)

# init_sigma 初始化也保持 float32
init_sigma = residual / torch.sqrt(init_phi_ij)
init_sigma = torch.mean(init_sigma, dim=1)
init_sigma2 = init_sigma * init_sigma

# init_h2 初始化也保持 float32
init_h2 = torch.abs(torch.randn_like(zij))  # 默认 float32
init_h2 = init_h2 / torch.sum(init_h2) * residual.shape[0] * residual.shape[1]
'''init_h2 = init_h2.double()
init_phi2 = init_phi2.double()
init_sigma2 = init_sigma2.double()'''
# 迭代并收敛参数
for i in range(100):  # 最大迭代次数设为100，可以根据实际情况调整
    print("parameter_begin")
    print(i)
    init_sigma_new, init_phi_new, init_h_new, label_ij, k_ij = iteration(zij, init_sigma2, init_phi2, init_h2, cluster_label)

    # 计算每个变量的改变量
    delta_sigma = torch.norm(init_sigma_new.float() - init_sigma) / torch.norm(init_sigma)
    delta_phi = torch.norm(init_phi_new.float() - init_phi) / torch.norm(init_phi)
    delta_h = torch.norm(init_h_new.float() - init_h2) / torch.norm(init_h2.float())
    threshold = 0.01
    sigma1 = init_sigma_new.unsqueeze(0).repeat(init_h2.shape[1], 1)
    sigma1 = sigma1.T
    phi1 = init_phi_new.unsqueeze(0).repeat(init_h2.shape[0], 1)
    sigphiij = torch.sqrt(sigma1 * phi1)
    z_updata1 = zij + sigphiij
    # 检查是否所有改变量都低于阈值，如果是，则认为已经收敛
    print(f'delta_sigma={delta_sigma},delta_phi={delta_phi},delta_h={delta_h}')
    #if (delta_sigma < threshold and delta_phi < threshold and delta_h < threshold):
        

        #break
    r2 = r2_change(img_62, z_updata1,num)
    print(f'r2={r2}')
    if r2>=0:
        break
    init_sigma2 = init_sigma_new
    init_phi2 = init_phi_new
    init_h2 = init_h_new

z_updata1 = torch.zeros_like(zij)
sigphiij_zf = torch.zeros_like(zij)
for i in range(sigphiij.shape[0]):
    for j in range(sigphiij.shape[1]):
        if abs((zij[i,j] + sigphiij[i,j]) - img_62[i,j]) < abs((zij[i,j] - sigphiij[i,j]) - img_62[i,j]):
            z_updata1[i,j] = zij[i,j] + sigphiij[i,j]
            sigphiij_zf[i,j] = sigphiij[i,j]
        else:
            z_updata1[i,j] = zij[i,j] - sigphiij[i,j]
            sigphiij_zf[i,j] = -sigphiij[i,j]
sigma = init_sigma2.unsqueeze(0).repeat(init_h2.shape[1], 1)
sigma = sigma.T
phi = init_phi2.unsqueeze(0).repeat(init_h2.shape[0], 1) 
for cluster in range(12):
    x = np.load('/share/home/kma/metabolism/data/t_midpoint.npy')
    plt.figure()
    plt.plot(x, img_62[cluster*1000+3], color='green', label='img')
    plt.plot(x, zij[cluster*1000+3], color='red', label='z_ij')
    plt.plot(x, z_updata1[cluster*1000+3], color='blue', label='z_update')
    plt.legend()
    plt.savefig(f"/share/home/kma/metabolism/wash_data/plot_{cluster}.png")

torch.save(init_sigma2, "/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/sigma_cluster12.pth")
torch.save(init_phi2, "/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/phi_cluster12.pth")
torch.save(init_h2, "/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/h_cluster12.pth")
torch.save(k_ij, '/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/kij_cluster12.pth')
torch.save(sigphiij_zf, '/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/sigphiij_zf_cluster12.pth')
'''


k_ij = zij / init_sigma[:, None] / init_phi[None,:]
sigma_phi = init_sigma[:, None] * init_phi[None,:]
# 展平k_ij
k_flatten = k_ij.view(-1)  # view(-1)将二维数组展平为一维数组
sigma_phi_flatten = sigma_phi.view(-1)
two_dim_array = torch.stack((k_flatten, sigma_phi_flatten))

# 对一维数组进行排序
k_sorted, indices = torch.sort(k_flatten)

# 初始化k_l和sig_phi_l
k_l = torch.zeros(6, device=device)
sig_phi_l = torch.zeros(6, device=device)

# 计算k_l和sig_phi_l
for i in range(6):
    # 计算第i个元素的索引
    i_index = int((i + 1) / 6 * k_sorted.numel())
    # 获取第i个元素的值并存储到k_l[i]
    k_l[i] = k_sorted[i_index]
    sig_phi_pos = indices[i_index]
    sig_phi_l[i] = two_dim_array[1,sig_phi_pos]


torch.save(k_l, "parameters_estimate/k_l.pth")
torch.save(sig_phi_l, "parameters_estimate/sig_phi_l.pth")
'''