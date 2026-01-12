import torch
import numpy as np
from torch.distributions import Normal
from sklearn.metrics import r2_score
import torch
from scipy.stats import norm  # 使用Scipy中的标准正态分布
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cmath
def map_to_standard_normal(Q):
    # 对输入张量Q进行排序
    
    sorted_Q, indices = torch.sort(Q)
    
    # 获取排序后的索引的秩，并转换成比例
    ranks = torch.arange(1, len(sorted_Q) + 1).float()
    proportions = (ranks - 0.5) / len(sorted_Q)
    
    # 使用标准正态分布的分位数函数找到对应的值
    standard_normals = norm.ppf(proportions.numpy())
    
    # 将结果重新排列回到原来的顺序
    _, unsort_indices = torch.sort(indices)
    mapped_Q = torch.tensor(standard_normals)[unsort_indices]
    
    return mapped_Q

def reflect_to_original_distribution(Q_standard_normal, Q):
    # 每个值的标准正态分布累计概率
    probabilities = torch.tensor(norm.cdf(Q_standard_normal))

    # 对 Q 进行排序
    sorted_Q, _ = torch.sort(Q)
    probabilities_np = probabilities.numpy()
    sorted_Q_np = sorted_Q.numpy()

    # 使用 numpy 的 interp 函数进行插值
    Q_prime_np = np.interp(probabilities_np, np.linspace(0, 1, num=len(sorted_Q_np)), sorted_Q_np)

    # 将结果转换回 torch 张量
    Q_prime = torch.tensor(Q_prime_np)
    return Q_prime
def r2_change(img, z, kind):
    sum = 0
    count = 0
    for i in range(kind):
        r2 = r2_score(img[i, :], z[i, :])
        if r2<0:
          count=count+1
          #print(i)
        #print(r2)
        sum = sum +r2
    print(count)
    return sum/kind

#sigma2 = torch.load('bootstrap/iterate_parameter_result_guc/sigma_cluster7.pth')
#sigma = torch.sqrt(sigma2)
phi2 = torch.load('/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/phi_cluster12.pth')
#phi = torch.sqrt(phi2)
sigphiij_zf = torch.load('/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/sigphiij_zf_cluster12.pth')
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
#phi = phi.unsqueeze(0).repeat(zij.shape[0], 1)
#sigma =  sigma.unsqueeze(1).repeat(1, zij.shape[1])
hij2 = torch.load('/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/h_cluster12.pth')
hij = torch.sqrt(hij2)

cluster_label = 12
residual = img_62 - zij
epsilon = residual/sigphiij_zf
#np.save('bootstrap/iterate_parameter_result_guc/epsilon.npy', epsilon)
hij_label = torch.load('/share/home/kma/metabolism/bootstrap/iterate_parameter_result_guc_ks1/h_label_ij_cluster7.pth').numpy()
eta = np.zeros_like(hij_label, dtype=float) 
q_function_list = []
for i in range(cluster_label):
    #生成一个标准正态分布
    n,t = np.where(hij_label == i+1)
    epsilon_ij = epsilon[n, t]
    sorted_data = np.sort(epsilon_ij)
    # 计算每个数据点的经验累积分布函数(ECDF)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / (len(sorted_data) + 1)
    # 找到每个经验分位数对应的理论分位数（使用标准正态分布）
    theoretical_quantiles = norm.ppf(empirical_cdf)
    # 获取排序后的索引
    sorted_indices = sorted(range(len(epsilon_ij)), key=lambda k: epsilon_ij[k])

    # 根据排序后的索引重排数组b
    eta_ij = [theoretical_quantiles[i] for i in sorted_indices]
    eta[n,t] = eta_ij
    # 构建 Q 函数：通过插值方法从经验分位数映射到理论分位数
    q_function = interp1d(theoretical_quantiles, sorted_data, bounds_error=False, fill_value="extrapolate")
    q_function_list.append(q_function)
#获得eta_ij后，使用3Dfft对每个time frame的3d周期图进行评估
fft_results = np.fft.fft(eta, axis=0)
lambda_weight_sum = np.dot(fft_results, 1/phi2)
# 计算复数的平方根
sqrt_lambda = np.sqrt(lambda_weight_sum)
sqrt_lambda_ij = np.tile(sqrt_lambda, (66, 1)).T
#generate！！！

for time in range(20):
    #ξ服从标准正态分布
    kci = np.random.randn(hij.shape[0], hij.shape[1])
    to_ifft = sqrt_lambda_ij*kci
    #此时获得了eta，但是有虚部！！！！
    ifft_results_eta = np.fft.ifft(to_ifft, axis=0)
    #只保留实部
    results_eta_real_part = np.real(ifft_results_eta)
    epsilon_gen = np.zeros_like(zij)
    for i in range(cluster_label):
        #生成一个标准正态分布
        n,t = np.where(hij_label == i+1)
        eta_cluster = results_eta_real_part[n,t]
        q = q_function_list[i]
        epsilon_gen_ij = q(eta_cluster)
        epsilon_gen[n,t] = epsilon_gen_ij


    hij = torch.nan_to_num(hij, nan=0.0)

    bootstrap_sample = zij + sigphiij_zf* epsilon_gen
    bootstrap_sample_np = bootstrap_sample.numpy()
    print(r2_change(img_62, bootstrap_sample_np, zij.shape[0]))
    bootstrap_sample_np[bootstrap_sample_np<0] = 0
    '''import matplotlib.pyplot as plt

    # 创建x轴坐标点
    x = np.arange(66)

    # 开始绘图
    plt.figure(figsize=(10, 5))  # 设置图像大小

    # 绘制两条曲线
    plt.plot(x, img_62[57,:], label='Array 1')  # 第一条曲线
    plt.plot(x, bootstrap_sample[57,:], label='Array 2')  # 第二条曲线

    # 添加标题和标签
    plt.title('Two Arrays Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 显示图例
    plt.legend()

    # 展示图形
    plt.show()'''
    #result[z, y, :] = bootstrap_sample_np
    np.save('/share/home/kma/metabolism/bootstrap/baseline_bootstrap_sample_guc/baseline_bootstrap_%d'%time, bootstrap_sample_np)





