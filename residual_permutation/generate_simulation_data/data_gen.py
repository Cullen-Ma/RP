import matplotlib.pyplot as mp
from scipy.optimize import least_squares
import torch
import math
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from util import gen_folder, linear_interpolation, index_of_x_min_cut
import random


def exception_handling(x):

    x[x < 0.0] = 0.0
    x[torch.isnan(x)] = 0


def feng_comb(t_samp, t_samp_mat, p):

    # 虽然参数p就是由归一化后的数据拟合而来，但由于前面的拟合可能会有一些差错，比方早期peak拟合并不好，所以在生成模拟数据时需要再次进行归一化

    C_0_samp = torch.zeros_like(t_samp, device=t_samp.device)
    C_0_mat = torch.zeros_like(t_samp_mat, device=t_samp_mat.device)
    A_1, A_2, A_3, lambda_1, lambda_2, lambda_3, alpha = p
    C_0_samp = (A_1 * t_samp ** alpha - A_2 - A_3) * torch.exp(-lambda_1 * t_samp) + \
                A_2 * torch.exp(-lambda_2 * t_samp) + \
                A_3 * torch.exp(-lambda_3 * t_samp)
    C_0_mat = (A_1 * t_samp_mat ** alpha - A_2 - A_3) * torch.exp(-lambda_1 * t_samp_mat) + \
               A_2 * torch.exp(-lambda_2 * t_samp_mat) + \
               A_3 * torch.exp(-lambda_3 * t_samp_mat)

    scale = torch.max(C_0_samp)

    return C_0_samp / scale, C_0_mat / scale


def feng_torch(t, p):

    A_1, A_2, A_3, lambda_1, lambda_2, lambda_3, alpha = p
    C_0 = (A_1 * t ** alpha - A_2 - A_3) * torch.exp(-lambda_1 * t) + \
           A_2 * torch.exp(-lambda_2 * t) + \
           A_3 * torch.exp(-lambda_3 * t)
    
    return C_0 


def feng_np(t, p):

    A_1, A_2, A_3, lambda_1, lambda_2, lambda_3, alpha = p
    C_0 = (A_1 * t ** alpha - A_2 - A_3) * np.exp(-lambda_1 * t) + \
            A_2 * np.exp(-lambda_2 * t) + \
            A_3 * np.exp(-lambda_3 * t)
    
    return C_0


def feng_fit(p, t, C_0_dc):

    return feng_np(t, p) - C_0_dc


def integration(t, f):

    _ =  0.5 * t[0] * f[0]
    for i in range(1, t.shape[0] - 1):
        _ += 0.5 * (t[i + 1] - t[i]) * (f[i + 1] + f[i])

    return _


def t_data_gen(delta_frames):

    # unit of delta_frames: s

    frame_split = torch.cumsum(torch.cat([torch.zeros(size=(1,)), delta_frames], dim=0), dim=0)
    t_mid = torch.zeros(size=(frame_split.shape[0] - 1, ))
    for i in range(t_mid.shape[0]):
        t_mid[i] = (frame_split[i] + frame_split[i + 1]) / 2

    return frame_split / 60, t_mid / 60  # unit conversion to min


def gen_t_samp_mat(end, t_res):

    # t_res must be divisible by 1
    # unit of t_res: second
    # unit of end: min

    step = t_res / 60
    n_samps = int(end / (t_res / 60)) + 1
    t_samp = torch.linspace(start=0, end=end, steps=n_samps)
    t_samp_mat = torch.zeros(size=[n_samps, n_samps])
    t_samp_mat_r = torch.zeros(size=[n_samps, n_samps])
    for i in range(n_samps):
        t_samp_mat[i, 0: i + 1] = t_samp[0: i + 1]
        t_samp_mat_r[i, 0: i + 1] = torch.flip(t_samp[0: i + 1], dims=[0])

    return step, t_samp, t_samp_mat, t_samp_mat_r  # unit: min


def TAC_gen(C_0_mat, kps, step, t_samp_mat_r):

    # t_res must be divisible by 1, such as 0.2, 0.1
    # unit of t_res: second
    # unit of end: min
    # rf: response function

    K1, k2, k3, k4 = kps

    k2_k3_k4 = k2 + k3 + k4
    k2_k4 = k2 * k4
    a1 = 0.5 * (k2_k3_k4 - (k2_k3_k4 ** 2 - 4 * k2_k4) ** 0.5)
    a2 = 0.5 * (k2_k3_k4 + (k2_k3_k4 ** 2 - 4 * k2_k4) ** 0.5)

    rf_1_r = torch.exp(-a1 * t_samp_mat_r)
    rf_2_r = torch.exp(-a2 * t_samp_mat_r)

    # covolution
    _ = C_0_mat * rf_1_r
    _1 = (torch.sum(_, dim=1) * step)

    _ = C_0_mat * rf_2_r
    _2 = (torch.sum(_, dim=1) * step)
    
    C_1_samp = K1 / (a2 - a1) * ((k4 - a1) * _1 + (a2 - k4) * _2)
    C_2_samp = K1 * k3 / (a2 - a1) * (_1 - _2)
    
    return C_1_samp, C_2_samp


def noise_model(nl_samp_mat, end, frame_interval, decay, noise_level):

    # unit of frame_interval: s

    exception_handling(nl_samp_mat)  # 数据进来的时候需要进行异常处理，如果出现了nan值将无法计算mean

    mean_mat = torch.zeros_like(nl_samp_mat)
    std_mat = torch.zeros_like(nl_samp_mat)

    n_frames = int(end * 60 / frame_interval)
    frame_split, t_mid = t_data_gen(torch.ones(size=[n_frames, ]) * frame_interval)

    for i in range(n_frames):
        index_start, index_end = int(frame_split[i] * 60 / t_res), int(frame_split[i + 1] * 60 / t_res)
        _1 = torch.mean(nl_samp_mat[index_start: index_end, :], dim=0, keepdim=True).repeat(index_end - index_start, 1)
        mean_mat[index_start: index_end, :] = _1
        if torch.sum(_1) == 0.0:
            _2 = 0.005 * torch.max(nl_samp_mat, dim=0, keepdim=True)[0].repeat(index_end - index_start, 1) * math.exp(decay * t_mid[i])
        else:
            _2 = noise_level * (_1 * math.exp(decay * t_mid[i]) / (frame_interval / 60)) ** 0.5
        std_mat[index_start: index_end, :] = _2

    # 耗时环节, 这里不需要进行异常处理，因为散点本身就会有一些小于0的，如果我们将这些小于0的赋值为0，会导致最终结果偏高
    n_samp_mat = torch.normal(mean=mean_mat, std=std_mat)  

    return n_samp_mat


def add_delay(nl_samp_mat, t_res, d_ts):

    # unit of d_ts: s, [nl_samp_mat.shape[1],]
    # nl_samp_mat是没有延迟的理想化
    
    _1 = torch.zeros_like(nl_samp_mat)
    for i_samp in range(nl_samp_mat.shape[1]):
        _2 = int(d_ts[i_samp] / t_res)
        if _2 != 0:
            _1[_2:, i_samp] = nl_samp_mat[: -_2, i_samp]
        else:
            _1[:, i_samp] = nl_samp_mat[:, i_samp]
        
    return _1


def from_samp_to_mid_mat(samp_mat, t_res, frame_split):

    index_start = (frame_split[:-1] * 60 / t_res).long()
    index_end = (frame_split[1:] * 60 / t_res).long()
    mid_mat = torch.stack([torch.mean(samp_mat[start:end, :], dim=0) for start, end in zip(index_start, index_end)])

    exception_handling(mid_mat)

    return mid_mat

  
if __name__ == '__main__':

    # 准备好需要的文件夹
    gen_folder('data1/fit')
    gen_folder('data1/kps_label')
    gen_folder('data1/input/C_0')
    gen_folder('data1/input/C_t/noise')
    gen_folder('data1/input/C_t/noiseless')
    gen_folder('data1/output/C_1')
    gen_folder('data1/output/C_2')
    

    # 生成数据的相关参数
    device = torch.device('cuda:0')
    units = {'UCD': ['GUC', ['2.0.0'], 
                            [12]]}  # C_0出现的粗估
    kernel_size = 1  # 均值滤波的卷积核大小
    end_1 = 60  # unit: min，模拟数据中动态扫描的末尾点，与下面delta_frames的终点必须一致
    end_2 = 55  # unit: min，t_col的中点，应确保t_col[-1] <= t_mid[-1]，避免外部插值
    t_res = 0.2  # unit: second
    n_samps = int(end_1 * 60 / t_res + 1)
    frame_interval = 1  # unit: s, 用于生成噪声的小区间长度
    decay = math.log(2) / 109.8  # 18F-FDG
    n_realizations = 100  # 每一个tissue进行多少次随机模拟

    # 注意：delta_frames决定着t_col的最后一个点
    #直接替换
    delta_frames = torch.cat([torch.ones(size=(30,)) * 2,  # 1 min
                              torch.ones(size=(12,)) * 10,  # 3 min
                              torch.ones(size=(6,)) * 30,  # 6 min
                              torch.ones(size=(12,)) * 120,  # 30 min
                              torch.ones(size=(6,)) * 300], dim=0)
    frame_split, t_mid = t_data_gen(delta_frames)
    t_col = torch.linspace(start=0, end=end_2, steps=int(end_2 * 60 + 1))  # 默认时间间隔是1秒
    torch.save(t_mid, 'data1/t_mid.pth')  
    torch.save(t_col, 'data1/t_col.pth') 

    # kinetic parameters(mean and std)
    # vessel的K1,k2,k3不能设置为0，否则影响TAC_gen运行，反正后面fv也会发挥作用
    # 前面的组织delay有较大概率小于0
    #                          fv              K1              k2              k3              k4           delay
    '''tissues = {'lung':        [(0.128, 0.039), (0.023, 0.012), (0.205, 0.090), (0.001, 0.004), (0.0, 0.0), (0.0, 1)],
               'vessel':      [(1.0,   0.0),   (0.001, 0.0),   (0.001, 0.0),   (0.001, 0.0),   (0.0, 0.0), (0.0, 1)],  
               'all_regions': [(0.076, 0.080), (0.529, 0.584), (1.171, 1.277), (0.037, 0.057), (0.0, 0.0), (6.377,  6.331)],
               'myocardium':  [(0.192, 0.068), (0.820, 0.356), (2.597, 1.215), (0.098, 0.088), (0.0, 0.0), (0.190,  0.512)],
               'lesion':      [(0.089, 0.092), (0.605, 0.573), (1.443, 1.272), (0.062, 0.074), (0.0, 0.0), (4.250,  5.254)],

               'grey_matter': [(0.030, 0.006), (0.107, 0.018), (0.163, 0.029), (0.066, 0.013), (0.0, 0.0), (5.238,  0.944)],
               'liver':       [(0.001, 0.003), (0.636, 0.291), (0.737, 0.381), (0.002, 0.001), (0.0, 0.0), (12.333, 2.536)],
               'muscle':      [(0.001, 0.001), (0.026, 0.012), (0.249, 0.136), (0.016, 0.006), (0.0, 0.0), (17.000, 3.715)],
               'spleen':      [(0.083, 0.029), (1.458, 0.467), (2.709, 0.882), (0.006, 0.003), (0.0, 0.0), (6.333,  1.278)]}'''
    '''tissues = {'lung':        [(0.128, 0.039), (0.023, 0.012), (0.205, 0.090), (0.001, 0.004), (0.0, 0.0), (0.0, 0.0)],
               'vessel':      [(1.0,   0.0),   (0.001, 0.0),   (0.001, 0.0),   (0.001, 0.0),   (0.0, 0.0), (0.0, 0.0)],  
               'all_regions': [(0.076, 0.080), (0.529, 0.584), (1.171, 1.277), (0.037, 0.057), (0.0, 0.0), (0.0, 0.0)],
               'myocardium':  [(0.192, 0.068), (0.820, 0.356), (2.597, 1.215), (0.098, 0.088), (0.0, 0.0), (0.0, 0.0)],
               'lesion':      [(0.089, 0.092), (0.605, 0.573), (1.443, 1.272), (0.062, 0.074), (0.0, 0.0), (0.0, 0.0)],

               'grey_matter': [(0.030, 0.006), (0.107, 0.018), (0.163, 0.029), (0.066, 0.013), (0.0, 0.0), (0.0, 0.0)],
               'liver':       [(0.001, 0.003), (0.636, 0.291), (0.737, 0.381), (0.002, 0.001), (0.0, 0.0), (0.0, 0.0)],
               'muscle':      [(0.001, 0.001), (0.026, 0.012), (0.249, 0.136), (0.016, 0.006), (0.0, 0.0), (0.0, 0.0)],
               'spleen':      [(0.083, 0.029), (1.458, 0.467), (2.709, 0.882), (0.006, 0.003), (0.0, 0.0), (0.0, 0.0)]}'''
    tissues = {'lung':        [(0.128, 0.0), (0.023, 0.0), (0.205, 0.0), (0.001, 0.0), (0.0, 0.0), (0.0, 0.0)],
               'vessel':      [(1.0,   0.0),   (0.001, 0.0),   (0.001, 0.0),   (0.001, 0.0),   (0.0, 0.0), (0.0, 0.0)],  
               'all_regions': [(0.076, 0.0), (0.529, 0.0), (1.171, 0.0), (0.037, 0.0), (0.0, 0.0), (0.0, 0.0)],
               'myocardium':  [(0.192, 0.0), (0.820, 0.0), (2.597, 0.0), (0.098, 0.0), (0.0, 0.0), (0.0, 0.0)],
               'lesion':      [(0.089, 0.0), (0.605, 0.0), (1.443, 0.0), (0.062, 0.0), (0.0, 0.0), (0.0, 0.0)],

               'grey_matter': [(0.030, 0.0), (0.107, 0.0), (0.163, 0.0), (0.066, 0.0), (0.0, 0.0), (0.0, 0.0)],
               'liver':       [(0.001, 0.0), (0.636, 0.0), (0.737, 0.0), (0.002, 0.0), (0.0, 0.0), (0.0, 0.0)],
               'muscle':      [(0.001, 0.0), (0.026, 0.0), (0.249, 0.0), (0.016, 0.0), (0.0, 0.0), (0.0, 0.0)],
               'spleen':      [(0.083, 0.0), (1.458, 0.0), (2.709, 0.0), (0.006, 0.0), (0.0, 0.0), (0.0, 0.0)]}

    noise_levels = [0.16]

    # preparation for time input
    step, t_samp, t_samp_mat, t_samp_mat_r = gen_t_samp_mat(end_1, t_res)
    t_samp = t_samp.to(device)
    t_samp_mat = t_samp_mat.to(device)
    t_samp_mat_r = t_samp_mat_r.to(device)

    # ===================================================== 拟合真实采集的C_0 =====================================================
    ps = [] 
    for unit in units:
        
        subset = units[unit][0]
        individuals = units[unit][1]
        d_0s = units[unit][2]
        
        for i_individual, individual in enumerate(individuals):
            
            t_mid_individual = torch.load('/share/home/kma/swr_simulation_data/t_mid/%s-%s-%s.pth' % (unit, subset, individual))
            C_0_ndc = torch.load('/share/home/kma/swr_simulation_data/TAC/C_0/%s-%s-%s(kernel_size_%d).pth' % (unit, subset, individual, kernel_size))
            scale = torch.max(C_0_ndc)
            C_0_ndc /= scale  # 在归一化的空间进行拟合
            d_0_init = d_0s[i_individual]
            d_0s_alt = np.linspace(start=d_0_init * 0.2, stop=d_0_init * 1.2, num=int(0.4 * d_0_init / 0.1) + 1, endpoint=True)  # 备选的
            r = []  # 用于记录残差值
            p_alt = []  # 用于保存拟合的参数，后面根据残差值进行选取
            for d_0 in d_0s_alt:

                fun = interp1d(x=np.array(t_mid_individual - d_0 / 60), y=np.array(C_0_ndc), kind='linear', fill_value="extrapolate")
                C_0_shift = fun(np.array(t_mid_individual))  

                p_alt.append(least_squares(feng_fit, 
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                             method='trf', loss='linear',
                             max_nfev=100,
                             ftol=1e-10, xtol=1e-10, gtol=1e-10, args=(np.array(t_mid_individual), C_0_shift)).x)
                r.append(torch.mean(integration(t_mid_individual, torch.abs(feng_torch(t_mid_individual, p_alt[-1]) - C_0_shift))).item())
                
            _ = r.index(min(r))
            d_0s.append(d_0s_alt[_])
            p = p_alt[_]
            fun = interp1d(x=np.array(t_mid_individual - d_0s_alt[_] / 60), y=np.array(C_0_ndc), kind='linear', fill_value="extrapolate")
            C_0_dc = fun(np.array(t_mid_individual))
            ps.append(p)
            
            
            # 画出拟合结果
            one_min_cut = index_of_x_min_cut(t_mid_individual, 1) + 1  # 原本返回的是索引，要加上一才是裁剪的数目
            fig_1 = mp.subplot(2, 2, 1)
            fig_1.plot(t_mid_individual, C_0_ndc, label='ndc', color='black')
            fig_1.plot(t_mid_individual, C_0_dc, label='dc', color='blue')
            mp.legend()
            fig_2 = mp.subplot(2, 2, 2)
            fig_2.plot(t_mid_individual[: one_min_cut], C_0_ndc[: one_min_cut], label='ndc', color='black')
            fig_2.plot(t_mid_individual[: one_min_cut], C_0_dc[: one_min_cut], label='dc', color='blue')
            mp.legend()
            fig_3 = mp.subplot(2, 2, 3)
            fig_3.scatter(t_mid_individual[: one_min_cut], C_0_ndc[: one_min_cut], label='ndc', color='black')
            fig_3.plot(t_mid_individual[: one_min_cut], feng_torch(t_mid_individual, p)[: one_min_cut], label='fit(feng)', color='blue')
            mp.legend()
            fig_4 = mp.subplot(2, 2, 4)
            fig_4.scatter(t_mid_individual[one_min_cut:], C_0_ndc[one_min_cut:], label='ndc', color='black')
            fig_4.plot(t_mid_individual[one_min_cut:], feng_torch(t_mid_individual, p)[one_min_cut:], label='fit(feng)', color='blue')
            mp.legend()

            mp.savefig('data1/fit/%s-%s-%s(kernel_size=%d).png' % (unit, subset, individual, kernel_size))
            mp.close()


    # ===================================================== 生成数据 =====================================================
    for i_C_0, p in enumerate(ps):
        
        # 随机生成d_0
        random_d_0 = random.choice(d_0s)
        random_d_0s = torch.normal(mean=0, std=0, size=[len(tissues) * n_realizations,])


        # 生成动力学参数（异常处理 + delay补加d_0）
        kps_label = torch.zeros(size=[6, len(tissues) * n_realizations])  # fv, K1, k2, k3, k4, d_t
        for i_tissue, tissue in enumerate(tissues):
            for i_kp, kp in enumerate(tissues[tissue]):
                _ = torch.normal(mean=kp[0], std=kp[1], size=[n_realizations,])
                if i_kp != 5:  # 除了延迟值以外的参数，如果小于0，校正回中心值
                    _[_ <= 0] = kp[0]
                kps_label[i_kp, i_tissue * n_realizations: (i_tissue + 1) * n_realizations] = _
        kps_label[torch.isnan(kps_label)] = 0        
        kps_label[5, :] += random_d_0s
        kps_label[5, :][kps_label[5, :] <= 0] = 0  # 延迟值必须大于等于0，d_t可以小于等于d_0
        torch.save(kps_label, 'data1/kps_label/i_C_0=%d.pth' % i_C_0)
        

        # 无噪声输入与输出所需要的索引
        ind_col = [i for i in range(0, int(end_2 * 60 * (1 / t_res)) + 1, int(1 / t_res))]


        # 生成模拟TAC(C_0无需加噪)
        C_0_samp, C_0_mat = feng_comb(t_samp, t_samp_mat, p) 
        C_0_mat *= (1 - 0.5 * torch.eye(t_samp_mat.shape[0], device=device))
        C_1s_samp = torch.zeros(size=[n_samps, len(tissues) * n_realizations])
        C_2s_samp = torch.zeros(size=[n_samps, len(tissues) * n_realizations])
        for i_C_t in tqdm(range(len(tissues) * n_realizations), desc='generating C1 and C2(i_C_0=%d)' % i_C_0):
            C_1s_samp[:, i_C_t], C_2s_samp[:, i_C_t] = \
            TAC_gen(C_0_mat=C_0_mat, kps=kps_label[1: 5, i_C_t], step=step, t_samp_mat_r=t_samp_mat_r)
        C_ts_samp = kps_label[0: 1, :] * C_0_samp.cpu().reshape(-1, 1).repeat(1, len(tissues) * n_realizations) + \
                    (1 - kps_label[0: 1, :]) * (C_1s_samp + C_2s_samp)


        # 初始状态init（col+mid）
        torch.save(C_0_samp[ind_col].cpu(), 'data1/input/C_0/initcol(i_C_0=%d).pth' % i_C_0)
        torch.save(C_1s_samp[ind_col, :].cpu(), 'data1/output/C_1/initcol(i_C_0=%d).pth' % i_C_0) 
        torch.save(C_2s_samp[ind_col, :].cpu(), 'data1/output/C_2/initcol(i_C_0=%d).pth' % i_C_0)
        torch.save(C_ts_samp[ind_col, :], 'data1/input/C_t/noiseless/initcol(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_0_samp.reshape(-1, 1).cpu(), t_res, frame_split).flatten(), 'data1/input/C_0/initmid(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_1s_samp.cpu(), t_res, frame_split), 'data1/output/C_1/initmid(i_C_0=%d).pth' % i_C_0) 
        torch.save(from_samp_to_mid_mat(C_2s_samp.cpu(), t_res, frame_split), 'data1/output/C_2/initmid(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_ts_samp, t_res, frame_split), 'data1/input/C_t/noiseless/initmid(i_C_0=%d).pth' % i_C_0)


        # 延迟状态general，其中C_0包含2种延迟状态（C_0_samp_add_random_d_0s---general，即没与C_t对齐。C_0_samp_add_d_ts---align即对齐状态）
        C_0_samp_add_random_d_0s = add_delay(C_0_samp.cpu().reshape(-1, 1).repeat(1, len(tissues) * n_realizations), t_res, random_d_0s)
        C_0_samp_add_d_ts = add_delay(C_0_samp.cpu().reshape(-1, 1).repeat(1, len(tissues) * n_realizations), t_res, kps_label[5, :])
        C_1s_samp = add_delay(C_1s_samp, t_res, kps_label[5, :])
        C_2s_samp = add_delay(C_2s_samp, t_res, kps_label[5, :])
        C_ts_samp = kps_label[0: 1, :] * C_0_samp_add_d_ts + (1 - kps_label[0: 1, :]) * (C_1s_samp + C_2s_samp)
        torch.save(C_0_samp_add_random_d_0s[ind_col, :], 'data1/input/C_0/generalcol(i_C_0=%d).pth' % i_C_0)
        torch.save(C_0_samp_add_d_ts[ind_col, :], 'data1/input/C_0/aligncol(i_C_0=%d).pth' % i_C_0)
        torch.save(C_1s_samp[ind_col, :], 'data1/output/C_1/generalcol(i_C_0=%d).pth' % i_C_0)
        torch.save(C_2s_samp[ind_col, :], 'data1/output/C_2/generalcol(i_C_0=%d).pth' % i_C_0)
        torch.save(C_ts_samp[ind_col, :], 'data1/input/C_t/noiseless/generalcol(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_0_samp_add_random_d_0s, t_res, frame_split), 'data1/input/C_0/generalmid(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_0_samp_add_d_ts, t_res, frame_split), 'data1/input/C_0/alignmid(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_1s_samp, t_res, frame_split), 'data1/output/C_1/generalmid(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_2s_samp, t_res, frame_split), 'data1/output/C_2/generalmid(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_ts_samp, t_res, frame_split), 'data1/input/C_t/noiseless/generalmid(i_C_0=%d).pth' % i_C_0)
        torch.save(from_samp_to_mid_mat(C_ts_samp, t_res, frame_split), 'data1/input/C_t/noiseless/generalmid(i_C_0=%d).pth' % i_C_0)

        # 增加噪声
        for noise_level in noise_levels:
            C_ts_noise = from_samp_to_mid_mat(noise_model(C_ts_samp, end_1, frame_interval, decay, noise_level), t_res, frame_split)
            torch.save(C_ts_noise, 'data1/input/C_t/noise/generalmid(i_C_0=%d,noise_level=%.2f).pth' % (i_C_0, noise_level))
            C_ts_noise = linear_interpolation(t_col, t_mid, C_ts_noise)
            torch.save(C_ts_noise, 'data1/input/C_t/noise/generalcol(i_C_0=%d,noise_level=%.2f).pth' % (i_C_0, noise_level))
        

    # ===================================================== 可视化生成结果 =====================================================
        
        gen_folder('data1/demo/%d' % i_C_0)
        one_min_cut_mid = index_of_x_min_cut(t_mid, 1) + 1
        one_min_cut_col = index_of_x_min_cut(t_col, 1) + 1

        kps = torch.load('data1/kps_label/i_C_0=%d.pth' % i_C_0)
        C_0_initcol = torch.load('data1/input/C_0/initcol(i_C_0=%d).pth' % i_C_0)
        C_0s_generalcol = torch.load('data1/input/C_0/generalcol(i_C_0=%d).pth' % i_C_0)
        C_0s_aligncol = torch.load('data1/input/C_0/aligncol(i_C_0=%d).pth' % i_C_0)
        C_1s_col = torch.load('data1/output/C_1/generalcol(i_C_0=%d).pth' % i_C_0)
        C_2s_col = torch.load('data1/output/C_2/generalcol(i_C_0=%d).pth' % i_C_0)
        C_ts_nlcol = torch.load('data1/input/C_t/noiseless/generalcol(i_C_0=%d).pth' % i_C_0)

        for i_fig in range(100):

            noise_level = random.choice(noise_levels)
            i_C_t = random.randint(0, C_ts_nlcol.shape[1] - 1)
            fv, K1, k2, k3, k4, d_t = kps[:, i_C_t]
            C_0_generalcol = C_0s_generalcol[:, i_C_t]
            C_0_aligncol = C_0s_aligncol[:, i_C_t]
            C_1_col = C_1s_col[:, i_C_t]
            C_2_col = C_2s_col[:, i_C_t]
            C_t_nlcol = C_ts_nlcol[:, i_C_t]
            C_t_nmid = torch.load('data1/input/C_t/noise/generalmid(i_C_0=%d,noise_level=%.2f).pth' % (i_C_0, noise_level))[:, i_C_t]
            C_t_ncol = torch.load('data1/input/C_t/noise/generalcol(i_C_0=%d,noise_level=%.2f).pth' % (i_C_0, noise_level))[:, i_C_t]

            mp.figure(figsize=(10, 4), dpi=100)
            
            fig_1 = mp.subplot(1, 2, 1)
            mp.title('fv=%.3f, K1=%.3f, k2=%.3f, k3=%.3f, k4=%.3f' % (fv, K1, k2, k3, k4))
            fig_1.plot(t_col, fv * C_0_initcol, label='fv * C_0_initcol', color='red', alpha=0.3)
            fig_1.plot(t_col, fv * C_0_generalcol, label='fv * C_0_generalcol', color='red', alpha=0.7)
            fig_1.plot(t_col, fv * C_0_aligncol, label='fv * C_0_aligncol', color='red')
            fig_1.plot(t_col, C_1_col, label='C_1_col', color='green')
            fig_1.plot(t_col, C_2_col, label='C_2_col', color='orange')
            fig_1.scatter(t_mid, C_t_nmid, marker='s', label='C_t_nmid(%.2f)' % noise_level, color='black')
            fig_1.plot(t_col, C_t_nlcol, label='C_t_nlcol', color='blue')
            fig_1.plot(t_col, C_t_ncol, label='C_t_ncol', color='blue', alpha=0.3)
            mp.legend()
            fig_2 = mp.subplot(1, 2, 2)
            mp.title('d_t=%.3fs' % d_t)
            fig_2.plot(t_col[: one_min_cut_col], fv * C_0_initcol[: one_min_cut_col], label='fv * C_0_initcol', color='red', alpha=0.3)
            fig_2.plot(t_col[: one_min_cut_col], fv * C_0_generalcol[: one_min_cut_col], label='fv * C_0_generalcol', color='red', alpha=0.7)
            fig_2.plot(t_col[: one_min_cut_col], fv * C_0_aligncol[: one_min_cut_col], label='fv * C_0_aligncol', color='red')
            fig_2.plot(t_col[: one_min_cut_col], C_1_col[: one_min_cut_col], label='C_1_col', color='green')
            fig_2.plot(t_col[: one_min_cut_col], C_2_col[: one_min_cut_col], label='C_2_col', color='orange')
            fig_2.scatter(t_mid[: one_min_cut_mid], C_t_nmid[: one_min_cut_mid], 
                              marker='s', label='C_t_nmid(%.2f)' % noise_level, color='black')
            fig_2.plot(t_col[: one_min_cut_col], C_t_nlcol[: one_min_cut_col], label='C_t_nlcol', color='blue')
            fig_2.plot(t_col[: one_min_cut_col], C_t_ncol[: one_min_cut_col], label='C_t_ncol', color='blue', alpha=0.3)
            mp.legend()

            mp.savefig('data1/demo/%d/%d.png' % (i_C_0, i_fig))
            mp.close()
