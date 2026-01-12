import torch
import matplotlib.pyplot as mp
import os


def gen_folder(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def gen_diff_mat(start, end, n_cols):
        
    # Finite difference method: coefficient matrix

    W = torch.zeros(size=[n_cols, n_cols])
    gs = (end - start) / (n_cols - 1)  # grid space
    W[0, 0], W[0, 1], W[0, 2] = -3 / (2 * gs), 4 / (2 * gs), -1 / (2 * gs)  # second-order
    W[-1, -3], W[-1, -2], W[-1, -1] = 1 / (2 * gs), -4 / (2 * gs), 3 / (2 * gs)  # second-order
    W[1, 0], W[1, 1], W[1, 2], W[1, 3] = -2 / (6 * gs),  -3 / (6 * gs), 6 / (6 * gs), -1 / (6 * gs)  # third-order
    W[-2, -4], W[-2, -3], W[-2, -2], W[-2, -1] = 1 / (6 * gs), -6 / (6 * gs), 3 / (6 * gs), 2 / (6 * gs)  # third-order
    for i in range(2, n_cols - 2):  # forth-order
        W[i, i - 2] = 1 / (12 * gs)
        W[i, i - 1] = -8 / (12 * gs)
        W[i, i + 1] = 8 / (12 * gs)
        W[i, i + 2] = -1 / (12 * gs)
    return W


def linear_regression(Xs, Y): 

    # 截距为0的情况

    n_Xs = len(Xs)
    n_cols, n_TACs = Xs[0].shape 
    for i_X in range(n_Xs):
        Xs[i_X] = Xs[i_X].T.reshape(n_TACs, n_cols, 1)
    Y = Y.T.reshape(n_TACs, n_cols, 1)
    X = torch.cat(Xs, dim=2)

    # 使用矩阵方法求解系数，假设截距为0
    XtX = torch.matmul(torch.transpose(X, 2, 1), X)
    XtX_inv = torch.linalg.inv(XtX)
    XtY = torch.matmul(torch.transpose(X, 2, 1), Y)
    coefficients = torch.matmul(XtX_inv, XtY)

    return [coefficients[:, i: i+1, 0].T for i in range(n_Xs)]



def linear_interpolation(t_new, t_old, value_old):

    # t_new: shape [n_new,], 要进行插值的查询点, 插值点的范围最好比t_old要长
    # t_old: shape [n_old,], 给定的时间点
    # value_old: shape [n_old, n_TACs], 每列是函数 f_i 在 t_old 上的值
    
    # 补充好初值点
    t_old = torch.cat([torch.zeros(size=[1, ]), t_old], dim=0)
    value_old = torch.cat([torch.zeros(size=[1, value_old.shape[1]]), value_old], dim=0)

    # 找到查询点在哪两个t_old 之间 (返回的是左边的索引)
    indices = torch.searchsorted(t_old, t_new).clamp(max=t_old.shape[0] - 1)
    indices_lower = (indices - 1).clamp(min=0)  # 保证索引不越界
    indices_upper = indices.clamp(max=t_old.shape[0] - 1)
    
    # 获取上下限的 t_old
    t_lower = t_old[indices_lower]
    t_upper = t_old[indices_upper]
    
    # 获取上下限的 value_old 对应值
    mid_lower = value_old[indices_lower]
    mid_upper = value_old[indices_upper]
    
    # 检查是否与已知点重合，若重合则直接返回已知值
    exact_match = (t_new == t_old[indices_upper])
    value_new = torch.where(exact_match.unsqueeze(-1), mid_upper, torch.zeros_like(mid_upper))
    
    # 计算插值权重（不处理完全重合的部分）
    weight_upper = (t_new - t_lower) / (t_upper - t_lower + 1e-8)
    weight_lower = 1 - weight_upper
    
    # 计算插值值
    value_new = torch.where(exact_match.unsqueeze(-1), mid_upper, 
                          weight_lower.unsqueeze(-1) * mid_lower + weight_upper.unsqueeze(-1) * mid_upper)

    return value_new



def demo_loss(start, end, 
              tralossrec_snl, tralossrec_sn, lossrec_rPINN, 
              vallossrec_snl, vallossrec_sn,  # rPINN没有验证集
              save_path, 
              snl_lowerbound, snl_upperbound,
              sn_lowerbound, sn_upperbound,
              rPINN_lowerbound, rPINN_upperbound):

    mp.figure(figsize=(20, 3), dpi=100)

    fig_1 = mp.subplot(1, 3, 1)
    fig_1.plot([i for i in range(start, end)], tralossrec_snl[start: end], label='snl(train)', color='blue')
    fig_1.plot([i for i in range(start, end)], vallossrec_snl[start: end], label='snl(validate)', color='red', alpha=0.3)
    mp.ylim(snl_lowerbound, snl_upperbound)
    mp.legend()
    mp.grid(axis='y')

    fig_2 = mp.subplot(1, 3, 2)
    fig_2.plot([i for i in range(start, end)], tralossrec_sn[start: end], label='sn(train)', color='blue')
    fig_2.plot([i for i in range(start, end)], vallossrec_sn[start: end], label='sn(validate)', color='red', alpha=0.3)
    mp.ylim(sn_lowerbound, sn_upperbound)
    mp.legend()
    mp.grid(axis='y')

    fig_3 = mp.subplot(1, 3, 3)
    fig_3.plot([i for i in range(start, end)], lossrec_rPINN[start: end], label='sPINN', color='blue')
    mp.ylim(rPINN_lowerbound, rPINN_upperbound)
    mp.legend()
    mp.grid(axis='y')

    mp.savefig(save_path, bbox_inches='tight')
    mp.close()



def bound(x, low, up):
    if low != None:
        x[x <= low] = low
    if up != None:
        x[x >= up] = up
    return x


def align(rC_0_generalcol, d_rS_1s_col):

    max_ind_rC_0_generalcol = rC_0_generalcol.argmax(dim=0)
    max_ind_d_S_1s = d_rS_1s_col.argmax(dim=0)  # [n_TACs], 获取每列最大值的索引
    
    # 计算最大值索引的差值
    row_shift = max_ind_d_S_1s - max_ind_rC_0_generalcol  # [n_TACs]
    
    # 创建一个结果矩阵
    n_cols, n_TACs = d_rS_1s_col.shape
    rC_0_aligncol = torch.zeros_like(d_rS_1s_col)

    # 构造每列的偏移
    for j in range(n_TACs):
        shift = row_shift[j].item()
        if shift > 0:
            rC_0_aligncol[shift: n_cols, j] = rC_0_generalcol[: n_cols-shift]
        else:
            rC_0_aligncol[:, j] = rC_0_generalcol  # 只出现在肺部，小概率事件，稍稍不对齐也不要紧

    return rC_0_aligncol


def compute_kps(C_0s_col, S_1s_col, S_2s_col, C_ts_col, D, revesible):

    S_1s_add_S_2s = S_1s_col + S_2s_col
    d_S_1s, d_S_2s = torch.mm(D, S_1s_col), torch.mm(D, S_2s_col)

    # 这里如果不.detach()会爆内存
    S_1s_col, S_2s_col, S_1s_add_S_2s, d_S_1s, d_S_2s = \
        S_1s_col.detach(), S_2s_col.detach(), S_1s_add_S_2s.detach(), d_S_1s.detach(), d_S_2s.detach()

    if revesible:
        [K1s, k2s_add_k3s, k4s_1] = linear_regression([C_0s_col, -S_1s_col, S_2s_col], d_S_1s)
        [k3s, k4s_2] = linear_regression([S_1s_col, -S_2s_col], d_S_2s)
        [fvs] = linear_regression([C_0s_col - S_1s_add_S_2s], C_ts_col - S_1s_add_S_2s)
        k2s = k2s_add_k3s - k3s
        k4s = (k4s_1 + k4s_2) / 2
        
    else:
        [K1s, k2s_add_k3s] = linear_regression([C_0s_col, -S_1s_col], d_S_1s)
        [k3s] = linear_regression([S_1s_col], d_S_2s)
        [fvs] = linear_regression([C_0s_col - S_1s_add_S_2s], C_ts_col - S_1s_add_S_2s)
        k2s = k2s_add_k3s - k3s
        k4s = torch.zeros_like(K1s, device=K1s.device)

    return d_S_1s, d_S_2s, \
           bound(fvs, 0, 1), \
           bound(K1s, 0, None), \
           bound(k2s, 0, None), \
           bound(k3s, 0, None), \
           bound(k4s, 0, None)


def index_of_x_min_cut(t, x):

    # unit of t: min
    # unit of x: min
    
    _ = torch.abs(t - x)
    cut = torch.argmin(_).item()

    return cut