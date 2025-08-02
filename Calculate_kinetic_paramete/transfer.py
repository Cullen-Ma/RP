import numpy as np
import torch
import scipy.io as io
from scipy.interpolate import interp1d
from util import delay_locate, generate_folder
from tqdm import tqdm
import pdb
data_indices = [0, 1, 2, 3]
termination_dict = {0: 15, 1: 12, 2: 12, 3: 10}  # 定位在30s附近

for data_index in data_indices:
    generate_folder('data/imgs/imgs_mat(%d)' % data_index)
    generate_folder('data/frame_split')
    generate_folder('data/t_midpoint')
    generate_folder('data/delta_frames')
    generate_folder('data/C_0')
    generate_folder('data/segment')
    generate_folder('result')

    # 读取pth转npy
    frame_split = torch.load('../../real/data/frame_split/frame_split(%d).pth' % data_index, map_location=torch.device('cpu'))
    t_midpoint = torch.load('../../real/data/t_midpoint/t_midpoint(%d).pth' % data_index, map_location=torch.device('cpu'))
    delta_frames = torch.load('../../real/data/delta_frames/delta_frames(%d).pth' % data_index, map_location=torch.device('cpu'))
    C_0 = torch.load('../../real/data/process_data/C_0/C_0(%d).pth' % data_index, map_location=torch.device('cpu'))
    segment = torch.load('../../real/data/process_data/segment/segment(%d).pth' % data_index, map_location=torch.device('cpu'))

    # 由于全人体数据太大，只好分别保存
    d_0_index, d_0_value = delay_locate(t_midpoint, C_0, 0, termination_dict[data_index])
    pdb.set_trace()
    print(f'indexshape={d_0_index.shape}')
    print(f'valueshape={d_0_value.shape}')
    C_0 = interp1d(x=np.array(t_midpoint - d_0_value), y=np.array(C_0), kind='linear', fill_value="extrapolate")(np.array(t_midpoint))
    imgs = torch.load('../../real/data/process_data/imgs/imgs(%d).pth' % data_index, map_location=torch.device('cpu'))
    t_len, z_size, x_size, y_size = imgs.shape
    for z in tqdm(range(z_size)):
        imgs_z_numpy = np.array(imgs[:, z, :, :])
        for x in range(x_size):
            for y in range(y_size):
                if segment[z, x, y] == 1:
                    d_T_index, d_T_value = delay_locate(t_midpoint, imgs[:, z, x, y], d_0_index, termination_dict[data_index])
                    f = interp1d(x=np.array(t_midpoint - d_T_value), y=np.array(imgs[:, z, x, y]), kind='linear', fill_value="extrapolate")
                    imgs_z_numpy[:, x, y] = f(np.array(t_midpoint))
        io.savemat('data/imgs/imgs_mat(%d)/z_%d.mat' % (data_index, z + 1), {'value': imgs_z_numpy.astype(np.float64)})

    # npy转换mat
    io.savemat('data/frame_split/frame_split(%d).mat' % data_index, {'value': np.array(frame_split).astype(np.float64)})
    io.savemat('data/t_midpoint/t_midpoint(%d).mat' % data_index, {'value': np.array(t_midpoint).astype(np.float64)})
    io.savemat('data/delta_frames/delta_frames(%d).mat' % data_index, {'value': np.array(delta_frames).astype(np.float64)})
    io.savemat('data/C_0/C_0(%d).mat' % data_index, {'value': np.array(C_0.reshape(-1, 1)).astype(np.float64)})
    io.savemat('data/segment/segment(%d).mat' % data_index, {'value': np.array(segment).astype(np.float64)})