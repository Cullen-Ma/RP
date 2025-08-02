import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as mp
import torch
import numpy as np
import scipy.io as io
from util import generate_folder
import hdf5storage
import pdb
data_indices = [0]
#methods = ['OLS', 'WLS']
methods = ['WLS']
epochs = 50
cmap = mp.cm.binary  # 'Reds'#
print('begin')

def convert_to_nan(img, segment):
    H, W = img.shape
    for h in range(H):
        for w in range(W):
            if segment[h, w] == 1 and img[h, w] == 0:
                img[h, w] = np.nan
    return img


for data_index in data_indices:
    for method in methods:

        if data_index == 0:
            truncation_list = {'fv': 1, 'K1': 0.6, 'k2': 2, 'k3': 0.3, 'k4': 0, 'Ki': 0.05}
            # 左肺肿瘤，血管不清晰
            z_start, z_end = 180, 196
            y_start, y_end = 100, 116
        if data_index == 2:
            truncation_list = {'fv': 1, 'K1': 0.5, 'k2': 2, 'k3': 0.3, 'k4': 0, 'Ki': 0.05}
            z_start, z_end = 180, 196
            y_start, y_end = 100, 116
        if data_index == 2:
            truncation_list = {'fv': 1, 'K1': 0.5, 'k2': 2, 'k3': 0.3, 'k4': 0, 'Ki': 0.05}
            z_start, z_end = 180, 196
            y_start, y_end = 100, 116
        if data_index == 3:
            truncation_list = {'fv': 1, 'K1': 0.5, 'k2': 2, 'k3': 0.3, 'k4': 0, 'Ki': 0.05}
            z_start, z_end = 180, 196
            y_start, y_end = 100, 116
        pi = np.array(hdf5storage.loadmat('/share/home/kma/real_1/result/pi_%s_data_index_%d_epoch_%d.mat' % (method, data_index, epochs))['pi'])
        #pi = np.array(hdf5storage.loadmat('/share/home/kma/real_1/result/pi_WLS_data_index_0_epoch_50rp0.mat')['pi'])
    
        print(pi.shape)
        mat_segment = io.loadmat('/share/home/kma/real_1/data/segment/segment(0).mat')
        segment = mat_segment['value']
        z_size, x_size, y_size = segment.shape
        Ki = pi[:, :, :, 1: 2] * pi[:, :, :, 3: 4] / (pi[:, :, :, 2: 3] + pi[:, :, :, 3: 4])
        pi = np.concatenate([pi, Ki], axis=-1)
        pi = pi[85:265,87,:,5]
        segment_s = mat_segment['value']
        segment_s = segment_s[85:265,87:88,:]
        z,x,y = np.where(segment_s==1)
        '''z,y = np.where(pi>0.011)
        mask = z >= 80
        z = z[mask]
        y = y[mask]
        np.save("z.npy",z)
        np.save('y.npy',y)
        for z1, y1 in zip(z, y):
            pi[z1, y1] = pi[91, 87]'''
        label = np.load('/share/home/kma/real_1/label_cluster_tac_kmeans_12.npy')
        segment_s[z,x,y] = label
        z1,x1,y1 = np.where((segment_s==2) | (segment_s==3))
        pi[z1,y1] = 0
        #pi[pi<0.012] = 0
        pi[pi>0.02] = 0.02
        pi = np.nan_to_num(pi, 0)
        z_t,y_t = np.where(pi[90:110,60:68]>0.007)
        pi[z_t+90, y_t+60] = 0.02
        pdb.set_trace()
        mp.imshow(pi, cmap='binary') 
        mp.axis('on') 
        mp.colorbar()
        mp.savefig('slice_image_ki.png')

        '''for parameter_index, parameter in enumerate(truncation_list):
            generate_folder('fig/%d/%s/MIP' % (data_index, method))
            generate_folder('fig/%d/%s/slice/intact/%s' % (data_index, method, parameter))
            generate_folder('fig/%d/%s/slice/capture/%s' % (data_index, method, parameter))
                
            _ = pi[:, :, :, parameter_index].copy()
            _[_ >= truncation_list[parameter]] = truncation_list[parameter]

            # MIP
            # coronal
            mp.figure(dpi=500)
            mp.axis('off')
            mp.xticks([])
            mp.yticks([])
            #mp.title(parameter)
            #mp.imshow(np.max(_, axis=1), cmap=cmap)
            #mp.colorbar()
            mp.savefig('fig/%d/%s/MIP/coronal(%s).png' % (data_index, method, parameter), bbox_inches='tight')
            mp.cla()
    
            # sagittal
            mp.figure(dpi=500)
            mp.axis('off')
            mp.xticks([])
            mp.yticks([])
            #mp.title(parameter)
            mp.imshow(np.max(_, axis=2), cmap=cmap)
            mp.colorbar()
            mp.savefig('fig/%d/%s/MIP/sagittal(%s).png' % (data_index, method, parameter), bbox_inches='tight')
            mp.cla()
                
            # slice(只保存coronal)
            for x_index in range(x_size):
                # coronal(intact)
                mp.figure(dpi=500)
                mp.axis('off')
                mp.xticks([])
                mp.yticks([])
                #mp.title(parameter)
                mp.imshow(_[:, x_index, :], cmap=cmap)
                # mp.colorbar(fraction=0.02, orientation='horizontal', pad=0)
                #mp.colorbar()
                mp.savefig('fig/%d/%s/slice/intact/%s/%d.png' % (data_index, method, parameter, x_index), bbox_inches='tight')
                mp.cla()
            
                # coronal(capture)
                mp.figure(dpi=500)
                mp.axis('off')
                mp.xticks([])
                mp.yticks([])
                #mp.title(parameter)
                mp.imshow(_[z_start: z_end, x_index, y_start: y_end], cmap=cmap)
                # mp.colorbar(fraction=0.02, orientation='horizontal', pad=0)
                #mp.colorbar()
                mp.savefig('fig/%d/%s/slice/capture/%s/%d.png' % (data_index, method, parameter, x_index), bbox_inches='tight')
                mp.cla()

                # sagittal
                generate_folder('fig/%d/%s/%s/sagittal/%d' % (data_index, method, str(delay_correct), parameter_index))
                for y_index in range(y_size):
                    mp.figure(dpi=500)
                    mp.axis('off')
                    mp.xticks([])
                    mp.yticks([])
                    #mp.title(parameter)
                    mp.imshow(_[:, :, y_index], cmap=cmap)
                    #mp.colorbar()
                    mp.savefig('fig/%d/%s/%s/sagittal/%d/%d.png' % (data_index, method, str(delay_correct), parameter_index, y_index), bbox_inches='tight')
                    mp.cla()

                # slice
                for x_index in range(x_size):
                    if parameter_index == 5:
                        # coronal
                        __ = _[:, x_index, :]
                        # convert_to_nan(__, segment=segment[:, x_index, :])  # 如果需要显示跌破lowbound的异常值，即0，则打开这行代码
                        mp.figure(dpi=500)
                        # mp.axis('off')
                        # mp.xticks([])
                        # mp.yticks([])
                        mp.title(parameter)
                        mp.imshow(__, cmap=cmap)
                        mp.colorbar()
                        mp.savefig('fig/%d/coronal/%s_%d(%d).png' % (data_index, name, parameter_index, x_index), bbox_inches='tight')  # , transparent=True
                        mp.cla()

                for y_index in range(y_size):
                    if parameter_index == 5:
                        # sagittal
                        mp.figure(dpi=500)
                        #mp.axis('off')
                        #mp.xticks([])
                        #mp.yticks([])
                        mp.title(parameter)
                        mp.imshow(_[:, :, y_index], cmap=cmap)
                        mp.colorbar()
                        mp.savefig('fig/%d/sagittal/%s_%d(%d).png' % (data_index, name, parameter_index, y_index), bbox_inches='tight')
                        mp.cla()
                '''