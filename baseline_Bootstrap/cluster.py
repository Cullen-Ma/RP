import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
import SimpleITK as sitk
from matplotlib.colors import ListedColormap, to_rgba
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##change!!
img = np.load('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1_ks3/img_cut_slicer.npy')
itk_img=sitk.ReadImage('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1/seg_pkc_1.nii')
seg = sitk.GetArrayFromImage(itk_img)
seg = seg[85,:,:]
frame = 86
z,y = np.where(seg == 1)
time_curves = img[:, z,y]
time_curves =time_curves.T
df = pd.DataFrame(time_curves)
df = df.fillna(method='ffill').fillna(method='bfill')
time_curves = df.values
cluster_kind = 12
kmeans = KMeans(n_clusters=cluster_kind, max_iter = 600,random_state=0)
kmeans.fit(time_curves)
cluster_labels = kmeans.labels_
distinct_elements = set(cluster_labels)
print(f'distinct_elements={cluster_labels.shape}')
np.save('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1_ks3/slice62_cluster60/label_cluster_tac_kmeans_12.npy',cluster_labels)
seg[z,y] = cluster_labels+1
unique_labels = np.unique(seg)
unique_labels = unique_labels.astype(int)
plt.rcParams.update({'font.size': 12})

custom_labels = ['D{}'.format(label) for label in unique_labels]
slices = cluster_labels
avg = np.zeros((cluster_kind,frame))
for i in range(cluster_kind):
    # 找到值为1的位置
    positions = np.where(slices == i)[0]  # 只取行索引，假设您只关心行位置
    tac = np.zeros((len(positions), frame))  # 使用 positions 的长度（行数）来初始化 tac 数组
    for j, pos in enumerate(positions):
        tac[j] = time_curves[pos, :]  # 使用 pos（行索引）来访问 time_curves 并赋值给 tac
    avg[i, :] = np.mean(tac, axis=0)  # 计算这些位置的平均值
print(avg)
np.save('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1_ks3/slice62_cluster60/avg_tac_kmeans_12.npy',avg)
    
