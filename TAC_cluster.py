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
#img = img[:,50:250,:]#GUC0.0.0
'''z = np.load('z.npy')
y = np.load('y.npy')
for z1, y1 in zip(z, y):
    img[:, z1, y1] = img[:, 91,87]
np.save('process_whole_body/img_slicer_cut.npy',img)'''
#downsampled_img = downsample_3d_array_slicing(img, 2, 2)
##change!!!!
itk_img=sitk.ReadImage('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1/seg_pkc_1.nii')
seg = sitk.GetArrayFromImage(itk_img)
seg = seg[85,:,:]
#np.save('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1/seg_pkc_1.npy',seg)
#pdb.set_trace()
#seg_copy = seg.copy()
#seg = np.load('/share/home/kma/metabolism/data_guc_cut/seg_mask/seg_guc100.npy')
#seg = np.load('/share/home/kma/metabolism/process_data/segment_slicer_cut.npy')
#plt.figure(figsize=(10, 8))
#plt.imshow(seg)
#plt.savefig("/share/home/kma/metabolism/data_guc_cut/plot1.png", dpi=300, bbox_inches='tight')
#seg = seg[::2, ::2]
frame = 86
z,y = np.where(seg == 1)
time_curves = img[:, z,y]
#time_curves = np.load('cut_data/muscle_noise_0.05_cut.npy')
time_curves =time_curves.T
df = pd.DataFrame(time_curves)
df = df.fillna(method='ffill').fillna(method='bfill')
time_curves = df.values
cluster_kind = 12
print(f'time_curves_len = {len(time_curves[:,1])}')
'''dist_matrix = pairwise_distances(time_curves)
average_distance = dist_matrix.mean()
eps = average_distance'''
#标准化
#scaler = StandardScaler()
#time_curves = scaler.fit_transform(time_curves)
kmeans = KMeans(n_clusters=cluster_kind, max_iter = 600,random_state=0)
#kmeans = GaussianMixture(n_components = 20)
kmeans.fit(time_curves)
cluster_labels = kmeans.labels_
'''index = np.where(cluster_labels==4)
tac_index2 = time_curves[index,:]
x = np.load('process_whole_body/t_midpoint.npy')
for i in range(39):
    plt.plot(x, tac_index2[0,i,:], label=f'Line {i + 1}')
plt.savefig('fig/avg.png')
'''
distinct_elements = set(cluster_labels)
print(f'distinct_elements={cluster_labels.shape}')
pdb.set_trace()

# 获取 Set3 颜色映射
set3_cmap = plt.cm.get_cmap('Set3', cluster_kind)

# 创建一个颜色列表，初始化为空
colors = []

# 如果需要的话，添加白色作为第一个颜色
if len(colors) == 0:  # 确保只在开始时添加一次白色
    colors.append(to_rgba('white'))

# 深化 Set3 颜色映射中的颜色，并添加到颜色列表中
for i in range(set3_cmap.N):
    # 获取原始颜色（RGB值）
    original_color = set3_cmap(i)
    
    # 将颜色变深，可以通过增加 RGB 分量中的最小值或者直接乘以一个小于1但大于0的系数
    # 这里我们简单地将每个分量乘以 0.8 来加深颜色
    darkened_color = tuple(min(1, c * 0.85) for c in original_color[:3]) + (original_color[3],)  # 保持alpha不变
    
    # 添加到颜色列表中
    colors.append(darkened_color)

# 创建自定义的 ListedColormap
custom_cmap = ListedColormap(colors)
#pdb.set_trace()

np.save('/share/home/kma/metabolism/data_bj_nodule_hospital/additional1_ks3/slice62_cluster60/label_cluster_tac_kmeans_12.npy',cluster_labels)
seg[z,y] = cluster_labels+1
unique_labels = np.unique(seg)
unique_labels = unique_labels.astype(int)


plt.rcParams.update({'font.size': 12})

custom_labels = ['D{}'.format(label) for label in unique_labels]


# 绘制图像
plt.figure(figsize=(10, 8))
plt.imshow(seg, cmap=custom_cmap)
cbar = plt.colorbar(ticks=unique_labels)  # 添加标签 "Categories" 作为例子
cbar.ax.set_yticklabels(custom_labels)  # 设置自定义的刻度标签

plt.savefig("/share/home/kma/metabolism/data_bj_nodule_hospital/additional1_ks3/plot1.png", dpi=300, bbox_inches='tight')
#seg_copy[seg==3] = 0
#np.save('/share/home/kma/metabolism/data_guc_cut/seg_mask/seg_guc100.npy',seg_copy)
'''for i in range(1,cluster_kind+1):
    z1,y1 = np.where(seg==i)
    seg[z1,y1] = 100
    plt.figure(figsize=(10, 8))
    plt.imshow(seg, cmap='hot')
    plt.colorbar() 
    plt.savefig(f"plot_{i}.png")
    seg[z,y] = cluster_labels+1'''
'''pca = PCA(n_components =5)
pca_tac = pca.fit_transform(time_curves)

tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=500)
tsne_results = tsne.fit_transform(pca_tac)
# 输出聚类标签
#print(kmeans.labels_)

#########################
# 创建一个颜色映射
#colors = plt.cm.get_cmap('twilight', cluster_kind)  # 选择适合20个类别的颜色映射
# 绘制散点图
colors = np.array(colors)
plt.figure(figsize=(10, 8))
for i in range(cluster_kind):  # 假设有20个不同的簇
    mask = cluster_labels == i
    plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                c=[colors[i+1]], label=f'D {i+1}')

plt.legend(fontsize = 15)
plt.title('t-SNE visualization of clustering TACs using Kmeans')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.savefig('/share/home/kma/metabolism/data_guc_cut/tsne2.png', dpi=300, bbox_inches='tight')'''

'''# 使用t-SNE进一步降维到3D
tsne = TSNE(n_components=3, verbose=1, perplexity=50, n_iter=600)
tsne_results_3d = tsne.fit_transform(pca_tac)

# 创建一个颜色映射
colors = plt.cm.get_cmap('hot', cluster_kind)  # 选择适合20个类别的颜色映射

# 绘制3D散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 创建一个颜色序列
color_sequence = np.linspace(0, 1, cluster_kind)

for i in range(cluster_kind):  # 假设有20个不同的簇
    mask = cluster_labels == i
    ax.scatter(tsne_results_3d[mask, 0], tsne_results_3d[mask, 1], tsne_results_3d[mask, 2],
               c=[colors(color_sequence[i])], label=f'Cluster {i}', alpha=0.6)

ax.set_title('3D t-SNE visualization of TAC clusters')
ax.set_xlabel('t-SNE component 1')
ax.set_ylabel('t-SNE component 2')
ax.set_zlabel('t-SNE component 3')
ax.legend()

# 添加网格线
ax.grid(True)

plt.show()

# 保存图像
plt.savefig('hot_3d.png', bbox_inches='tight')'''
########################################
#cluster_labels = kmeans.predict(time_curves)





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
    
