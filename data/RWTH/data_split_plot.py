""" 绘制三维图 """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import colorsys


font1 = FontProperties(fname="font/simhei.ttf", size=12)


def plot_3D(capacity, zlabel, fig_size, file_name, elev=12., azim=-135, line_width=3):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(projection='3d')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)

    num_colors = 124  # 定义彩虹颜色的数量
    # rainbow_colors = [colorsys.hsv_to_rgb(i / num_colors, 1, 1) for i in range(num_colors)] # 生成彩虹颜色
    rainbow_colors = [colorsys.hsv_to_rgb(i / num_colors, 1, 0.8) for i in range(num_colors)] # 生成彩虹颜色


    for i in range(len(capacity)):
        ax.plot(range(len(capacity[i])), capacity[i], zs=i, zdir='y', linewidth=line_width, alpha=0.8, c=rainbow_colors[i])
        # ax.plot(data_x[i], data_z[i], zs=i, zdir='y', c=colors[i], linewidth=line_width, alpha=alpha)
        # ax.scatter(range(len(capacity[i])), capacity[i], zs=i, zdir='y', alpha=0.8, s=1)   # c='red'

    # 填充三维图像，按电池内周期划分 | Fill the 3D image, divided by the cycle in the battery
    # x1, x2 = np.arange(int(350*0.6)), np.arange(int(350*0.6), 350)  # 循环次数 | Cycle
    # y = np.arange(len(capacity))    # 电池数量 | Number of batteries
    # _xx, _yy = np.meshgrid(x1, y)
    # x1, y1 = _xx.ravel(), _yy.ravel()
    # _xx, _yy = np.meshgrid(x2, y)
    # x2, y2 = _xx.ravel(), _yy.ravel()
    # z1, z2 = np.ones_like(x1)*0.8, np.ones_like(x2)*0.8
    # ax.bar3d(x1, y1, z1, 1, 1, 1.1, shade=False, alpha=0.01, color='#F0FFB8')
    # ax.bar3d(x2, y2, z2, 1, 1, 1.1, shade=False, alpha=0.01, color='#BFDDFF')

    # 填充三维图像，按不同电池单体划分 | Fill the 3D image, divided by different battery monomers
    x = np.arange(0, 350, 1)    # 循环次数 | Cycle
    y1, y2 = np.arange(int(len(capacity)*0.6)), np.arange(int(len(capacity)*0.6), len(capacity))  # 电池数量 | Number of batteries
    _xx, _yy = np.meshgrid(x, y1)
    x1, y1 = _xx.ravel(), _yy.ravel()
    _xx, _yy = np.meshgrid(x, y2)
    x2, y2 = _xx.ravel(), _yy.ravel()
    z1, z2 = np.ones_like(x1)*0.8, np.ones_like(x2)*0.8
    ax.bar3d(x1, y1, z1, 1, 1, 1.1, shade=False, alpha=0.01, color='#F0FFB8')
    ax.bar3d(x2, y2, z2, 1, 1, 1.1, shade=False, alpha=0.01, color='#BFDDFF')


    # z = np.arange(0.8, 1.9, 0.1)   # 电池容量 | Battery capacity
    # ax.plot_surface(x, y, z, alpha=0.5)

    ax.set_xlabel('Cycle', labelpad=6, fontsize=12)   # fontproperties=font1
    ax.set_ylabel('Battery ID', labelpad=6, fontsize=12)
    ax.set_zlabel('Capacity', labelpad=4, fontsize=12) #, labelpad=20
    ax.set_zlim(0.8, 1.9)   # OCV-SOC
    # ax.set_xlim(2.9, 4.1)
    ax.set_ylim(0, len(capacity))
    # ax.view_init(elev=elev, azim=azim)
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

    ax.view_init(elev=17., azim=-38)
    # plt.subplots_adjust(top=0.01, bottom=0.001, right=0.92, left=0.001, hspace=0, wspace=0)
    plt.subplots_adjust(left=0.001, right=0.92, top=0.7, bottom=0.1)
    # plt.margins(1, 1, 1)
    fig.tight_layout()  # 全局整理图片尺寸

    # plt.xlabel('Voltage(V)', fontproperties=font2, labelpad=6, fontsize=fontsize)
    # plt.ylabel('Cycle', fontproperties=font2, labelpad=6, fontsize=fontsize)
    # plt.zlabel(zlabel, fontproperties=font2, labelpad=6, fontsize=fontsize) #, labelpad=20

    plt.savefig('dataset_split1.png', transparent=True, dpi=300)    # tight_layout=True
    # plt.savefig(file_name, bbox_inches='tight', transparent=True, dpi=300)  # tight_layout=True
    # plt.show()


# 读取数据并去除空值
data4fig = []
data = pd.read_csv('data.csv', header=0)
for i in range(data.shape[1]):
  data_temp = data.iloc[:,i].values
  data4fig.append(data_temp[~np.isnan(data_temp)])
print(len(data4fig))
print(type(data4fig[0]))
print(data.iloc[:,0].shape, data4fig[0].shape)

# 绘制图像
plot_3D(data4fig, 'Capacity', (7, 6), 'data4fig.png', elev=12., azim=-135, line_width=1)