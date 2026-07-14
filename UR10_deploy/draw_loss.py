# import matplotlib.pyplot as plt
# import re
# import numpy as np
# import pandas as pd


# def plot_loss_curve(log_file_path):
#     steps = []
#     losses = []

#     with open(log_file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             # 使用正则表达式匹配 step 和 loss
#             loss_match = re.search(r'loss:([0-9.]+)', line)
#             if loss_match:
#                 loss = float(loss_match.group(1))
#                 losses.append(loss)

#     plt.figure(figsize=(10, 6))
#     plt.plot(np.arange(len(losses)), losses, label='Loss', alpha=0.7, linewidth=1)
#     plt.title('Training Loss Curve')
#     plt.xlabel('Step')
#     plt.ylabel('Loss')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# def plot_and_save_comparison(log_file_path, save_path):
#     """
#     绘制并保存滤波前后六维力的对比图像。
#     """
#     steps = []
#     losses = []
#     losses = np.loadtxt(log_file_path)
#     print(losses.shape)
#     timestamps = np.array(range(0, losses.shape[0]))
#     original_data = losses


#     def apply_edge_preserving_filter(force_data, median_window=51, mean_window=29):
#         if force_data is None or len(force_data) == 0:
#             return force_data
#         # 将 numpy 数组转换为 DataFrame 以利用高效的 rolling 算子
#         df = pd.DataFrame(force_data)
#         df_med = df.rolling(window=median_window, center=True, min_periods=1).median()
#         df_smooth = df_med.rolling(window=mean_window, center=True, min_periods=1).mean()
#         return df_smooth.values.astype(np.float32)
#     filtered_data = apply_edge_preserving_filter(losses)


#     ft_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
#     y_units = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']


#     # 计算执行进度百分比 (%)
#     timestamps = np.arange(losses.shape[0])
#     progress = ((timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])) * 100
#     fig, axes = plt.subplots(1, 1, figsize=(12, 18), sharex=True)
#     axis_name = ft_cols
#     axes.plot(progress, original_data, label=f'Original {axis_name}', color='blue', alpha=0.5, linewidth=1.2)
#     axes.plot(progress, filtered_data, label=f'Filtered {axis_name}', color='red', alpha=0.8, linewidth=1.5)
#     axes.set_ylabel(f'{axis_name} ({y_units})')
#     axes.legend(loc='upper right')
#     axes.grid(True, linestyle='--', alpha=0.6)
#     axes.set_xlabel('Execution Progress (%)')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

# # ==========================================
# # 1. 从 txt 文件读取 [N] 维的 Loss 数据
# # ==========================================
# # 假设您的文件名叫 'loss.txt'
# # 如果数据是用逗号隔开的，可以加上参数: delimiter=','
# txt_path = 'loss.txt'

# try:
#     raw_loss = np.loadtxt(txt_path)
#     print(f"成功读取数据，数据维度为: {raw_loss.shape}")
# except Exception as e:
#     print(f"读取文件失败，请检查路径或格式是否正确。错误信息: {e}")
#     # 模拟数据备份（仅用于在没有txt时测试代码，实际运行时请注释掉）
#     raw_loss = 2.5 * np.exp(-np.arange(1000) / 200) + 0.5 + np.random.normal(0, 0.2, 1000)

# N = len(raw_loss)
# steps = np.arange(N)

# # ==========================================
# # 2. 计算平滑后的数据 (Savitzky-Golay 滤波器)
# # ==========================================
# # 【调参建议】：
# # window_length 是平滑窗口的大小，必须是一个正奇数。
# # - 如果您的 N 很大（比如几万），可以将窗口调大（如 101, 201, 501），曲线会更平滑。
# # - 如果您的 N 较小（比如几百），窗口调小（如 11, 21, 31），否则会过度平滑。
# window_size = 51

# # 鲁棒性检查：窗口大小不能超过总数据量 N，且必须是奇数
# if window_size >= N:
#     window_size = N - 1 if (N - 1) % 2 != 0 else N - 2
# if window_size < 3:
#     window_size = 3

# loss_savgol = savgol_filter(raw_loss, window_length=window_size, polyorder=3)

# # ==========================================
# # 3. 论文级可视化配置 (科研美化)
# # ==========================================
# # 设置全局衬线字体 (适合 Times New Roman 论文排版)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams['font.size'] = 14
# plt.rcParams['axes.linewidth'] = 1.5  # 边框线宽

# fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# # 【核心步骤】
# # a. 绘制原始噪声数据（灰色底图，高透明度 alpha=0.2~0.3，不喧宾夺主）
# ax.plot(steps, raw_loss, color='tab:gray', alpha=0.25, linewidth=0.8, label='Raw Loss')

# # b. 绘制平滑后的趋势线（深色主调，加粗 linewidth=2.5，精准穿过噪声中心且无相位滞后）
# ax.plot(steps, loss_savgol, color='tab:red', linewidth=2.5, label='Smoothed Trend')

# # ==========================================
# # 4. 图表细节打磨
# # ==========================================
# ax.set_title('Training Loss Convergence', fontsize=16, fontweight='bold', pad=15)
# ax.set_xlabel('Training Steps', fontsize=14)
# ax.set_ylabel('Loss', fontsize=14)

# # 坐标轴刻度线向内（顶刊常见标准）
# ax.tick_params(axis='both', direction='in', length=6, width=1.5)

# # 添加浅色虚线网格，辅助阅读
# ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# # 优化图例：去除边框，背景透明，防止遮挡曲线
# ax.legend(loc='upper right', frameon=False, fontsize=12)

# # 紧凑布局，自动裁剪多余白边
# plt.tight_layout()

# # ==========================================
# # 5. 保存为论文专用的 PDF 矢量图
# # ==========================================
# # bbox_inches='tight' 确保导出的 PDF 不会截断坐标轴标签，LaTeX 导入该 PDF 时极其清晰
# plt.savefig('loss_curve_paper.pdf', format='pdf', bbox_inches='tight')
# plt.show()


# if __name__ == "__main__":
#     # plot_loss_curve('/home/8TDisk/0527model_decoder/metrics_log.txt')
#     plot_and_save_comparison("/home/k202/openpi/openpi/loss.txt",
#                              "/home/k202/openpi/openpi/loss.png")


#     def get_loss(log_file_path):
#         losses = []
#         with open(log_file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 # 使用正则表达式匹配 step 和 loss
#                 loss_match = re.search(r'loss:([0-9.]+)', line)
#                 if loss_match:
#                     loss = float(loss_match.group(1))
#                     losses.append(loss)
#         losses = np.array(losses)[::1]
#         return losses






import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


txt_path = '/home/k202/openpi/openpi/loss.txt'

try:
    raw_loss = np.loadtxt(txt_path)
    print(f"成功读取数据，数据维度为: {raw_loss.shape}")
except Exception as e:
    print(f"读取文件失败，请检查路径或格式是否正确。错误信息: {e}")
    # 模拟数据备份（仅用于在没有txt时测试代码，实际运行时请注释掉）
    raw_loss = 2.5 * np.exp(-np.arange(1000) / 200) + 0.5 + np.random.normal(0, 0.2, 1000)

N = len(raw_loss)
steps = np.arange(N)

window_size = 31
# 鲁棒性检查：窗口大小不能超过总数据量 N，且必须是奇数
if window_size >= N:
    window_size = N - 1 if (N - 1) % 2 != 0 else N - 2
if window_size < 3:
    window_size = 3

loss_savgol = savgol_filter(raw_loss, window_length=window_size, polyorder=3)


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5  # 边框线宽
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.plot(steps, raw_loss, color='tab:gray', alpha=0.25, linewidth=0.8, label='Raw Loss')
ax.plot(steps, loss_savgol, color='tab:red', linewidth=2.5, label='Smoothed Trend')

ax.set_title('Training Loss Convergence', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Training Steps', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_ylim(0, 0.75)
ax.tick_params(axis='both', direction='in', length=6, width=1.5)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.legend(loc='upper right', frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig('/home/k202/openpi/openpi/loss.png', format='pdf', bbox_inches='tight')
plt.show()
