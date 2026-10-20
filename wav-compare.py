import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# === 读取三张灰度图 ===
change_map = cv2.imread("/opt/data/private/xj/my_experience/sence_MCD/data/dataset/GT.png", cv2.IMREAD_GRAYSCALE)
diff_raw_low = cv2.imread("/opt/data/private/xj/my_experience/sence_MCD/dawn-orig/CM_iter_0.png", cv2.IMREAD_GRAYSCALE)
diff_common = cv2.imread("/opt/data/private/xj/my_experience/sence_MCD/dawn_0/CM_iter_0.png", cv2.IMREAD_GRAYSCALE)

# 检查读取是否成功
if change_map is None or diff_raw_low is None or diff_common is None:
    raise FileNotFoundError("有图像路径错误，请检查文件路径是否正确")

# 如果尺寸不一致，裁剪到相同大小
min_h = min(change_map.shape[0], diff_raw_low.shape[0], diff_common.shape[0])
min_w = min(change_map.shape[1], diff_raw_low.shape[1], diff_common.shape[1])
change_map = change_map[:min_h, :min_w]
diff_raw_low = diff_raw_low[:min_h, :min_w]
diff_common = diff_common[:min_h, :min_w]

# 归一化到 [0,1]
change_map = change_map.astype(np.float32) / 255.0
diff_raw_low = diff_raw_low.astype(np.float32) / 255.0
diff_common = diff_common.astype(np.float32) / 255.0

# === 计算 SSIM ===
corr_raw = ssim(change_map, diff_raw_low,
                data_range=1.0,
                win_size=11,
                gaussian_weights=True)

corr_common = ssim(change_map, diff_common,
                   data_range=1.0,
                   win_size=11,
                   gaussian_weights=True)

print(f"SSIM(change_map vs diff_raw_low) = {corr_raw:.4f}")
print(f"SSIM(change_map vs diff_common)  = {corr_common:.4f}")

