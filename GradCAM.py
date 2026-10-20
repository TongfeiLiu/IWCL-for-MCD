import torch
import numpy as np
import cv2
from Glouster2 import Wave_Con_Net

class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: 要解释的模型
        target_layer: 目标层 (nn.Module)，比如 model.layer4[-1].conv3
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None, raw_cam=False, original_image=None, colormap=cv2.COLORMAP_JET):
        """
        input_tensor: [B, C, H, W]
        class_idx: 目标类别 (int)，None 表示自动选择预测类别
        raw_cam: 是否只返回归一化的 CAM (二维 array)
        original_image: 原始图 (numpy [H,W,3], RGB 或 BGR 都可)
        colormap: 热力图颜色 (默认 JET)

        return:
            - 如果 raw_cam=True: 仅返回归一化 CAM (numpy, HxW, 0~1)
            - 如果 raw_cam=False 且 original_image=None: 返回彩色热力图 (HxWx3, BGR)
            - 如果 raw_cam=False 且 original_image!=None: 返回叠加原图的结果
        """
        # 清空缓存
        self.gradients = []
        self.activations = []

        # 前向
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 反向
        self.model.zero_grad()
        output[0, class_idx].backward()

        # 取激活和梯度
        grads = self.gradients[0].cpu().data.numpy()[0]     # [C, H, W]
        acts = self.activations[0].cpu().data.numpy()[0]    # [C, H, W]

        # GAP 权重
        weights = grads.mean(axis=(1, 2))   # [C]

        # 加权求和
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ReLU + 归一化
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam

        if raw_cam:
            return cam

        # 转成彩色热力图
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam, (W, H))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)

        if original_image is None:
            return heatmap
        else:
            # 调整原图大小
            orig_resized = cv2.resize(original_image, (W, H))
            if orig_resized.max() <= 1.0:  # 如果是 [0,1] 归一化的图像
                orig_resized = np.uint8(orig_resized * 255)
            result = cv2.addWeighted(orig_resized, 0.6, heatmap, 0.4, 0)
            return result

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import torch
    from torchvision import transforms, models

    # 1. 加载原图
    img = Image.open("/opt/data/private/xj/my_experience/sence_MCD/data/dataset/T2.png").convert("RGB")
    W, H = img.size  # 原图尺寸

    # 2. 预处理 (不改变尺寸，只ToTensor和归一化)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0)

    # 3. 初始化 GradCAM
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    target_layer = model.layer4[-1].conv3
    gradcam = GradCAM(model, target_layer)

    # 4. 得到叠加结果 (保持原图大小)
    original_image = np.array(img)[:, :, ::-1]  # RGB->BGR
    overlay = gradcam.generate(input_tensor, original_image=original_image)

    # 5. 保存结果
    cv2.imwrite("gradcam_result.jpg", overlay)
    print("✅ Grad-CAM 结果已保存到 gradcam_result.jpg")


