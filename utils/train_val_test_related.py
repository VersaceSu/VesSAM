import random
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False, shuffle_multimodal=True, prompt_flag=None):
    sparse_embeddings, dense_embeddings, graph_embeddings = model.prompt_encoder(
        branch_points=batched_input.get("branch_points", None),
        mid_points=batched_input.get("mid_points", None),
        skeleton=batched_input.get("skeleton", None),
        masks=batched_input.get("mask", None),
        shuffle_multimodal=shuffle_multimodal,
        prompt_flag=prompt_flag
    )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        graph_embeddings=graph_embeddings,
        multimask_output=args.num_multimask_outputs,
    )

    if args.num_multimask_outputs:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions


def evaluate_on_test(args, model, test_loader):
    # 没什么用，是在训练过程中使用的
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        batch, batched_input = random.choice(list(enumerate(test_loader)))
        # for batch, batched_input in enumerate(test_loader):
        batched_input = to_device(batched_input, args.device)

        # 获取分支点、中点、骨架
        branch_points = batched_input['branch_points']
        mid_points = batched_input['mid_points']
        skeleton = batched_input['skeleton']
        image = batched_input['image']
        mask = batched_input['mask']

        # 模型推理
        image_embeddings = model.image_encoder(image)
        predicts, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
        predicts = torch.sigmoid(predicts)  # 如果输出是 logits，需要先经过 sigmoid
        binary_predicts = (predicts > 0.5).float()  # 应用阈值 0.5，将预测值转为二值

        # 可视化
        # i = random.randint(0, len(image))
        for i in range(len(image)):
            visualize_results(image[i], mask[i], branch_points[i], mid_points[i], skeleton[i], binary_predicts[i])


def visualize_results(image, mask, branch_points, mid_points, skeleton, count, predicts=None, save_path="visualizations", file_name="result.png"):
    """
    可视化预测结果和关键点。

    参数:
        image (tensor): 输入图像
        mask (tensor): 真实标签（ground truth）
        branch_points (tensor): 分支点
        mid_points (tensor): 中点
        skeleton (tensor): 骨架
        predicts (tensor, optional): 预测结果
    """
    # 转换为 numpy 数组
    image = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    mask = mask.cpu().numpy().squeeze(0)
    branch_points = branch_points.cpu().numpy()
    mid_points = mid_points.cpu().numpy()
    skeleton = skeleton.cpu().numpy().squeeze(0)  # 去除单通道维度

    if predicts is not None:
        predicts = predicts.cpu().numpy().squeeze(0)

        # 创建一个新的图像，用于叠加显示
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()  # 将二维数组展平，便于索引

        # Image 显示
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Image")

        # Mask 显示
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Ground Truth Mask")

        # Skeleton, Branch_Points, Mid_Points 显示
        axs[2].imshow(image, cmap='gray')
        axs[2].imshow(skeleton, cmap='Blues', alpha=0.5)
        axs[2].scatter(branch_points[:, 1], branch_points[:, 0], color='red', s=10, label='Branch Points')
        axs[2].scatter(mid_points[:, 1], mid_points[:, 0], color='yellow', s=10, label='Mid Points')
        axs[2].set_title("Skeleton, Branch Points, Mid Points")
        axs[2].legend()

        # Predict 显示
        if predicts is not None:
            axs[3].imshow(predicts, cmap='gray')  # 显示为二值图
            axs[3].set_title("Prediction")

        # Mask 和 Prediction 差异区域标记
        if predicts is not None:
            # 创建一个差异图
            diff = np.zeros_like(mask)

            # mask为背景 (0)，predict为前景 (1)
            diff[(mask == 0) & (predicts == 1)] = 1

            # mask为前景 (1)，predict为背景 (0)
            diff[(mask == 1) & (predicts == 0)] = 2

            # 使用不同的颜色标记差异区域
            axs[4].imshow(diff, cmap='coolwarm', alpha=0.7)
            axs[4].set_title("Difference between Mask and Prediction")

        # 最后一个图像显示 Mask 和 Prediction 的叠加结果
        if predicts is not None:
            axs[5].imshow(mask, cmap='gray', alpha=0.5)
            axs[5].imshow(predicts, cmap='gray', alpha=0.5)
            axs[5].set_title("Mask and Prediction Overlay")

        plt.tight_layout()
        save_path = os.path.join('./work_dir', save_path)
        os.makedirs(save_path, exist_ok=True)  # 创建目录（如果不存在）
        save_file = os.path.join(save_path, file_name[:-4] + str(count) + ".png")
        plt.savefig(save_file, dpi=300)
        plt.close(fig)  # 关闭图像，释放内存

        # plt.show()
