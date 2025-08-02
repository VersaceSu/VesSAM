import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from Models.image_encoder import ImageEncoderViT
from Models.mask_decoder import MaskDecoder
from Models.Prompt_encoder import PromptEncoder


class InputAdapter(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

class Ves_SAM(nn.Module):
    # 设置一些默认的常量
    mask_threshold: float = 0.0  # 掩膜阈值，默认为0.0
    image_format: str = "RGB"  # 图像的格式，默认为 RGB

    def __init__(
            self,
            image_encoder: ImageEncoderViT,  # 图像编码器（ViT）
            prompt_encoder: PromptEncoder,  # 提示编码器
            mask_decoder: MaskDecoder,  # 掩膜解码器
            pixel_mean: List[float] = [123.675, 116.28, 103.53],  # 输入图像的均值，用于标准化
            pixel_std: List[float] = [58.395, 57.12, 57.375],  # 输入图像的标准差，用于标准化
            # input_adapter: InputAdapter = InputAdapter(target_size=1024),
    ) -> None:
        """
        SAM 用于从图像和输入的提示信息预测物体掩膜。

        参数:
          image_encoder (ImageEncoderViT): 用于将图像编码为图像嵌入的骨干网络。
          prompt_encoder (PromptEncoder): 编码各种类型的输入提示（例如点、框、掩膜等）。
          mask_decoder (MaskDecoder): 从图像嵌入和编码后的提示中预测掩膜。
          pixel_mean (list(float)): 用于标准化图像的均值。
          pixel_std (list(float)): 用于标准化图像的标准差。
        """
        super().__init__()
        self.image_encoder = image_encoder  # 图像编码器
        self.prompt_encoder = prompt_encoder  # 提示编码器
        self.mask_decoder = mask_decoder  # 掩膜解码器
        # 将均值和标准差注册为 buffer，不会被梯度计算影响
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # self.input_adapter = input_adapter

    @property
    def device(self) -> Any:
        """返回模型所在的设备（CPU或GPU）"""
        return self.pixel_mean.device

    @torch.no_grad()  # 在推理过程中不计算梯度
    def forward(
            self,
            batched_input: List[Dict[str, Any]],  # 输入图像的批次，每个图像是一个字典
            multimask_output: bool,  # 是否需要输出多个掩膜（针对不同的目标）
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从提供的图像和提示信息预测掩膜。

        如果提示信息不事先知道，建议使用 SamPredictor 类，而不是直接调用模型。

        参数:
          batched_input (list(dict)): 输入图像的批次，每个元素是一个字典，包含以下键:
              'image': 输入图像，形状为 3xHxW，已经经过模型输入要求的预处理。
              'original_size': 图像原始大小，格式为 (H, W)。
              'point_coords': 点提示，形状为 BxNx2，已经变换到输入框架。
              'point_labels': 点提示标签，形状为 BxN。
              'boxes': 框提示，形状为 Bx4，已经变换到输入框架。
              'mask_inputs': 掩膜输入，形状为 Bx1xHxW。
          multimask_output (bool): 是否预测多个掩膜，还是返回单个掩膜。

        返回:
          (list(dict)): 返回每个输入图像的掩膜预测结果，字典中包含以下键：
              'masks': 预测的二值掩膜，形状为 BxCxHxW，其中 B 是输入批次的大小，
                      C 是掩膜数量（根据 multimask_output），(H, W) 是图像原始大小。
              'iou_predictions': 模型对掩膜质量的预测，形状为 BxC。
              'low_res_logits': 低分辨率的 logits，形状为 BxCxHxW，其中 H=W=256。
        """
        # 对输入图像进行预处理和标准化
        # input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        print(batched_input["image"].shape)
        print(batched_input["image"].requires_grad)

        input_images = batched_input["image"]
        mask = batched_input['mask']
        skeleton = batched_input['skeleton']
        branch_points = batched_input['branch_points']
        mid_points = batched_input['mid_points']

        # 使用图像编码器提取图像嵌入
        image_embeddings = self.image_encoder(input_images)

        # outputs = []  # 用于存储输出的掩膜预测结果
        # for image_record, curr_embedding in zip(batched_input, image_embeddings):
        #     # 获取当前图像的点坐标和点标签（如果有的话）
        #     if "point_coords" in image_record:
        #         points = (image_record["point_coords"], image_record["point_labels"])
        #     else:
        #         points = None
        # 使用提示编码器处理点提示、框提示、掩膜提示等
        sparse_embeddings, dense_embeddings, graph_embeddings = self.prompt_encoder(
            branch_points=branch_points,
            mid_points=mid_points,
            skeleton=skeleton,
            masks=mask,
            # image_record.get("mask_inputs", None),
        )
        # 使用掩膜解码器进行掩膜预测
        masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            graph_embeddings=graph_embeddings,
            multimask_output=multimask_output,
            # image_embeddings=curr_embedding.unsqueeze(0),
            # image_pe=self.prompt_encoder.get_dense_pe(),
            # sparse_prompt_embeddings=sparse_embeddings,
            # dense_prompt_embeddings=dense_embeddings,
            # multimask_output=multimask_output,
        )

        # 后处理：去除填充并将掩膜缩放回原始图像大小
        # masks = self.postprocess_masks(
        #     low_res_masks,
        #     input_size=image_record["image"].shape[-2:],  # 输入图像的尺寸
        #     original_size=image_record["original_size"],  # 原始图像的尺寸
        # )
        masks = masks > self.mask_threshold  # 根据阈值将掩膜转为二值
        # 将结果添加到输出列表
        outputs = {
            "masks": masks,  # 预测的掩膜
            "iou_predictions": iou_predictions,  # 掩膜质量的预测
            # "low_res_logits": low_res_masks,  # 低分辨率的 logits
        }
        return masks, iou_predictions

    def postprocess_masks(
            self,
            masks: torch.Tensor,  # 掩膜张量，形状为 BxCxHxW
            input_size: Tuple[int, ...],  # 输入图像的尺寸
            original_size: Tuple[int, ...],  # 原始图像的尺寸
    ) -> torch.Tensor:
        """
        去除填充并将掩膜缩放到原始图像的尺寸。

        参数:
          masks (torch.Tensor): 来自掩膜解码器的掩膜，形状为 BxCxHxW。
          input_size (tuple(int, int)): 输入图像的尺寸，格式为 (H, W)，用于去除填充。
          original_size (tuple(int, int)): 图像的原始尺寸，格式为 (H, W)，用于将掩膜缩放到原始尺寸。

        返回:
          (torch.Tensor): 已经缩放到原始尺寸的掩膜，形状为 BxCxHxW。
        """
        # 将掩膜缩放到图像编码器的图像大小
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # 去除填充的区域
        masks = masks[..., : input_size[0], : input_size[1]]
        # 将掩膜缩放回原始图像的大小
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """标准化输入图像的像素值，并填充到正方形输入尺寸。"""
        # 标准化图像的颜色
        x = (x - self.pixel_mean) / self.pixel_std
        # 填充到正方形
        h, w = x.shape[-2:]  # 获取图像的高度和宽度
        padh = self.image_encoder.img_size - h  # 计算垂直方向的填充
        padw = self.image_encoder.img_size - w  # 计算水平方向的填充
        x = F.pad(x, (0, padw, 0, padh))  # 填充图像
        return x
