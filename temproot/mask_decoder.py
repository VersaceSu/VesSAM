import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from Models.common import LayerNorm2d
from Models.Mytransformer import TwoWayTransformer


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        修改后的 MaskDecoder，新增了 graph_embedding 输入。

        Arguments:
          transformer_dim (int): transformer 的通道维度
          transformer (nn.Module): 用于预测 mask 的 transformer
          num_multimask_outputs (int): 预测多张 mask 的数量
          activation (nn.Module): 上采样时使用的激活函数
          iou_head_depth (int): 用于预测 mask 质量的 MLP 层数
          iou_head_hidden_dim (int): 用于预测 mask 质量的 MLP 隐藏层维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # 定义 IOU token 和 Mask token
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 用于上采样的卷积层
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # 用于不同 Mask tokens 的 Hypernetworks MLP
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # IOU 预测头
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,  # [B, 256, H, W]
            image_pe: torch.Tensor,  # [1, 256, H, W]
            sparse_prompt_embeddings: torch.Tensor,  # [B, 32, H2]
            dense_prompt_embeddings: torch.Tensor,  # [B, 256, H, W]
            graph_embeddings: torch.Tensor,  # [B, 32, H2]
            multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过图像和提示嵌入预测 mask。

        Arguments:
          image_embeddings (torch.Tensor): 图像编码器的嵌入
          image_pe (torch.Tensor): 图像的位置信息
          sparse_prompt_embeddings (torch.Tensor): 点和框的嵌入
          dense_prompt_embeddings (torch.Tensor): 输入的 mask 嵌入
          graph_embeddings (torch.Tensor): 图嵌入
          multimask_output (bool): 是否返回多张 mask

        Returns:
          torch.Tensor: 预测的 mask
          torch.Tensor: 预测的 mask 质量
        """

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            graph_embeddings=graph_embeddings,
        )

        # 选择返回的 mask
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # 返回结果
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            graph_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测 mask。
        """

        # 生成输出 token，IOU token 和 mask token
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight],
                                  dim=0)  # iou_token:[1,256]  mask_tokens:[4,256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings, graph_embeddings), dim=1)

        # 处理图像和提示嵌入
        src = image_embeddings
        src = src.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        src = src + dense_prompt_embeddings  # 添加 dense 嵌入
        image_pe = image_pe.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 使用 transformer 进行计算
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # 上采样 mask 嵌入
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, 4, 32]

        b, c, h, w = upscaled_embedding.shape  # [B, 32, 256, 256]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 预测 mask 质量
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# MLP定义（用于生成mask的预测）
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


if __name__ == '__main__':
    transformer_dim = 256  # 假设 transformer_dim 为256
    depth = 4
    embedding_dim = 256
    num_heads = 8
    mlp_dim = 2048
    transformer = TwoWayTransformer(
        depth=depth,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim
    )

    # 初始化 MaskDecoder
    mask_decoder = MaskDecoder(
        transformer_dim=transformer_dim,
        transformer=transformer,
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    # 创建输入张量
    batch_size = 4
    image_embedding = torch.randn(batch_size, 256, 16, 16)  # 图像嵌入
    image_pe = torch.randn(1, 256, 16, 16)  # 图像位置编码
    sparse_embeddings = torch.randn(batch_size, 32, 256)  # 稀疏嵌入
    dense_embeddings = torch.randn(batch_size, 256, 64, 64)  # 密集嵌入（mask嵌入）
    graph_embeddings = torch.randn(batch_size, 32, 256)  # 图嵌入

    # 调用 MaskDecoder 的 forward 方法
    masks, iou_pred = mask_decoder(
        image_embeddings=image_embedding,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        graph_embeddings=graph_embeddings,
        multimask_output=True,
    )

    # 输出 masks 和 iou_pred 的维度
    print(f"masks shape: {masks.shape}")  # 预计输出维度：[batch_size, num_mask_tokens, height, width]
    print(f"iou_pred shape: {iou_pred.shape}")  # 预计输出维度：[batch_size, num_mask_tokens]