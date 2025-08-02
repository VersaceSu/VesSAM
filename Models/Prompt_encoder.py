import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from Dataloader import VesselDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Any, Optional, Tuple, Type
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from Models.common import LayerNorm2d

import random
from Models.Mytransformer import CrossAttentionTransformer, ThreeWayCrossAttentionTransformer


class PositionEmbeddingRandom(nn.Module):
    """
    使用随机空间频率的位置信息编码。
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        # coords = coords @ self.positional_encoding_gaussian_matrix
        coords = coords @ self.positional_encoding_gaussian_matrix.to(torch.float32)
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size

        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]

        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class GraphEmbedding(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 定义图卷积网络的层
        self.conv1 = GCNConv(2, embed_dim)  # 输入特征维度是2（点的坐标），输出维度是embed_dim
        self.conv2 = GCNConv(embed_dim, embed_dim * 2)
        self.conv3 = GCNConv(embed_dim * 2, embed_dim * 4)

    def build_graph(self, points: torch.Tensor) -> Data:
        """
        使用给定的点（分叉点和中点）创建图。图的节点是点本身，边是每两个点之间的欧几里得距离。

        Args:
            points (torch.Tensor): 点坐标，形状为 [batch_size, num_points, 2]

        Returns:
            Data: PyG图数据对象，包含边索引和节点特征。
        """
        num_points = points.shape[0]

        # 将点坐标转换为浮点类型
        points = points.float()

        # 计算点之间的距离，作为边的权重
        dist_matrix = torch.cdist(points, points)  # 计算每两个点的欧几里得距离，形状为 [num_points, num_points]

        # 创建边索引，选取距离小于阈值的点作为边
        # print((dist_matrix < 1.0).nonzero(as_tuple=True))
        edge_index = torch.stack((dist_matrix < 1.0).nonzero(as_tuple=True), dim=0)  # 只选择距离小于1的点作为边
        # print(edge_index.shape)

        # 使用点的坐标作为节点特征（坐标本身作为特征）
        x = points.view(num_points, 2)  # 保持为 [num_points, 2]

        # 转换为PyG的数据对象
        data = Data(x=x, edge_index=edge_index)

        return data

    def build_split_graph(self, branch_points: torch.Tensor, mid_points: torch.Tensor) -> Batch:
        """
        使用批量的分叉点和中点创建图。

        Args:
            branch_points (torch.Tensor): 分叉点坐标，形状为 [batch_size, num_branchpoints, 2]
            mid_points (torch.Tensor): 中点坐标，形状为 [batch_size, num_midpoints, 2]

        Returns:
            Batch: PyG 批量图数据对象
        """
        batch_size, num_branchpoints, _ = branch_points.shape
        _, num_midpoints, _ = mid_points.shape

        data_list = []

        for i in range(batch_size):
            # 提取当前样本的分叉点和中点
            branch_sample = branch_points[i]  # 形状 [num_branchpoints, 2]
            mid_sample = mid_points[i]  # 形状 [num_midpoints, 2]

            # 合并点
            all_points = torch.cat([branch_sample, mid_sample], dim=0)  # [num_points, 2]
            num_points = all_points.shape[0]

            # 计算点之间的欧几里得距离
            dist_matrix = torch.cdist(all_points, all_points)

            # 定义边规则
            branch_idx = torch.arange(num_branchpoints)  # 分叉点索引
            mid_idx = torch.arange(num_branchpoints, num_points)  # 中点索引

            # 分叉点 -> 最近的中点
            branch_to_mid = (dist_matrix[branch_idx][:, mid_idx] < 1.0).nonzero(as_tuple=False)
            # 中点 -> 中点
            mid_to_mid = (dist_matrix[mid_idx][:, mid_idx] < 1.0).nonzero(as_tuple=False)

            # 构建边索引
            edge_index = torch.cat([
                torch.stack((branch_to_mid[:, 0], branch_to_mid[:, 1] + num_branchpoints)),  # 分叉点 -> 中点
                torch.stack((mid_to_mid[:, 0] + num_branchpoints, mid_to_mid[:, 1] + num_branchpoints))  # 中点 -> 中点
            ], dim=1)

            # 节点类型特征
            node_types = torch.cat([
                torch.ones(num_branchpoints, 1),  # 分叉点标记为 1
                torch.zeros(num_midpoints, 1)  # 中点标记为 0
            ], dim=0)
            node_features = torch.cat([all_points, node_types], dim=1)  # [num_points, 3]

            # 构建图数据对象
            data = Data(x=node_features, edge_index=edge_index)
            data_list.append(data)

        # 合并为批量图
        batch_graph = Batch.from_data_list(data_list)
        return batch_graph

    # def forward(self, branch_points: torch.Tensor, mid_points: torch.Tensor) -> torch.Tensor:
    #     """
    #     使用批量的分叉点和中点构建图并进行图卷积处理，和上面是匹配的
    #
    #     Args:
    #         branch_points (torch.Tensor): 分叉点坐标，形状为 [batch_size, num_branchpoints, 2]
    #         mid_points (torch.Tensor): 中点坐标，形状为 [batch_size, num_midpoints, 2]
    #
    #     Returns:
    #         torch.Tensor: 图嵌入结果，形状为 [batch_size, num_points, embed_dim * 4]
    #     """
    #     # 构建批量图
    #     batch_graph = self.build_graph(branch_points, mid_points)
    #
    #     # 图卷积处理
    #     x = self.conv1(batch_graph.x, batch_graph.edge_index)
    #     x = F.relu(x)
    #     x = self.conv2(x, batch_graph.edge_index)
    #     x = F.relu(x)
    #     x = self.conv3(x, batch_graph.edge_index)
    #
    #     # 按批量拆分结果
    #     node_embeddings = x.split(batch_graph.batch.bincount().tolist(), dim=0)
    #     return torch.stack(node_embeddings, dim=0)  # [batch_size, num_points, embed_dim * 4]

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        构建图并使用图卷积网络对其进行处理。

        Args:
            points (torch.Tensor): 点坐标，形状为 [batch_size, num_points, 2]

        Returns:
            torch.Tensor: 图嵌入结果
        """
        batch_size, num_points, _ = points.shape

        # 为每个样本单独构建图
        graph_embeddings = []
        for i in range(batch_size):
            # 提取每个样本的点
            sample_points = points[i]  # 形状 [num_points, 2]
            # print(sample_points)
            # 构建图
            data = self.build_graph(sample_points)

            # 图卷积处理
            # print(data.x.type)
            # print(data.edge_index.type)
            x = self.conv1(data.x, data.edge_index)  # 输入是坐标（2维）
            x = F.relu(x)
            x = self.conv2(x, data.edge_index)
            x = F.relu(x)
            x = self.conv3(x, data.edge_index)

            # 将每个样本的图嵌入添加到结果列表
            graph_embeddings.append(x)

        # 合并所有样本的图嵌入
        graph_embeddings = torch.stack(graph_embeddings, dim=0)  # 形状 [batch_size, num_points, embed_dim]

        return graph_embeddings


class PromptEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # image_embedding_size: Tuple[int, int],
                 img_size: int,
                 base_chans: int,
                 activation: Type[nn.Module] = nn.GELU,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 graph_layers: int = 2,
                 point_emb_chans: int = 256
                 ) -> None:
        """
        编码血管分割任务的点提示（分叉点、中点）和骨架信息。

        Arguments:
          embed_dim (int): 提示的嵌入维度
          img_size (int): 输入图像的大小，
          base_chans (int): 输入掩膜的通道数
          activation (nn.Module): 用于激活输入掩膜的激活函数，默认为 GELU
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        # self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(point_emb_chans // 2)
        # 为分叉点和中点设计独立的嵌入层
        self.branch_point_embed = nn.Embedding(16, point_emb_chans)
        self.mid_point_embed = nn.Embedding(16, point_emb_chans)
        self.no_mask_embed = nn.Embedding(1, embed_dim * 2)
        self.graph_layers = graph_layers
        self.point_emb_chans = point_emb_chans

        # 骨架输入的嵌入层
        self.scale_skeleton = 4  # 这个比例是和skeleton_downscaling所对应的
        self.down_skeleton_dim = base_chans // self.scale_skeleton
        self.skeleton_downscaling = nn.Sequential(
            nn.Conv2d(1, base_chans, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(base_chans),
            activation(),
            nn.Conv2d(base_chans, base_chans * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(base_chans * 2),
            activation(),
            nn.Conv2d(base_chans * 2, base_chans * 4, kernel_size=3, stride=2,
                      padding=1),
            # LayerNorm2d(base_chans * 4),
            # activation(),
            # nn.Conv2d(base_chans * 4, base_chans * 4, kernel_size=1),
        )
        # mask输入的嵌入层
        self.scale_mask = 4  # 这个比例是和mask_downscaling所对应的
        self.down_mask_dim = base_chans // self.scale_mask
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, base_chans, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(base_chans),
            activation(),
            nn.Conv2d(base_chans, base_chans * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(base_chans * 2),
            activation(),
            nn.Conv2d(base_chans * 2, base_chans * 4, kernel_size=3, stride=2,
                      padding=1),
        )
        self.cross_attention = CrossAttentionTransformer(self.point_emb_chans, num_heads, num_layers)
        self.graph_embedding = GraphEmbedding(embed_dim, num_layers)
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )
        self.three_cross_transformer = ThreeWayCrossAttentionTransformer(embed_dim=256, num_heads=8, num_layers=4)

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """嵌入掩膜图像"""
        return self.mask_downscaling(masks)

    def _embed_skeletons(self, skeletons: torch.Tensor) -> torch.Tensor:
        """嵌入骨架图像"""
        return self.skeleton_downscaling(skeletons)

    def get_dense_pe(self) -> torch.Tensor:
        """
        返回密集位置编码，适用于编码点提示。
        """
        return self.pe_layer((self.img_size // 8, self.img_size // 8)).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        嵌入点提示（分叉点和中点），同时考虑点的类型（分叉点、中点）。

        Arguments:
          points (torch.Tensor): 点坐标，形状为 [batch_size, num_points, 2]
          labels (torch.Tensor): 点的标签，形状为 [batch_size, num_points]
        """
        point_embeddings = self.pe_layer.forward_with_coords(points, (self.img_size, self.img_size))

        # 将分叉点和中点的嵌入扩展到正确的形状
        branch_point_embed_weight = self.branch_point_embed.weight[None, :, :].repeat(point_embeddings.shape[0], 1, 1)
        mid_point_embed_weight = self.mid_point_embed.weight[None, :, :].repeat(point_embeddings.shape[0], 1, 1)
        # 根据标签选择不同的嵌入层（分叉点 vs 中点）
        point_embeddings[labels == 0] += branch_point_embed_weight[labels == 0]  # 分叉点
        point_embeddings[labels == 1] += mid_point_embed_weight[labels == 1]  # 中点
        return point_embeddings

    def _get_device(self) -> torch.device:
        return self.branch_point_embed.weight.device

    def _get_batch_size(
            self,
            branch_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            mid_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            skeloton: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        # if branch_points is not None:
        #     return branch_points.shape[0]
        # elif mid_points is not None:
        #     return mid_points.shape[0]
        # elif skeloton is not None:
        #     return skeloton.shape[0]
        # elif masks is not None:
        #     return masks.shape[0]
        # else:
        #     return 1
        return next((x.shape[0] for x in [branch_points, mid_points, skeloton, masks] if x is not None), 1)

    def forward(self,
                branch_points: Optional[torch.Tensor] = None,
                mid_points: Optional[torch.Tensor] = None,
                skeleton: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None,
                mode_probs: Optional[dict] = None,
                shuffle_multimodal: Optional[bool] = True,
                prompt_flag: Optional[list] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入的分叉点、中点、骨架和掩膜进行嵌入，生成稀疏嵌入和密集嵌入。

        Arguments:
            branch_points (torch.Tensor, optional): 分叉点坐标，形状为 [batch_size, num_points, 2]
            mid_points (torch.Tensor, optional): 中点坐标，形状为 [batch_size, num_points, 2]
            skeleton (torch.Tensor, optional): 骨架图像，形状为 [batch_size, 1, H, W]
            masks (torch.Tensor, optional): 掩膜图像，形状为 [batch_size, 1, H, W]
            mode_probs: Optional[dict] = None,
            shuffle_multimodal (torch.bool,optional): 对是否shuffle的标签，默认为True也就是训练状态，如果为False就是测试
            prompt_flag (list,optional): 从外部传送多种Prompt的使用情况，默认为None，就是训练状态，如果不为空，就是在测试

        Returns:
            torch.Tensor: 稀疏嵌入（点和骨架），形状为 [batch_size, N, embed_dim]
            torch.Tensor: 密集嵌入（掩膜），形状为 [batch_size, embed_dim, H, W]
        """

        # 获取批量大小
        bs = self._get_batch_size(branch_points, mid_points, skeleton, masks)

        if shuffle_multimodal:
            mode_probs = mode_probs or {'branch': 0.5, 'mid': 0.5, 'skeleton': 0.5, 'mask': 0.1}

            if random.random() > mode_probs['branch']: branch_points = None
            if random.random() > mode_probs['mid']: mid_points = None
            if random.random() > mode_probs['skeleton']: skeleton = None
            if random.random() > mode_probs['mask']: masks = None
        else:
            if prompt_flag is not None:
                [skeletons_flag, branch_flag, mid_flag, mask_flag] = prompt_flag
                if not skeletons_flag: skeleton = None
                if not branch_flag: branch_points = None
                if not mid_flag: mid_points = None
                if not mask_flag: masks = None

        cross_transfoemer1 = True
        cross_transfoemer2 = True

        # branch_points = None
        # mid_points = None
        # masks = None
        # skeleton = None

        # 对 branch_points 和 mid_points 随机采样

        # *********************************************************************************
        # *********************************************************************************
        # *********************************************************************************

        # 初始化空的稀疏嵌入张量
        sparse_embeddings = torch.empty((bs, 0, self.point_emb_chans), device=self._get_device())  # [B, 0, 256] 空
        graph_embedding = torch.empty((bs, 0, self.point_emb_chans), device=self._get_device())  # [B, 0, 256] 空
        dense_embeddings = torch.empty((bs, 0, self.img_size // 8, self.img_size // 8),
                                       device=self._get_device())  # [B, 0, 256,256] 空

        # 处理分叉点（branch_points）
        if branch_points is not None:
            branch_labels = torch.zeros(branch_points.shape[0], branch_points.shape[1],
                                        device=branch_points.device)  # [batch_size, num_points] 分叉点标签
            branch_embeddings = self._embed_points(branch_points, branch_labels)
            sparse_embeddings = torch.cat([sparse_embeddings, branch_embeddings], dim=1)  # 合并分叉点嵌入

        # 处理中点（mid_points）
        if mid_points is not None:
            mid_labels = torch.ones(mid_points.shape[0], mid_points.shape[1],
                                    device=mid_points.device)  # [batch_size, num_points] 中点标签
            mid_embeddings = self._embed_points(mid_points, mid_labels)
            sparse_embeddings = torch.cat([sparse_embeddings, mid_embeddings], dim=1)  # 合并中点嵌入

        if branch_points is not None or mid_points is not None:
            valid_points = []
            if branch_points is not None: valid_points.append(branch_points)
            if mid_points is not None: valid_points.append(mid_points)
            if valid_points: all_points = torch.cat(valid_points, dim=1)
            graph_embedding = torch.cat([graph_embedding, self.graph_embedding(all_points)], dim=1)  # 图神经网络处理

        target_dim = self.img_size // 8
        # target_dim1 = self.image_embedding_size[1] // 8
        # 处理骨架图像（skeleton）

        # if skeleton is not None:
        #     skeleton_embedding = self._embed_skeletons(skeleton)
        # else:
        #     # 如果没有骨架图像，使用一个空的嵌入（没有骨架的情况）
        #     skeleton_embedding = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, target_dim0, target_dim1)
        skeleton_embedding = self._embed_skeletons(skeleton) if skeleton is not None else (
            self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, target_dim, target_dim))
        dense_embeddings = torch.cat([dense_embeddings, skeleton_embedding], dim=1)  # 合并skeleton嵌入

        # 处理掩膜（masks）

        # if masks is not None:
        #     mask_embeddings = self._embed_masks(masks)  # 对掩膜进行嵌入
        # else:
        #     # 如果没有掩膜，仍然使用默认的无掩膜嵌入
        #     mask_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, target_dim0, target_dim1)
        mask_embedding = self._embed_masks(masks) if masks is not None else (
            self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, target_dim, target_dim))
        dense_embeddings = torch.cat([dense_embeddings, mask_embedding], dim=1)  # 合并mask嵌入


        # **********判断sprase和dense是否都为空，需不需要进行cross_transformer**********
        if sparse_embeddings.shape[1] > 0 and dense_embeddings.shape[1] > 0 and cross_transfoemer1:
            sparse_embeddings, dense_embeddings = self.cross_attention(dense_embeddings, sparse_embeddings)
            dense_embeddings = dense_embeddings.reshape(bs, self.img_size // 8, self.img_size // 8, -1).permute(0, 3, 1, 2)
        # else:
        #     print("Skipping cross_attention for sparse_embeddings or dense_embeddings due to empty input.")

        # 判断是否需要进行图形嵌入的cross_attention
        if dense_embeddings.shape[1] > 0 and graph_embedding.shape[1] > 0 and cross_transfoemer2:
            graph_embedding, dense_embeddings = self.cross_attention(dense_embeddings, graph_embedding)
            dense_embeddings = dense_embeddings.reshape(bs, self.img_size // 8, self.img_size // 8, -1).permute(0, 3, 1, 2)
        # else:
        #     print("Skipping cross_attention for graph_embedding due to empty input.")

        return sparse_embeddings, dense_embeddings, graph_embedding


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset_dir = 'data'
    dataset_name = 'Aorta'  # Example dataset
    split = 'train'  # Use 'train', 'val', or 'test' split

    dataset = VesselDataset(dataset_name=dataset_name, data_dir=dataset_dir, split=split, transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 设置参数
    embed_dim = 64  # 嵌入维度
    image_embedding_size = (256, 256)  # 图像嵌入的空间大小 (H, W)
    input_image_size = (256, 256)  # 输入图像的大小 (H, W)
    mask_in_chans = 32  # 输入掩膜的通道数

    model = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=input_image_size,
        base_chans=mask_in_chans
    )

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}")

            image = batch['image']  # Shape: [batch_size, channels, height, width]
            mask = batch['mask']  # Shape: [batch_size, 1, height, width]
            skeleton = batch['skeleton']  # Shape: [batch_size, 1, height, width]
            branch_points = batch['branch_points']  # Shape: [batch_size, max_branch_points, 2]
            mid_points = batch['mid_points']  # Shape: [batch_size, max_mid_points, 2]

            sparse_embeddings, skeleton_embedding, graph_embedding = model(branch_points, mid_points, skeleton, mask)

            # 打印输出结果
            print(
                f"sparse_embeddings: {sparse_embeddings.shape}")  # 应该是 [batch_size, num_points*2, embed_dim][4, 40, 64]
            print(
                f"skeleton_embedding: {skeleton_embedding.shape}")  # 应该是 [batch_size, embed_dim, 64, 64][4, 128, 64, 64]
            print(f"graph_embedding: {graph_embedding.shape}")  # 应该是 [batch_size, num_points*2, embed_dim][4, 40, 64]

            break
