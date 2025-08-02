import torch
from torch import Tensor, nn
import math
from typing import Tuple, Type, Any
from Models.common import MLPBlock  # 假设 MLPBlock 是一个自定义的多层感知机模块


class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, dense_embeddings: torch.Tensor, sparse_embeddings: torch.Tensor) -> tuple[
        Tensor | Any, Tensor | Any]:
        """
        Args:
            dense_embeddings: [batch_size, H, W, embed_dim]
            sparse_embeddings: [batch_size, N, embed_dim]
        Returns:
            torch.Tensor: 更新后的 sparse_embeddings
        """
        batch_size, embed_dim, H, W = dense_embeddings.shape
        dense_embeddings = dense_embeddings.view(batch_size, -1, embed_dim * 2)

        for attn_layer in self.attn_layers:
            sparse_embeddings, _ = attn_layer(sparse_embeddings, dense_embeddings, dense_embeddings)
            dense_embeddings, _ = attn_layer(dense_embeddings, sparse_embeddings, sparse_embeddings)

        return sparse_embeddings, dense_embeddings


class ThreeWayCrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        """
        构建一个支持三种输入特征的交叉注意力Transformer

        Args:
            embed_dim: 特征嵌入的维度
            num_heads: 注意力头的数量
            num_layers: Transformer层的数量
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 使用MultiheadAttention来进行三种输入的交叉注意力计算
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(
            self,
            dense_embeddings: torch.Tensor,
            sparse_embeddings: torch.Tensor,
            graph_embeddings: torch.Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        进行三种特征的交叉注意力计算，更新输入的三种特征

        Args:
            dense_embeddings: [batch_size, embed_dim, H, W]  稠密特征
            sparse_embeddings: [batch_size, N, embed_dim]    稀疏特征
            graph_embeddings: [batch_size, M, embed_dim]     图特征

        Returns:
            tuple: 更新后的稀疏特征、稠密特征和图特征
        """
        # 处理dense_embeddings
        batch_size, embed_dim, H, W = dense_embeddings.shape
        dense_embeddings = dense_embeddings.view(batch_size, -1, embed_dim * 2)  # [B, H*W, embed_dim]

        # 处理sparse_embeddings：保持其维度 [B, N, embed_dim]
        sparse_embeddings = sparse_embeddings  # [B, N, embed_dim]

        # 处理graph_embeddings：保持其维度 [B, M, embed_dim]
        graph_embeddings = graph_embeddings  # [B, M, embed_dim]

        # 初始化交叉注意力计算
        for attn_layer in self.attn_layers:
            # 在每一层内同时进行三种特征的交叉
            sparse_embeddings, _ = attn_layer(sparse_embeddings, dense_embeddings, dense_embeddings)
            dense_embeddings, _ = attn_layer(dense_embeddings, sparse_embeddings, sparse_embeddings)
            graph_embeddings, _ = attn_layer(graph_embeddings, dense_embeddings, dense_embeddings)
            dense_embeddings, _ = attn_layer(dense_embeddings, graph_embeddings, graph_embeddings)

        # 返回更新后的特征
        return sparse_embeddings, dense_embeddings, graph_embeddings


class TwoWayTransformer(nn.Module):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer.
          embedding_dim (int): the channel dimension for the input embeddings.
          num_heads (int): the number of heads for multihead attention. Must divide embedding_dim.
          mlp_dim (int): the channel dimension internal to the MLP block.
          activation (nn.Module): the activation to use in the MLP block.
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        # 创建多个 transformer block
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),  # 第一个层跳过位置编码
                )
            )

        # 最后一个注意力层，用于从点到图像的注意力
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
            self,
            image_embedding: Tensor,
            image_pe: Tensor,
            point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # 转换为 (B, H*W, C)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # 位置编码同样处理

        # 准备查询和键
        queries = point_embedding  # 查询点的嵌入
        keys = image_embedding  # 图像嵌入

        # 逐层应用 transformer blocks
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # 最后一个跨点到图像的注意力层
        q = queries + point_embedding  # 加入点的嵌入
        k = keys + image_pe  # 加入图像的位置编码
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse inputs,
        (2) cross attention of sparse inputs to dense inputs, (3) MLP block on sparse inputs,
        and (4) cross attention of dense inputs to sparse inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings.
          num_heads (int): the number of heads in the attention layers.
          mlp_dim (int): the hidden dimension of the mlp block.
          activation (nn.Module): the activation of the mlp block.
          skip_first_layer_pe (bool): skip the PE on the first layer.
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # 跨点到图像的注意力层
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        # MLP block
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        # 跨图像到点的注意力层
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # 自注意力层
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # 跨点到图像的注意力层
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # 跨图像到点的注意力层
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


if __name__ == '__main__':
    # # 创建输入数据
    # B, C, H, W, N_points = 4, 256, 64, 64, 69
    # image_embedding = torch.randn(B, C, H, W)  # 图像嵌入
    # image_pe = torch.randn(B,C,H,W)  # 图像位置编码
    # point_embedding = torch.randn(B, N_points, C)  # 点嵌入
    # print(f"image_embedding shape: {image_embedding.shape}")  # 期望输出: (B, N_points, embedding_dim)
    # print(f"image_pe shape: {image_pe.shape}")  # 期望输出: (B, N_points, embedding_dim)
    # print(f"point_embedding shape: {point_embedding.shape}")  # 期望输出: (B, N_points, embedding_dim)
    #
    # # 创建 TwoWayTransformer 模型
    # depth = 4
    # embedding_dim = 256
    # num_heads = 8
    # mlp_dim = 2048
    # transformer = TwoWayTransformer(
    #     depth=depth,
    #     embedding_dim=embedding_dim,
    #     num_heads=num_heads,
    #     mlp_dim=mlp_dim
    # )
    #
    # # 前向传播
    # queries, keys = transformer(image_embedding, image_pe, point_embedding)
    #
    # # 打印输出形状和中间结果
    #
    # print(f"Queries shape: {queries.shape}")  # 期望输出: (B, N_points, embedding_dim)
    # print(f"Keys shape: {keys.shape}")  # 期望输出: (B, H*W, embedding_dim)
    #
    # # 打印其中一些中间输出（例如，Transformer 中的某些层的输出）

    # 初始化模型
    cross_attention_transformer = ThreeWayCrossAttentionTransformer(embed_dim=256, num_heads=8, num_layers=4)

    # 示例输入
    dense_embeddings = torch.randn(4, 128, 64, 64)  # [B, embed_dim, H, W]
    sparse_embeddings = torch.randn(4, 32, 256)  # [B, N, embed_dim]
    graph_embeddings = torch.randn(4, 32, 256)  # [B, M, embed_dim]

    # 前向传播
    sparse_embeddings, dense_embeddings, graph_embeddings = cross_attention_transformer(
        dense_embeddings, sparse_embeddings, graph_embeddings
    )

    print(sparse_embeddings.shape)  # [4, 32, 256]
    print(dense_embeddings.shape)  # [4, 256, 256] (或者 [4, H*W, embed_dim])
    print(graph_embeddings.shape)  # [4, 32, 256]
