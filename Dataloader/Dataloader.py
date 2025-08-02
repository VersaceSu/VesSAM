import os
import random
import json
import numpy as np
import torch
import msgpack
from torch.utils.data import Dataset
from skimage import io
import networkx as nx
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_geometric.data import Dataset, Data, Batch
import cv2


# 自定义转换：旋转分支点和中点坐标
def rotate_points(points, angle, size):
    center = np.array(size) // 2
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_points = []
    for point in points:
        # 将点的坐标原点转换为中心点
        shifted_point = point - center
        rotated_point = np.dot(rotation_matrix, shifted_point)
        rotated_point = rotated_point + center
        rotated_points.append(rotated_point)
    return rotated_points


# 自定义转换：翻转分支点和中点坐标
def flip_points(points, flip_type, size):
    if flip_type == 'horizontal':
        return [(size[0] - x, y) for x, y in points]
    elif flip_type == 'vertical':
        return [(x, size[1] - y) for x, y in points]
    return points


class MyTransform:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=(0, 90)),  # 随机选择旋转角度
                transforms.RandomRotation(degrees=(90, 180)),
                transforms.RandomRotation(degrees=(180, 270)),
                transforms.RandomRotation(degrees=(270, 360)),
            ]),  # 随机旋转
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomVerticalFlip(),  # 随机垂直翻转
            transforms.ToTensor()  # 转换为 Tensor
        ])

    def __call__(self, image, mask, skeleton, branch_points, mid_points):
        # 先进行图像的变换
        transformed_image = self.image_transform(image)
        transformed_mask = self.image_transform(mask)
        transformed_skeleton = self.image_transform(skeleton)

        # 获取变换后的图像大小
        size = image.size

        # 更新分支点和中点的坐标
        # 1. 旋转变换
        angle = random.choice([0, 90, 180, 270])
        branch_points = rotate_points(branch_points, angle, size)
        mid_points = rotate_points(mid_points, angle, size)

        # 2. 翻转变换
        flip_type = random.choice(['horizontal', 'vertical'])
        branch_points = flip_points(branch_points, flip_type, size)
        mid_points = flip_points(mid_points, flip_type, size)

        return transformed_image, transformed_mask, transformed_skeleton, torch.tensor(branch_points), torch.tensor(mid_points)


class VesselDataset(Dataset):
    def __init__(self, dataset_name, data_dir, split="train", transform=None, jsonfile="dataset_info.json", max_branch_points=16,
                 max_mid_points=16):
        """
        :param dataset_name: 数据集名称（如 "DRIVE", "ARIA", "HRF"等）
        :param data_dir: 数据集的根目录，包含不同子数据集的文件夹
        :param split: 数据集划分（"train", "val", "test"）
        :param transform: 数据预处理的变换函数
        """
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(data_dir, dataset_name)  # 获取特定数据集的路径
        self.split = split
        self.transform = transform
        self.max_branch_points = max_branch_points
        self.max_mid_points = max_mid_points

        self.dataset_info_path = os.path.join(self.data_dir, jsonfile)
        self.dataset_info = self._load_dataset_info()  # 读取 dataset_info.json
        self.images = []
        self.masks = []
        self.skeletons = []
        self.graph = []
        self.branch_points = []
        self.mid_points = []

        self._load_data()

    def _load_dataset_info(self):
        """读取 dataset_info.json 文件"""
        with open(self.dataset_info_path, 'r') as f:
            return json.load(f)

    def _load_data(self):
        """根据 dataset_info.json 加载数据并划分为训练集、验证集、测试集"""
        # 获取所有的文件名（不带扩展名的文件名）
        all_files = list(self.dataset_info.keys())

        # 根据 split 参数划分数据
        train_ratio = 0.8
        val_ratio = 0.2
        test_ratio = 0.

        # 根据划分比例进行拆分
        total_size = len(all_files)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # 使用随机选择的方法划分数据集
        indices = list(range(total_size))
        seed = 42  # 你可以选择任意整数
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        if self.split == "train":
            self._populate_data(train_indices)
        elif self.split == "val":
            self._populate_data(val_indices)
        elif self.split == "test":
            self._populate_data(test_indices)
        elif self.split == "all":
            self._populate_data(indices)

    def _populate_data(self, indices):
        """根据给定的索引列表加载数据"""
        for idx in indices:
            file_name = list(self.dataset_info.keys())[idx]
            info = self.dataset_info[file_name]

            image_path = info["image_path"]
            mask_path = info["mask_path"]
            skeleton_path = info["skeleton_path"]
            # graph_path = info["graph_path"]
            branch_points_path = info["branch_points_path"]
            mid_points_path = info["mid_points_path"]

            self.images.append(image_path)
            self.masks.append(mask_path)
            self.skeletons.append(skeleton_path)
            # self.graph.append(graph_path)
            self.branch_points.append(branch_points_path)
            self.mid_points.append(mid_points_path)

    def _pad_or_trim(self, points, max_len):
        """填充或截断分支点和中点数据"""
        points = points[:max_len]  # 截断到最大长度
        while len(points) < max_len:
            points.append(points[-1])  # 用最后一个元素填充到最大长度
        return points

    def _pad_or_trim_random(self, points, max_len):
        """随机截断或用前面的元素填充分支点和中点数据"""
        if len(points) > max_len:
            points = random.sample(points, max_len)  # 随机截断
        elif len(points) < max_len:
            fill_count = max_len - len(points)
            # 循环填充直到达到 max_len
            points.extend(points * (fill_count // len(points)) + points[:fill_count % len(points)])
        return points

    def _convert_to_pyg_graph(self, skeleton_graph):
        """
        将网络图转化为PyG图。- 图的节点是一个字典，边是由节点的索引对组成的
        """
        G = nx.node_link_graph(skeleton_graph)  # 将msgpack格式的图转换为networkx图
        edges = list(G.edges())  # 每个元素是 (source, target)
        edges = [(u[0], v[0]) for u, v in edges]  # 这里提取每个边的source和target坐标
        edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()  # t()用于转置，确保符合PyG格式 (2, N)
        node_features = torch.zeros(len(G.nodes()), dtype=torch.float)  # 用零向量填充节点特征
        data = Data(x=node_features, edge_index=edge_tensor)
        return data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图像和标签
        image = io.imread(self.images[idx])  # 读取图像
        if len(image.shape) == 2:
            image = cv2.merge([image, image, image])

        mask = io.imread(self.masks[idx])  # 读取标签
        skeleton = io.imread(self.skeletons[idx])  # 读取骨架图

        # with open(self.graph[idx], 'rb') as f:
        #     skeleton_graph = msgpack.load(f)
        # 这边有点问题，还没搞定，估计是搞不定了
        # skeleton_graph = self._convert_to_pyg_graph(skeleton_graph)
        # skeleton_graph = nx.node_link_graph(skeleton_graph)

        # 读取分支点和中点（假设它们是JSON格式）
        with open(self.branch_points[idx], 'r') as f:
            branch_points = json.load(f)
        with open(self.mid_points[idx], 'r') as f:
            mid_points = json.load(f)
        branch_points = torch.tensor(self._pad_or_trim_random(branch_points, self.max_branch_points))
        mid_points = torch.tensor(self._pad_or_trim_random(mid_points, self.max_mid_points))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            skeleton = self.transform(skeleton)
        # image, mask, skeleton, branch_points, mid_points = self.transform(image, mask, skeleton, branch_points, mid_points)

        # 返回图像、标签、骨架、分支点和中点
        return {
            'image': image,
            'mask': mask,
            'skeleton': skeleton,
            # 'graph': skeleton_graph,
            'branch_points': branch_points,
            'mid_points': mid_points,
            'image_name': self.masks[idx].split('/')[-1]
        }


class MultiDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """
        :param data_dir: 数据集的根目录，包含所有数据集文件夹
        :param split: 数据集划分（"train", "val", "test"）
        :param transform: 数据预处理的变换函数
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.datasets = []  # 存储不同数据集的 Dataset 对象
        self._load_all_datasets()

    def _load_all_datasets(self):
        """加载所有数据集并将其合并"""
        # 需要支持的所有数据集名称
        dataset_names = ["DRIVE", "ARIA", "HRF", "IOSTAR", "XCAD", "Aorta", "LSCI"]

        for dataset_name in dataset_names:
            dataset = VesselDataset(dataset_name, self.data_dir, split=self.split, transform=self.transform)
            self.datasets.append(dataset)

        self.images = []
        self.masks = []

        for dataset in self.datasets:
            self.images.extend(dataset.images)
            self.masks.extend(dataset.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = io.imread(self.images[idx])  # 读取图像
        mask = io.imread(self.masks[idx])  # 读取标签

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def data_split(dataset, split_ratio=(0.7, 0.15, 0.15)):
    """
    根据划分比例将数据分为训练集、验证集和测试集
    :param dataset: 输入的数据集
    :param split_ratio: 划分比例
    :return: 训练集、验证集和测试集的 dataset 对象
    """
    train_ratio, val_ratio, test_ratio = split_ratio

    # 计算各个子集的大小
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # 使用随机拆分
    indices = list(range(total_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # # 用于读取单个数据集
    # dataset = VesselDataset(dataset_name="Aorta", data_dir="./data", split="train")
    # train_dataset, val_dataset, test_dataset = data_split(dataset)
    #
    # # 用于读取多个数据集并混合
    # multi_dataset = MultiDataset(data_dir="./data", split="train")
    # train_multi_dataset, val_multi_dataset, test_multi_dataset = data_split(multi_dataset)
    # 数据集目录和名称
    dataset_dir = 'data'
    dataset_name = 'Aorta'  # 假设你的数据集名为 "DRIVE"
    split = 'train'  # 选择数据集的分割方式，可以是 'train', 'val', 'test'

    # transform = MyTransform()
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为 Tensor
        # 可以根据需要添加其他转换，比如归一化
    ])

    # 创建数据集实例
    dataset = VesselDataset(dataset_name=dataset_name, data_dir=dataset_dir, split=split, transform=transform)

    # 使用 DataLoader 加载数据
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 取一个批次的数据进行测试
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"Skeleton shape: {batch['skeleton'].shape}")
        print(f"Branch points shape: {batch['branch_points'].shape}")
        print(f"Mid points shape: {batch['mid_points'].shape}")

        # 测试数据是否能成功加载并进行转换（查看部分数据）
        print(f"First image shape: {batch['image'][0].shape}")
        print(f"First mask shape: {batch['mask'][0].shape}")
        print(f"First skeleton shape: {batch['skeleton'][0].shape}")
        print(f"First branch points: {batch['branch_points'][0].shape}")
        print(f"First mid points: {batch['mid_points'][0].shape}")

        # 可以选择在这之后断开循环，只测试一次
        break
