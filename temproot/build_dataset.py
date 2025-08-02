import os
import json
import cv2
import numpy as np
from SkeletonExtractor import SkeletonExtractor
from tqdm import tqdm
import networkx as nx
import msgpack
import networkx as nx
import os
from PIL import Image


def custom_encoder(obj):
    if isinstance(obj, np.int64):
        return int(obj)  # 转换为 Python int
    elif isinstance(obj, tuple):  # 如果是元组，转换为列表
        return list(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class DatasetBuilder:
    def __init__(self, data_dir: str, output_dir: str, max_size: int = 256, is_build_grap: bool = False):
        """
        :param data_dir: 原始数据集目录，包含images和masks文件夹
        :param output_dir: 处理后保存结果的目录
        :param max_size: 最大图像尺寸，默认512
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_size = max_size
        self.is_build_grap = is_build_grap
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.output_image_dir = os.path.join(output_dir, 'images')
        self.output_mask_dir = os.path.join(output_dir, 'masks')
        self.output_skeleton_dir = os.path.join(output_dir, 'skeletons')
        self.output_branch_dir = os.path.join(output_dir, 'branch_points')
        self.output_midpoint_dir = os.path.join(output_dir, 'midpoints')
        self.json_file = os.path.join(output_dir, 'dataset_info.json')
        if self.is_build_grap:
            self.output_graph_dir = os.path.join(output_dir, 'graph')
            os.makedirs(self.output_graph_dir, exist_ok=True)

        # 创建输出目录
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_mask_dir, exist_ok=True)
        os.makedirs(self.output_skeleton_dir, exist_ok=True)
        os.makedirs(self.output_branch_dir, exist_ok=True)
        os.makedirs(self.output_midpoint_dir, exist_ok=True)

    def graph_to_dict(self, G: nx.Graph) -> dict:
        """
        将图结构转换为字典格式，方便存储。
        :param G: 网络图
        :return: 图的字典表示
        """
        graph_dict = {"nodes": [], "edges": []}
        for node in G.nodes(data=True):
            graph_dict["nodes"].append({"node": node[0], "data": node[1]})
        for edge in G.edges(data=True):
            graph_dict["edges"].append({"edge": (edge[0], edge[1]), "data": edge[2]})
        return graph_dict

    def process_image(self, image_name: str):
        """
        处理单个图像，提取骨架、分支点和中点
        """
        image_path = os.path.join(self.image_dir, image_name)
        mask_name = image_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 打开图像和掩膜
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # 调整图像和掩膜大小
        image = image.resize((self.max_size, self.max_size), Image.Resampling.LANCZOS)
        mask = mask.resize((self.max_size, self.max_size), Image.NEAREST)

        # 保存调整大小后的图像和掩膜
        image.save(image_path)  # 可以重新保存调整大小后的图像
        mask.save(mask_path)  # 可以重新保存调整大小后的掩膜

        # 初始化骨架提取器
        extractor = SkeletonExtractor(mask_path, max_size=self.max_size, is_build_graph=self.is_build_grap)
        results = extractor.extract_skeleton(length_threshold=30, is_ocpre=False)
        filtered_nums = len(results["filtered_branch_points"])
        mid_points_nums = len(results["mid_points"])
        print(f"Number of filtered branch points: {filtered_nums}")
        print(f"Number of mid points: {mid_points_nums}")

        # 保存处理后的图像和mask
        output_image_path = os.path.join(self.output_image_dir, image_name)
        output_mask_path = os.path.join(self.output_mask_dir, mask_name)
        cv2.imwrite(output_image_path, cv2.imread(image_path))
        cv2.imwrite(output_mask_path, cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))

        # 保存骨架图像
        skeleton_path = os.path.join(self.output_skeleton_dir, image_name.replace('.jpg', '_skeleton.png'))
        cv2.imwrite(skeleton_path, results['skimage_result'] * 255)

        if self.is_build_grap:
            skeleton_graph = results['G_skimage_result']
            graph_dict = nx.node_link_data(skeleton_graph)
            skeleton_graph_path = os.path.join(self.output_graph_dir, image_name.replace('.jpg', '_graph.msgpack'))
            with open(skeleton_graph_path, 'wb') as f:
                msgpack.dump(graph_dict, f, default=custom_encoder)  # 使用自定义编码器进行序列化

        # 保存分支点
        branch_points = results['filtered_branch_points']
        branch_points_path = os.path.join(self.output_branch_dir, image_name.replace('.jpg', '_branches.json'))
        with open(branch_points_path, 'w') as f:
            json.dump(branch_points.tolist(), f)

        # 保存中点
        mid_points = results['mid_points']
        mid_points_path = os.path.join(self.output_midpoint_dir, image_name.replace('.jpg', '_midpoints.json'))
        with open(mid_points_path, 'w') as f:
            json.dump(mid_points.tolist(), f)

        return {
            'image_path': output_image_path,
            'mask_path': output_mask_path,
            'skeleton_path': skeleton_path,
            'graph_path': skeleton_graph_path,
            'branch_points_path': branch_points_path,
            'mid_points_path': mid_points_path
        }

    def build_dataset(self):
        """
        构建整个数据集
        """
        dataset_info = {}
        image_names = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

        for image_name in tqdm(image_names, desc="Processing images"):
            image_info = self.process_image(image_name)
            dataset_info[image_name] = image_info

        # 保存dataset信息到JSON文件
        with open(self.json_file, 'w') as f:
            json.dump(dataset_info, f, indent=4)


if __name__ == "__main__":
    data_dir = 'rawdata/Aorta'
    output_dir = 'data/Aorta'
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    is_build_grap = True

    builder = DatasetBuilder(data_dir, output_dir, is_build_grap=is_build_grap)
    builder.build_dataset()
