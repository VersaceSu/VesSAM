import cv2
import numpy as np
import networkx as nx
from numpy import ndarray, dtype, signedinteger
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import logging
import concurrent.futures
from typing import List, Tuple, Any

# 日志设置
logging.basicConfig(level=logging.INFO)


class SkeletonExtractor:
    def __init__(self, image_path: str, max_size: int = 512, is_build_graph: bool = False):
        """
        初始化骨架提取器。
        :param image_path: 输入的图像路径
        :param max_size: 图像的最大尺寸，默认为 512
        """
        self.image_path = image_path
        self.max_size = max_size
        self.binary_image = None
        self.is_ocpre = False
        self.length_threshold = 20
        self.num_key_points_thresholds = 16
        self.is_build_graph = is_build_graph
        self.up_bound_of_branch_points_nums = 50
        self.distance_threshold = 16

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """调整图像大小，确保最大尺寸为 max_size"""
        h, w = image.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        return image

    def preprocess_image(self, kernel_size: int = 5, is_ocpre: bool = False) -> np.ndarray:
        """读取并处理图像"""
        binary_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if binary_image is None:
            raise ValueError(f"Unable to read the image from {self.image_path}")

        binary_image[binary_image > 0] = 255
        binary_image = self.resize_image(binary_image)

        # 开闭运算去噪
        if is_ocpre:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            binary_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

        self.binary_image = binary_image
        return binary_image

    def zhang_suen_thinning(self, image: np.ndarray) -> np.ndarray:
        """Zhang-Suen 细化算法"""
        return cv2.ximgproc.thinning(image)

    def morphological_skeleton(self, image: np.ndarray) -> np.ndarray:
        """形态学骨架化"""
        eroded = cv2.erode(image, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(eroded, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def skimage_skeletonize(self, image: np.ndarray) -> np.ndarray:
        """skimage 骨架化"""
        binary_image = image / 255
        return skeletonize(binary_image)

    def prune_skeleton(self, skeleton: np.ndarray, length_threshold: int = 50) -> np.ndarray:
        """剪枝操作：去除小于指定长度的骨架"""
        labeled_skeleton = label(skeleton, connectivity=2)
        props = regionprops(labeled_skeleton)
        pruned_skeleton = np.copy(skeleton)
        for prop in props:
            if prop.area < length_threshold:
                for coord in prop.coords:
                    pruned_skeleton[coord[0], coord[1]] = 0
        return pruned_skeleton

    def detect_branch_points(self, skeleton: np.ndarray, min_branch_neighbors: int = 3) -> tuple[
        ndarray[Any, dtype[signedinteger[Any]]], list[tuple[Any, ndarray[Any, Any]]]]:
        """利用卷积对分支点进行高效的 8 连接性检测"""
        skeleton = skeleton.astype(np.uint8)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
        branch_points = np.logical_and(skeleton == 1, neighbor_count >= min_branch_neighbors)
        branch_points_coords = np.argwhere(branch_points)  # 获取分支点的坐标

        # 计算每个分支点的度数并返回
        branch_point_degrees = [(point, neighbor_count[tuple(point)]) for point in branch_points_coords]

        return branch_points_coords, branch_point_degrees

    def filter_branch_points_by_conv_degree(self, branch_point_degrees: list,
                                            distance_threshold: int = 10) -> np.ndarray:
        """
        过滤密集的分支点，保留度数较高且距离较远的分支点。

        参数:
        - branch_point_degrees: 分支点及其度数的列表，格式为 [(point, degree), ...]。
        - distance_threshold: 过滤分支点时的最小距离，距离小于此值的分支点将被移除。

        返回:
        - 过滤后的分支点坐标（numpy数组，形状为 (n, 2)）。
        """
        # 按照度数从高到低排序
        branch_point_degrees_sorted = sorted(branch_point_degrees, key=lambda x: x[1], reverse=True)

        filtered_branch_points = []
        for point, degree in branch_point_degrees_sorted:
            keep_point = True
            # 遍历已有的过滤后的分支点，判断距离是否过近
            for fp in filtered_branch_points:
                if np.linalg.norm(np.array(point) - np.array(fp)) < distance_threshold:
                    keep_point = False
                    break
            if keep_point:
                filtered_branch_points.append(point)
            if len(filtered_branch_points) > self.up_bound_of_branch_points_nums:
                return np.array(filtered_branch_points)

        return np.array(filtered_branch_points)

    def filter_branch_points_by_graph_degree(self, G: nx.Graph, branch_points: List[Tuple[int, int]],
                                             distance_threshold: int = 10) -> np.ndarray:
        """过滤分支点，保留那些足够远且度数较高的点(使用的是构建图的方式）"""
        branch_points_with_degree = [(tuple(point), G.degree[tuple(point)]) for point in branch_points]
        branch_points_sorted = sorted(branch_points_with_degree, key=lambda x: x[1], reverse=True)

        filtered_points = []
        for point, _ in branch_points_sorted:
            keep_point = True
            for fp in filtered_points:
                if np.linalg.norm(np.array(point) - np.array(fp)) < distance_threshold:
                    keep_point = False
                    break
            if keep_point:
                filtered_points.append(point)

        return np.array(filtered_points)

    def skeleton_to_graph(self, skeleton: np.ndarray, order: int = 1) -> nx.Graph:
        """将骨架转换为图结构"""
        order_list = [(-1, 0), (1, 0), (0, -1), (0, 1)] if order == 1 else [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1),
                                                                            (-1, 1), (1, -1), (1, 1)]
        G = nx.Graph()
        skeleton_coords = np.argwhere(skeleton > 0)

        for y, x in skeleton_coords:
            G.add_node((y, x))
            for dy, dx in order_list:
                if (y + dy, x + dx) in map(tuple, skeleton_coords):
                    G.add_edge((y, x), (y + dy, x + dx))

        return G

    def extract_skeleton(self, length_threshold: int = 20, is_ocpre: bool = False) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, nx.Graph, np.ndarray, np.ndarray]:
        """应用不同的骨架提取方法并剪枝"""
        logging.info("Starting image preprocessing...")
        binary_image = self.preprocess_image(is_ocpre=is_ocpre)

        logging.info("Starting skeleton extraction...")
        zhang_suen_result = self.prune_skeleton(self.zhang_suen_thinning(binary_image), length_threshold)
        morphological_result = self.prune_skeleton(self.morphological_skeleton(binary_image), length_threshold)
        skimage_result = self.prune_skeleton(self.skimage_skeletonize(binary_image), length_threshold)

        logging.info("Detecting branch points...")
        branch_points_conv, branch_point_degrees = self.detect_branch_points(skimage_result)

        if self.is_build_graph:
            logging.info("Converting skeleton to graph...")
            G_skimage_result = self.skeleton_to_graph(skimage_result)

        logging.info("Filtering branch points by degree...")
        # 如果一个图的分叉节点少于num_key_points_thresholds, 就不筛选了
        if len(branch_points_conv) > self.num_key_points_thresholds:
            filtered_branch_points = self.filter_branch_points_by_conv_degree(branch_point_degrees,
                                                                              distance_threshold=self.distance_threshold)
        else:
            logging.info("branch points nums not enough to needed filtering")
            filtered_branch_points = branch_points_conv
        # filtered_branch_points = self.filter_branch_points_by_degree(G_skimage_result, branch_points_conv,distance_threshold=30)

        logging.info("Extracting mid points...")
        mid_points = self.extract_mid_points(skimage_result, branch_points_conv)
        # mid_points = self.extract_mid_points(skimage_result, filtered_branch_points)

        result_dict = {
            "binary_image": binary_image,
            "zhang_suen_result": zhang_suen_result,
            "morphological_result": morphological_result,
            "skimage_result": skimage_result,
            "filtered_branch_points": filtered_branch_points,
            "mid_points": mid_points
        }

        if self.is_build_graph:
            result_dict["G_skimage_result"] = G_skimage_result

        return result_dict

    def extract_mid_points(self, skeleton: np.ndarray, branch_points: np.ndarray, long_segment_threshold: int = 20) -> \
            List[Tuple[int, int]]:
        """提取长线段的中点"""
        skeleton_copy = skeleton.copy()

        for y, x in branch_points:
            skeleton_copy[y, x] = 0

        labeled_skeleton, num_features = label(skeleton_copy, connectivity=2, return_num=True)
        mid_points = []

        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled_skeleton == i)
            if len(coords) > long_segment_threshold:
                midpoint = coords[len(coords) // 2]
                mid_points.append((midpoint[0], midpoint[1]))

        return np.array(mid_points)

    def display_results(self, results: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], branch_points: np.ndarray,
                        mid_points: np.ndarray):
        """可视化结果"""
        _, ax = plt.subplots(1, 4, figsize=(12, 12))
        ax[0].imshow(results[0], cmap='gray')
        ax[0].set_title("Original Image")

        ax[1].imshow(results[1], cmap='gray')
        ax[1].scatter(branch_points[:, 1], branch_points[:, 0], color='r', s=1)
        ax[1].set_title("Zhang-Suen Thinned Skeleton")

        ax[2].imshow(results[2], cmap='gray')
        ax[2].scatter(branch_points[:, 1], branch_points[:, 0], color='r', s=1)
        ax[2].set_title("Morphological Thinned Skeleton")

        ax[3].imshow(results[3], cmap='gray')
        ax[3].scatter(branch_points[:, 1], branch_points[:, 0], color='r', s=1)

        # if isinstance(mid_points, list):
        #     mid_points = np.array(mid_points)
        ax[3].scatter(mid_points[:, 1], mid_points[:, 0], color='b', s=10)
        ax[3].set_title("Skimage Thinned Skeleton")

        plt.show()


if __name__ == '__main__':
    image_path = './data/Aorta/masks/0279.png'

    # 创建一个 SkeletonExtractor 实例
    skeleton_extractor = SkeletonExtractor(image_path)

    # 调用 extract_skeleton 方法来提取骨架、分支点和中点
    result = skeleton_extractor.extract_skeleton(length_threshold=50, is_ocpre=False)

    filtered_nums = len(result["filtered_branch_points"])
    mid_points_nums = len(result["mid_points"])

    # 打印出提取到的分支点和中点的数量
    print(f"Number of filtered branch points: {filtered_nums}")
    print(f"Number of mid points: {mid_points_nums}")

    # 可视化结果
    skeleton_extractor.display_results(
        (result["binary_image"], result["zhang_suen_result"], result["binary_image"], result["skimage_result"]),
        result["filtered_branch_points"], result["mid_points"]
    )

    # 可选：保存处理后的骨架图像
    # cv2.imwrite('processed_zhang_suen.png', zhang_suen_result)
    # cv2.imwrite('processed_morphological.png', morphological_result)
    # cv2.imwrite('processed_skimage.png', skimage_result)

    # 你还可以在此基础上继续进行其他分析或保存提取的特征点
    # 例如，可以将分支点保存到文件中
    # np.savetxt('branch_points.csv', filtered_branch_points, delimiter=',', fmt='%d')
    # np.savetxt('mid_points.csv', mid_points, delimiter=',', fmt='%d')
