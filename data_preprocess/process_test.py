import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import time


def resize_image(image, max_size=512):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return image


def preprocess_image(image_path, kernel_size=5, is_ocpre=False):
    # 读取二值图像
    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 确保图像是二值图（0和255）
    binary_image[binary_image > 0] = 255
    # 调整大小
    binary_image = resize_image(binary_image)

    # 开闭运算去噪
    if is_ocpre:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        binary_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    return binary_image


# 1. Zhang-Suen 细化算法
def zhang_suen_thinning(image):
    return cv2.ximgproc.thinning(image)


# 2. 形态学骨架化
def morphological_skeleton(image):
    eroded = cv2.erode(image, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(eroded, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# 3. skimage骨架化
def skimage_skeletonize(image):
    binary_image = image / 255
    return skeletonize(binary_image)


# 剪枝操作
def prune_skeleton(skeleton, length_threshold=50):
    labeled_skeleton = label(skeleton, connectivity=2)
    props = regionprops(labeled_skeleton)
    pruned_skeleton = np.copy(skeleton)
    for prop in props:
        if prop.area < length_threshold:
            for coord in prop.coords:
                pruned_skeleton[coord[0], coord[1]] = 0
    return pruned_skeleton


def filter_branch_points_by_degree(G, branch_points, distance_threshold=10):
    """
    Filters branch points to retain only those that are sufficiently distant from each other,
    prioritizing points with higher degrees (more connections).

    Parameters:
    - G: The graph representing the skeleton.
    - branch_points: List of coordinates for detected branch points.
    - distance_threshold: Minimum allowed distance between retained branch points.

    Returns:
    - Filtered list of branch points.
    """
    # 计算每个分叉点的连接度并按降序排序
    branch_points_with_degree = [(tuple(point), G.degree[tuple(point)]) for point in branch_points]
    print(branch_points_with_degree)
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


# 主函数：应用不同的骨架提取方法并剪枝
def extract_skeleton(image_path, length_threshold=20, is_ocpre=False):
    start_time = time.time()
    t1 = time.time()

    binary_image = preprocess_image(image_path, is_ocpre=is_ocpre)

    t2 = time.time()
    print(f"Image preprocess took {t2 - t1:.4f} seconds")
    t1 = time.time()

    zhang_suen_result = prune_skeleton(zhang_suen_thinning(binary_image), length_threshold)
    morphological_result = prune_skeleton(morphological_skeleton(binary_image), length_threshold)
    skimage_result = prune_skeleton(skimage_skeletonize(binary_image), length_threshold)
    t2 = time.time()
    print(f"Skeleton extraction took {t2 - t1:.4f} seconds")
    t1 = time.time()

    # ******************卷积提取特征点和长距离骨架中间节点****************
    branch_points_conv = detect_branch_points(skimage_result)
    t2 = time.time()
    print(f"Skeleton Conv took {t2 - t1:.4f} seconds")
    t1 = time.time()

    # *********************添加一下图结构的测试************************
    # G_skimage_result, branch_points_skimage_result = skeleton_to_graph(skimage_result)
    G_skimage_result = skeleton_to_graph(skimage_result)
    t2 = time.time()
    print(f"Skeleton_to_graph {t2 - t1:.4f} seconds")
    t1 = time.time()

    # 去掉一些密集的点
    filtered_branch_points = filter_branch_points_by_degree(G_skimage_result, branch_points_conv, distance_threshold=40)
    # 提取中间节点
    mid_points = extract_mid_points(skimage_result, branch_points_conv)

    t2 = time.time()
    print(f"filtered_branch_points took {t2 - t1:.4f} seconds")

    return binary_image, zhang_suen_result, morphological_result, skimage_result, G_skimage_result, filtered_branch_points, mid_points


# 显示结果
def display_results(original, zhang_suen, morphological, skimage):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title('Original Binary Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Zhang-Suen Thinning')
    plt.imshow(zhang_suen, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Morphological Skeleton')
    plt.imshow(morphological, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('skimage Skeletonize')
    plt.imshow(skimage, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(zhang_suen, cmap='gray')
    plt.scatter([x[1] for x in branch_points], [x[0] for x in branch_points], color='red', marker='*', s=50,
                label='Branch Points')
    plt.scatter([x[1] for x in mid_points], [x[0] for x in mid_points], color='green', marker='*', s=50,
                label='Branch Points')
    plt.title('Skeleton with Zhang-Suen Thinning Branch Points')
    plt.legend()
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(skimage, cmap='gray')
    plt.scatter([x[1] for x in branch_points], [x[0] for x in branch_points], color='red', marker='*', s=50,
                label='Branch Points')
    plt.title('Skeleton with Branch Points')
    plt.legend()
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def skeleton_to_graph(skeleton, Order=1):
    if Order == 1:
        Order_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if Order == 2:
        Order_list = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    G = nx.Graph()
    skeleton_coords = np.argwhere(skeleton > 0)

    for y, x in skeleton_coords:
        G.add_node((y, x))

        # 添加边
        for dy, dx in Order_list:
            if (y + dy, x + dx) in map(tuple, skeleton_coords):
                G.add_edge((y, x), (y + dy, x + dx))

    # 分叉点：度数大于 2 的节点
    # branch_points = [node for node in G.nodes if G.degree(node) > 2]
    # return G, branch_points
    return G


# 利用卷积对分支点进行高效的 8 连接性检测
def detect_branch_points(skeleton, min_branch_neighbors=3):
    # 转换为uint表达帮助后续计算
    skeleton = skeleton.astype(np.uint8)
    # 定义8联通卷积核
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # 应用卷积来计算邻域
    neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)

    plt.imshow(neighbor_count, cmap='hot')
    plt.title("Neighbor Count")
    plt.colorbar()
    plt.show()

    # 分支点：总和至少为 min_branch_neighbors的分支点
    branch_points = np.logical_and(skeleton == 1, neighbor_count >= min_branch_neighbors)
    return np.argwhere(branch_points)


# def filter_branch_points(branch_points, distance_threshold=100):
#     # 过滤分支点，只保留那些彼此距离足够远的分支点
#     filtered_points = []
#     for point in branch_points:
#         keep_point = True
#         for fp in filtered_points:
#             if np.linalg.norm(np.array(point) - np.array(fp)) < distance_threshold:
#                 keep_point = False
#                 break
#         if keep_point:
#             filtered_points.append(point)
#     return np.array(filtered_points)


# 提取长线段的中点
def extract_mid_points(skeleton, branch_points, long_segment_threshold=20):
    skeleton_copy = skeleton.copy()

    # 在分叉点位置断开连通区域
    for y, x in branch_points:
        skeleton_copy[y, x] = 0

    labeled_skeleton, num_features = label(skeleton_copy, connectivity=2, return_num=True)
    mid_points = []

    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_skeleton == i)
        if len(coords) > long_segment_threshold:
            midpoint = coords[len(coords) // 2]
            mid_points.append((midpoint[0], midpoint[1]))

    return mid_points


def display_skeleton_points(skeleton):
    # 已经融合进display_results
    plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap='gray')
    plt.scatter([x[1] for x in branch_points], [x[0] for x in branch_points], color='red', marker='*', s=50,
                label='Branch Points')
    plt.title('Skeleton with Branch Points')
    plt.legend()
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    image_path = 'data_demo/Retinal1.png'  # 替换为你的图像路径
    binary_image, zhang_suen_result, morphological_result, skimage_result, G, branch_points, mid_points = extract_skeleton(
        image_path,
        length_threshold=50)
    display_results(binary_image, zhang_suen_result, morphological_result, skimage_result)
