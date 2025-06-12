import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import json

def save_pcd_as_ply(pcds, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcds)
    o3d.io.write_point_cloud(filename, pcd)


def symmetric_point_cloud_torch(points):    
    # Mirror the points across the Y axis
    mirrored_points = points.clone()
    mirrored_points[:, 0] = -mirrored_points[:, 0]
    
    # Combine original and mirrored points to form a symmetric point cloud
    symmetric_points = torch.cat((points, mirrored_points), dim=0)
    
    N = points.size(0)
    # Compute the distance matrix
    dists = torch.cdist(symmetric_points, symmetric_points, p=2)
    
    # To avoid considering the distance of each point to itself, set the diagonal to a large value
    dists.fill_diagonal_(float('inf'))
    
    paired_indices = []
    used = torch.zeros(symmetric_points.size(0), dtype=torch.bool, device=points.device)
    
    # Find unique pairs of points
    for i in range(symmetric_points.size(0)):
        if used[i]:
            continue
        # Find the closest point not already paired
        closest_idx = torch.argmin(dists[i])
        if used[closest_idx]:
            continue
        paired_indices.append((i, closest_idx))
        used[i] = True
        used[closest_idx] = True
    
    # Ensure the output size is the same as input
    if len(paired_indices) > N:
        paired_indices = paired_indices[:N]
    elif len(paired_indices) < N:
        remaining_indices = [i for i in range(len(dists)) if not used[i]]
        for i in remaining_indices[:N - len(paired_indices)]:
            paired_indices.append((i, i))
    
    # Average the paired points to get the final points
    averaged_points = torch.stack([(symmetric_points[i] + symmetric_points[j]) / 2 
                                   for i, j in paired_indices])
    
    return averaged_points
    

def remove_points_near_origin(point_cloud, threshold=2e-2):
    distances = torch.norm(point_cloud, dim=1)
    mask = distances > threshold
    filtered_point_cloud = point_cloud[mask]
    common_mask = torch.ones(25, dtype=torch.bool).to(mask.device)
    mask = torch.cat((common_mask, mask))
    return filtered_point_cloud, mask


def sort_point_cloud(points):
    """
    Sorts the input point cloud based on z-axis values from top to bottom.
    If z values are close, sorts by x-axis from left to right.

    Args:
    points (torch.Tensor): Input point cloud with shape (N, 3).

    Returns:
    torch.Tensor: Sorted point cloud.
    """
    # First sort by z-axis (descending)
    points_np = points.clone()
    points_np = points_np.detach().cpu().numpy()
    indices = torch.from_numpy(np.lexsort((-points_np[:, 0], -points_np[:, 1])))
    
    # sorted_points, indices = torch.sort(points[:, 1], descending=True)
#     points = points[indices]

#     # Now sort by x-axis within the groups of similar z values
#     unique_z_values = torch.unique(points[:, 1], sorted=True, return_inverse=True)[1]

#     sorted_indices = []
#     for z_value in torch.unique(unique_z_values):
#         group_indices = torch.where(unique_z_values == z_value)[0]
#         group_points = points[group_indices]
#         group_sorted_indices = torch.sort(group_points[:, 0])[1]
#         sorted_indices.append(group_indices[group_sorted_indices])
#     sorted_indices = torch.cat(sorted_indices)

    return points[indices]

def modify_skin_weights(original_pcd, skinning_tensor, new_pcd):
    # result = torch.zeros_like(skinning_tensor)
    # top_four_values, top_four_indices = torch.topk(skinning_tensor, k=2, dim=1)
    # for i in range(skinning_tensor.size(0)):
    #     result[i, top_four_indices[i]] = top_four_values[i]
    max_values, _ = torch.max(skinning_tensor, dim=1, keepdim=True)
    skinning_tensor[skinning_tensor < max_values * 0.15] = 0.0
    result = skinning_tensor
    
    row_sums = result.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1  # 避免除以0，对于全0行，这会保持它们为全0
    result /= row_sums
    
    result_raw = transfer_skin_weights(original_pcd, result, new_pcd)
    
    return result, result_raw

# def transfer_skin_weights(original_pcd, original_weights, new_pcd, k=1):
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(original_pcd)
#     distances, indices = nbrs.kneighbors(new_pcd)
#     new_weights = original_weights[indices].mean(axis=1)
    
#     return new_weights

def pairwise_distances(x, y):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, 0.0, float('inf'))

def transfer_skin_weights(original_pcd, original_weights, new_pcd, k=1):
    # Compute pairwise distances between new_pcd and original_pcd
    distances = pairwise_distances(new_pcd, original_pcd)
    
    # Find the k nearest neighbors
    knn_distances, knn_indices = torch.topk(distances, k=k, dim=1, largest=False)
    
    # Get the weights for the nearest neighbors and compute their mean
    knn_weights = original_weights[knn_indices]  # (M, k, W)
    new_weights = knn_weights.mean(dim=1)        # (M, W)
    return new_weights

def load_ply(file_path):
    """Load a PLY file and convert it to a point cloud using Open3D."""
    pcd = o3d.io.read_point_cloud(file_path)  # 读取ply文件
    point_cloud = np.asarray(pcd.points)      # 获取点云的顶点坐标
    point_cloud = torch.from_numpy(point_cloud)
    return point_cloud




def project_pcd2image(point_cloud, K, extrinsic, image_size=(256, 256), original_size=(1024, 1024)):
    """
    将3D点云投影到2D图像平面上，并缩放到256x256图像大小
    
    参数:
    point_cloud: (N, 3) 形状的点云，N是点的数量
    K: (3, 3) 相机内参矩阵
    extrinsic: (4, 4) 相机外参矩阵
    image_size: 输出图像的尺寸 (height, width)
    original_size: 原始渲染图像的尺寸 (height, width)
    
    返回:
    pixel_coords: 点在渲染图像中的像素坐标 (N, 2)
    """
    # 将点云转换为齐次坐标
    point_cloud_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    
    # 将点云从世界坐标系转换到相机坐标系
    point_camera_coords = (extrinsic @ point_cloud_homogeneous.T).T[:, :3]
    
    # 提取相机内参
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # 投影到图像平面
    u = fx * (point_camera_coords[:, 0] / point_camera_coords[:, 2]) + cx
    v = fy * (point_camera_coords[:, 1] / point_camera_coords[:, 2]) + cy
    
    # 创建原始的像素坐标
    pixel_coords = np.stack([u, v], axis=-1)
    
    # 缩放像素坐标到256x256
    scale_x = image_size[1] / original_size[1]
    scale_y = image_size[0] / original_size[0]
    pixel_coords[:, 0] *= scale_x
    pixel_coords[:, 1] *= scale_y
    
    # 限制像素坐标在目标图像的范围内
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, image_size[1] - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, image_size[0] - 1)
    
    pixel_coords = np.floor(pixel_coords).astype(int)
    
    return pixel_coords
    


def load_camera_params_from_json(json_path):
    # 读取json文件
    with open(json_path, 'r') as f:
        camera_params = json.load(f)
    
    # 提取相机的原点和坐标轴
    origin = np.array(camera_params['origin'])
    x = np.array(camera_params['x'])
    y = np.array(camera_params['y'])
    z = np.array(camera_params['z'])
    
    # 构建旋转矩阵 R
    R = np.column_stack((x, y, z))  # 将x, y, z 作为列向量拼接成3x3的旋转矩阵
    
    # 构建平移向量 t
    t = origin.reshape(3, 1)
    
    # 构建 4x4 外参矩阵 T
    extrinsic_matrix = np.eye(4)  # 先创建一个 4x4 单位矩阵
    extrinsic_matrix[:3, :3] = R  # 填入旋转矩阵
    extrinsic_matrix[:3, 3] = t.flatten()  # 填入平移向量
    
    return extrinsic_matrix


class MultiViewImageFeatureExtractor(nn.Module):
    def __init__(self, image_encoder, image_processor):
        super(MultiViewImageFeatureExtractor, self).__init__()
        self.clip_image_encoder = image_encoder
        self.clip_image_processor = image_processor
    
    def forward(self, image_folder):
        image_features = []
        for filename in os.listdir(image_folder):
            if filename.endswith(".png"):
                image_path = os.path.join(image_folder, filename)
                raw_image = Image.open(image_path)

                # 将图像转换为 CLIP 模型的输入格式
                image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values[0]
                
                # 提取图像特征
                with torch.no_grad():
                    image_embeds = self.clip_image_encoder(image.unsqueeze(0)).image_embeds.unsqueeze(1)  # patch/class
                    image_features.append(image_embeds)
        
        # 将所有视角的特征拼接成一个大张量
        image_features = torch.cat(image_features, dim=1)
        return image_features