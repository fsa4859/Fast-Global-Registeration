# examples/Python/Advanced/global_registration.py

import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    # pcd_down = o3d.geometry.voxel_down_sample(pcd,voxel_size=voxel_size)

    # radius_normal = voxel_size * 2
    # o3d.geometry.estimate_normals(pcd_down,
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5

    # pcd_fpfh = o3d.registration.compute_fpfh_feature(
    #     pcd_down,
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def prepare_dataset(voxel_size, source_input, target_input):
    # print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud(source_path)
    # target = o3d.io.read_point_cloud(target_path)
    # source_input = np.load(source_path)
    source_input = source_input[np.where(np.linalg.norm(source_input, axis=1) < 100)[0]]
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_input)
    # source = source.geometry.select_by_index(np.where(np.linalg.norm(np.asarray(source.points), axis=1) < 100)[0])
    o3d.geometry.estimate_normals(source)
    # source.geometry.estimate_normals()

    # target_input = np.load(target_path)
    target_input = target_input[np.where(np.linalg.norm(target_input, axis=1) < 100)]
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_input)
    # target = target.geometry.select_by_index(np.where(np.linalg.norm(np.asarray(target.points), axis=1) < 100)[0])
    o3d.geometry.estimate_normals(target)

    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)

    source, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


if __name__ == "__main__":
    voxel_size = 0.05  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down,
                             result_ransac.transformation)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)