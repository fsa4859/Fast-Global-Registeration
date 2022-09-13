# examples/Python/Advanced/fast_global_registration.py

import open3d as o3d
from global_registration import *
import numpy as np
import copy
from open3d import read_point_cloud
import os
import copy
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm
from numpy.linalg import inv

import time

def mat2ang_np(mat):
    r = R.from_matrix(mat)
    return r.as_euler("XYZ", degrees=False)

def icp_o3d(src, dst, voxel_size=0.5, trans_init=None):
    '''
    Don't support init_pose and only supports 3dof now.
    Args:
        src: <Nx3> 3-dim moving points
        dst: <Nx3> 3-dim fixed points
        n_iter: a positive integer to specify the maxium nuber of iterations
        init_pose: [tx,ty,theta] initial transformation
        torlerance: the tolerance of registration error
        metrics: 'point' or 'plane'
        
    Return:
        src: transformed src points
        R: rotation matrix
        t: translation vector
        R*src + t
    '''
    # List of Convergence-Criteria for Multi-Scale ICP:
    threshold = 1.25
    if trans_init is None:
        trans_init = np.eye(4)
    criteria = o3d.registration.ICPConvergenceCriteria(1e-6, 1e-6, max_iteration=80)

    registration = o3d.registration.registration_icp(
        src, dst, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPlane(),
        criteria
    )

    transformation = registration.transformation
    R = transformation[:3, :3]
    t = transformation[:3, 3:]
    return R.copy(), t.copy()

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


if __name__ == "__main__":

    voxel_size = 0.05  # means 5cm for the dataset
    
    pre_path = "/home/chao1804/Desktop/fast_register/2012-01-08"
    gt_pose = np.load(os.path.join(pre_path,"gt_pose.npy"))
    print("gt_pose::"+str(gt_pose.shape))

    files = sorted(os.listdir(pre_path))
    files.remove("group_matrix.npy")
    files.remove("gt_pose.npy")
#     print("files:::"+str((files)))
    len_files = len(files)
    # len_files = 400

    pose_est = np.zeros((len_files,6),dtype=np.float32)

    starting_point = 0
    
    for f_index in tqdm(range(len_files)):
        t_file = files[f_index]
        s_file = files[f_index+1]
        # print("gt_pose[f_index+1]:"+str(gt_pose[f_index+1]))
        # print("gt_pose[f_index+1,:-3]:"+str(gt_pose[f_index+1][-3:]))
        # assert()
        RS = R.from_euler('xyz', [gt_pose[f_index+1][-3:]], degrees=True)
        TS = gt_pose[f_index+1][:-3]
        POSES = np.eye(4)
        POSES[:3,3] = TS.T
        POSES[:3,:3] = RS.as_matrix()

        RT = R.from_euler('xyz', [gt_pose[f_index][-3:]], degrees=True)
        TT = gt_pose[f_index][:-3]
        POSET = np.eye(4)
        POSET[:3,3] = TT.T
        POSET[:3,:3] = RT.as_matrix()
        # print("POSES::::"+str(POSES))
        # print("inv(POSES):::"+str(inv(POSES)))
        # print("POSET::::"+str(POSET))

        TST = np.matmul(inv(POSES),POSET)
        T_ = TST[:3,3]
        R_ = TST[:3,:3]
        # print("TST::"+str(TST))

        source_file= os.path.join(pre_path,s_file)
        target_file= os.path.join(pre_path,t_file)
        source_data = read_point_cloud(source_file)
        target_data = read_point_cloud(target_file)
        source_data = np.asarray(source_data.points, dtype=np.float32)
        target_data = np.asarray(target_data.points, dtype=np.float32)

        source, target, source_fpfh, target_fpfh = \
                prepare_dataset(voxel_size, source_data, target_data)

        # draw_registration_result(source, target, TST)
        #draw_registration_result(source, target, TST)

        start = time.time()
        result_fast = execute_fast_global_registration(source, target,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
        # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
        # draw_registration_result(source, target,
        #                             result_fast.transformation)
        ##################
        # criteria = o3d.registration.ICPConvergenceCriteria(1e-6, 1e-6, max_iteration=80)

        # result_icp = o3d.registration.registration_icp(
        #     source, target, 1.25, np.eye(4),
        #     o3d.registration.TransformationEstimationPointToPlane(),
        #     criteria
        # )
        #################
        R0, t0 = icp_o3d(source, target, 0.5, trans_init=result_fast.transformation)
        result_icp_transformation = np.eye(4)
        result_icp_transformation[:3,:3] = R0
        result_icp_transformation[:3,3] = t0.T


        # draw_registration_result(source, target, result_icp_transformation)
        # draw_registration_result(source, target, result_icp.transformation)

        #     result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
        #                                      voxel_size, result_fast)
        # print(result_icp)
        # print(result_icp.transformation)

        # R0 = result_icp.transformation[:3,:3]
        # t0 = result_icp.transformation[:3,3]

        # rot_matrix = R.from_matrix(result_icp.transformation[:3,:3])

        # [roll, pitch, yaw] = rot_matrix.as_euler('xyz', degrees=False)

        if f_index == starting_point:
            R_cum = R0
            t_cum = t0
        else:
            R_cum = np.matmul(R_cum , R0)
            t_cum = np.matmul(R_cum,t0) + t_cum
        #     draw_registration_result(source, target, result_icp.transformation)
        pose_est[f_index+1,:3] = t_cum.T
        pose_est[f_index+1, 3:] = mat2ang_np(R_cum)

        pose_est[f_index+1,:3] = T_.T
        pose_est[f_index+1, 3:] = mat2ang_np(R_)
        # pose_est[f_index+1,4] = pitch
        # pose_est[f_index+1,5] = yaw
        # print("result_icp.transformation:"+str(result_icp.transformation))
        # print("R_cum:::"+str(R_cum))
        # print("t_cum:::"+str(t_cum))
        # assert()
    save_name = os.path.join('pose_ests/0.npy')
    np.save(save_name,pose_est)

    # print('saving results')
    # pose_est = torch.from_numpy(pose_est)
    # local_pc,valid_id = dataset[:]
    # global_pc = utils.transform_to_global_2D(pose_est,local_pc)
    # utils.plot_global_point_cloud(global_pc,pose_est,valid_id,checkpoint_dir)
