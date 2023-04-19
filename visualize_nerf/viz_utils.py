import open3d as o3d
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv
from PIL import Image
import torchvision.transforms as T
import sys
sys.path.append('/home/zubairirshad/nerf_pl')
from visualize_nerf.transform_utils import *
from visualize_nerf.utils import *

# def get_masked_textured_pointclouds(rgb, intrinsics, depth_range= (0, 10), width= 480, height=640):
#     xmap = np.array([[y for y in range(width)] for z in range(height)])
#     ymap = np.array([[z for y in range(width)] for z in range(height)])

#     pixel_coords = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32)

#     # Convert pixel coordinates to normalized image coordinates
#     norm_pixel_coords = np.hstack((pixel_coords, np.ones((height*width, 1))))
#     norm_pixel_coords = np.dot(np.linalg.inv(intrinsics), norm_pixel_coords.T).T


#     depth = np.linspace(depth_range[0], depth_range[1], num=height*width).reshape(height, width)
#     depth_masked = depth.reshape(-1)[:, np.newaxis]
#     xmap_masked = xmap.flatten()[:, np.newaxis]
#     ymap_masked = ymap.flatten()[:, np.newaxis]
#     rgb = rgb.reshape(-1, 3) / 255.0
#     pt2 = depth_masked
#     pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
#     pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
#     points = np.concatenate((pt0, pt1, pt2), axis=1)
#     return points, rgb



def visualize_poses(all_c2w, all_c2w_val, all_c2w_test, fov, obj_location, 
                    sphere_radius=1, bbox_dimension= None, RTs=None,
                    W=None, asset_pose_inv=None, box=None, spherical = False):
    things_to_draw = []
    focal = (640 / 2) / np.tan((fov / 2) / (180 / np.pi))
    frustums = []
    fig = pv.figure()
    ref_camera = 1
    draw_camera_grid = False
    all_c2w = all_c2w_test
    ref_cameras = [0]
    for i, C2W in enumerate(all_c2w):
        img_size = (640, 480)
        # if asset_pose_inv is not None:
        #     scale_factor = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
        #     C2W = asset_pose_inv @ C2W
        #     C2W[:3,3]/=scale_factor
        #     C2W[3,3] = 1
        if i in ref_cameras:
            frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.5, color=[0,1,0]))
        # elif i==ref_camera:
        #     frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.5, color=[0,1,0]))
        else:
            frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.05, color=[1,0,0]))
    # for C2W in all_c2w_test:
    #     if asset_pose_inv is not None:
    #         scale_factor = np.max([(box[1,0]-box[0,0]), (box[1,1]-box[0,1]), (box[1,2]-box[0,2])])
    #         C2W = asset_pose_inv @ C2W
    #         C2W[:3,3]/=scale_factor
    #         C2W[3,3] = 1
    #     frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.05, color=[0,0,1]))

    if bbox_dimension is not None:
        if isinstance(bbox_dimension, dict):
            all_R = bbox_dimension['R']
            all_T = bbox_dimension['T']
            all_s = bbox_dimension['s']
            
            for Rot,Tran,sca in zip(all_R, all_T, all_s):
                bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
                things_to_draw.append(bbox)

    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    # mat = o3d.visualization.rendering.MaterialRecord()
    # mat.shader = "unlitLine"
    # mat.line_width = 10  # note that this is scaled with respect to pixels,

    # grid_size=[16, 16, 16]
    grid_size = [96, 96, 96]
    sfactor = 10

    side_lengths = [1, 1, 1]
    world_grid = get_world_grid([[-side_lengths[0], side_lengths[0]],
                                        [-side_lengths[1], side_lengths[1]],
                                        [0, side_lengths[2]],
                                        ], [int(grid_size[0]/sfactor), int(grid_size[1]/sfactor), int(grid_size[2]/sfactor)] )  # (1, grid_size**3, 3)

    # side_lengths = [0.5, 0.5, 0.5]
    # side_length = 3
    # side_lengths = [3, 3, 6]
    # world_grid = get_world_grid([[-side_lengths[0], side_lengths[0]],
    #                                     [-side_lengths[1], side_lengths[1]],
    #                                     [-side_lengths[2], side_lengths[2]],
    #                                     ], [int(grid_size[0]), int(grid_size[1]), int(grid_size[2])] )  # (1, grid_size**3, 3)
    
    # # side_lengths = [1, 1, 1]
    # # print("world_grid", world_grid.shape)
    # wg_homo = torch.cat((world_grid.squeeze(0), torch.ones(world_grid.shape[1],1)), dim=-1)
    # world_grid = torch.FloatTensor(all_c2w[49]) @ wg_homo.T
    # world_grid = world_grid[:3, :].T

    print("===================================\n\n\n")
    print("all_c2w[49][:3,:3]", all_c2w[49][:3,:3])
    print("===================================\n\n\n")
    # world_grid, mag_original = contract_samples(world_grid.squeeze(0))
    # world_grid = (torch.FloatTensor(all_c2w[49][:3,:3]) @ world_grid.squeeze(0).T).T

    print("world grid afer trasnform", world_grid.shape)

    # norm_pose_0 = [[ 7.14784828e-01, -3.56641552e-01, 6.01572484e-01, 6.02849754e-01],
    #     [ 6.99344443e-01,  3.64515616e-01, -6.14854223e-01, -6.16159694e-01],
    #     [-6.48906216e-18,  8.60194844e-01,  5.09965519e-01, 5.11048288e-01],
    #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
    
    # norm_pose_0 = torch.FloatTensor(norm_pose_0)
    # wg_homo = torch.cat((world_grid.squeeze(0), torch.ones(world_grid.shape[1],1, device=world_grid.device)), dim=-1)
    # world_grid = norm_pose_0 @ wg_homo.T
    # world_grid = world_grid[:3, :].T
    # world_grid = world_grid.unsqueeze(0)


    # print("world_grid", world_grid.shape)
    # # world_grid = world_grid[:3, :].T

    # wg_homo = torch.cat((world_grid.squeeze(0), torch.ones(world_grid.shape[1],1)), dim=-1)
    # world_grid = torch.FloatTensor(all_c2w[49]) @ wg_homo.T
    # world_grid = world_grid[:3, :].T

    print("torch.max min world_grid", torch.max(world_grid[:,0]), torch.min(world_grid[:,0]))
    print("torch.max min world_grid", torch.max(world_grid[:,1]), torch.min(world_grid[:,1]))
    print("torch.max min world_grid", torch.max(world_grid[:,2]), torch.min(world_grid[:,2]))


    # world_grid, mag_original = contract_samples(world_grid)
    # world_grid = world_grid_contract


    # world_grid = _inverse_contract(world_grid)

    print("world_grid", world_grid.shape)


    world_grid_o3d = o3d.geometry.PointCloud()
    world_grid_o3d.points = o3d.utility.Vector3dVector(world_grid.squeeze(0).numpy())
    # world_grid_o3d.paint_uniform_color([0, 0.706, 1])

    # print("world_grid.squeeze(0).numpy()", world_grid.squeeze(0).numpy().shape)
    # if not draw_camera_grid:
    #     things_to_draw.append(world_grid_o3d)
    

    # world_grid = inverse_contract_samples(world_grid, mag_original)
    # print("world grid after inverse contract samples", world_grid.shape)
    # world_grid = world_grid.unsqueeze(0)

    # world_grids = world_grid.clone().view(-1,3).unsqueeze(0)
    world_grids = world_grid.clone().view(-1,3)


    
    scale_factor = 2
    w = 640
    h = 480
    width  = w/scale_factor
    height = h/scale_factor
    
    intrinsics = np.array([
        [focal, 0., 640 / 2.0],
        [0., focal, 480 / 2.0],
        [0., 0., 1.],
    ])

    K = torch.FloatTensor(intrinsics)

    K = K/scale_factor
    K[-1,-1] = 1
    

    c = torch.FloatTensor([640/2, 480/2]).unsqueeze(0)
    focal_all = torch.FloatTensor([focal]).unsqueeze(-1).repeat((1, 2))
    focal_all[..., 1] *= -1.0
    focal_all = focal_all/scale_factor
    c = c/scale_factor



    #=============================================\n\n\n
        #=============================================\n\n\n
            #=============================================\n\n\n
    #all at once
    # c2w_all = torch.FloatTensor(all_c2w[ref_cameras])
    # all_im = []
    # rt = False
    # for ref_camera in ref_cameras:
        
    #     if ref_camera<10:
    #         img_name = 'midsize_muscle_02-00'+ str(ref_camera) +'.png'
    #     else:
    #         img_name = 'midsize_muscle_02-0'+ str(ref_camera) +'.png'
    #     img_path = '/home/zubairirshad/pd-api-py/PD_v3_eval/test_novelobj/SF_6thAndMission_medium0/val/rgb/' + img_name

    #     im = Image.open(img_path)


    #     print("im", np.array(im).shape)
    #     im = im.resize((int(width),int(height)), Image.LANCZOS)
    #     transform = T.ToTensor()
    #     im = transform(im).unsqueeze(0)
    #     print("im", im.shape)
    #     _,_,height,width = im.shape
    #     all_im.append(im)
    # im = torch.cat(all_im, dim=0)
    # print("im", im.shape)

    # NS = len(ref_cameras)
    # print("world_grids", world_grids.shape)
    # if rt:
    #     world_grids = repeat_interleave(world_grids.unsqueeze(0), NS)
    #     c2w_single = [convert_pose(torch.FloatTensor(all_c2w[ref_camera])).unsqueeze(0) for ref_camera in ref_cameras]
    #     c2w_all = torch.cat(c2w_single, dim=0).to(dtype = torch.float)
    #     camera_grids_w2c, uv_rt, im_z = w2i_projection(world_grids.clone(), c2w_all, K)
    #     print("im_z", im_z.shape)
    #     uv_pn = uv_rt
    # else:

    #     camera_grids_w2c = world2camera(world_grids.clone().unsqueeze(0), c2w_all.clone(), NS=NS)
    #     print("camera_grids_w2c", camera_grids_w2c.shape)
    #     uv_pn = projection(camera_grids_w2c.clone(), focal_all.clone(), c.clone(), NV=1)
    

    # im_x = uv_pn[:,:, 0]
    # im_y = uv_pn[:,:, 1]
    # im_grid_pn = torch.stack([2 * im_x / (width - 1) - 1, 2 * im_y / (height - 1) - 1], dim=-1)

    # if rt:
    #     mask_z = camera_grids_w2c[:,:,2]>0

    # else:
    #     mask_z = camera_grids_w2c[:,:,2]<1e-3

    # mask = im_grid_pn.abs() <= 1
    # print("maskz, mask", mask_z.shape, mask.shape)
    # mask = (mask.sum(dim=-1) == 2) & (mask_z)
    # print("mask", mask.shape)


    # im_grid_pn = im_grid_pn.unsqueeze(2)
    # print("im", im.shape, im_grid_pn.shape)
    # data_im = F.grid_sample(im, im_grid_pn, align_corners=True, mode='bilinear', padding_mode='zeros')
    
    # if rt:
    #     print("data_im, mask", data_im.shape, mask.shape)
    #     im_z[mask.unsqueeze(-1)==False] = 0
    #     im_z_mean = im_z[im_z > 0].mean()
    #     im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
    #     im_z_norm = (im_z - im_z_mean) / im_z_std
    #     im_z_norm[im_z <= 0] = 0
    #     print("im_z_norm",im_z_norm.shape)
    

    # data_im[mask.unsqueeze(1).unsqueeze(-1).repeat(1, data_im.shape[1], 1,1) == False] = 0

    # print("mask", mask.shape)

    # print("mask.unsqueeze(1).unsqueeze(-1).repeat(1, data_im.shape[1], 1,1", mask.unsqueeze(1).unsqueeze(-1).repeat(1, data_im.shape[1], 1,1).shape)

    # print("data_im", data_im.shape)
    # data_im = data_im.squeeze().view(NS, 3, int(grid_size[0]/sfactor), int(grid_size[1]/sfactor), int(grid_size[2]/sfactor)).permute(0, 2,3,4, 1)
    # all_data_im = data_im.view(NS, -1, 3)
    # print("all_data_im", all_data_im.shape)



    #=============================================\n\n\n
        #=============================================\n\n\n
            #=============================================\n\n\n

    #=============================================\n\n\n
    # single view CHECK
    #=============================================\n\n\n
    # print("focal all", focal_all, c)
    rt = False
    all_data_im = []
    for ref_camera in ref_cameras:
        if ref_camera<10:
            img_name = 'midsize_muscle_02-00'+ str(ref_camera) +'.png'
        else:
            img_name = 'midsize_muscle_02-0'+ str(ref_camera) +'.png'
        img_path = '/home/zubairirshad/pd-api-py/PD_v3_eval/test_novelobj/SF_6thAndMission_medium0/val/rgb/' + img_name
        im = Image.open(img_path)
        print("im", np.array(im).shape)

        # plt.imshow(np.array(im))
        # plt.show()
        im = im.resize((int(width),int(height)), Image.LANCZOS)

        # points, rgb = get_masked_textured_pointclouds(np.array(im), intrinsics)
        transform = T.ToTensor()

        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # pcd.colors = o3d.utility.Vector3dVector(rgb)
        # o3d.visualization.draw_geometries([pcd])
        
        im = transform(im).unsqueeze(0)
        _,_,height,width = im.shape
        
        # K = torch.FloatTensor(intrinsics)
        # points_3d = torch.inverse(K) @ points_2d.transpose(1, 2)
        # points_3d = points_3d.transpose(1, 2)
        
        # transform wg in world coordinates i.e. c1 to world coordinates i.e. c2
        # wg_homo = torch.cat((world_grids, torch.ones(world_grid.shape[1],1)), dim=-1 )
        # camera_grids_w2c = torch.linalg.inv(torch.FloatTensor(all_c2w[ref_camera])) @ wg_homo.T
        # camera_grids_w2c = camera_grids_w2c[:3, :].T
        # camera_grids_w2c = camera_grids_w2c.unsqueeze(0)

        # c2w_single = torch.FloatTensor(all_c2w[0]).unsqueeze(0)

        # c2w_single = torch.linalg.inv(torch.FloatTensor(all_c2w[ref_camera]).unsqueeze(0))

        if rt:
            c2w_single = convert_pose(all_c2w[ref_camera])
            c2w_single = torch.FloatTensor(c2w_single).unsqueeze(0)
            camera_grids_w2c, uv_rt = w2i_projection(world_grids.clone().unsqueeze(0), c2w_single, K)

        else:
            c2w_single = torch.FloatTensor(all_c2w[ref_camera]).unsqueeze(0)
            camera_grids_w2c = world2camera(world_grids.clone(), c2w_single.clone(), NS=1)
            # camera_grids_w2c = world2camera_matrix(world_grids.clone(), c2w_single.clone())
            uv_pn = projection(camera_grids_w2c.clone(), focal_all.clone(), c.clone(), NV=1)

        
        # c2w_single = convert_pose(all_c2w[ref_camera])


        # c2w_single = torch.FloatTensor(all_c2w[ref_camera]).unsqueeze(0)
        # camera_grids_w2c = world2camera(world_grids.clone(), c2w_single.clone(), NS=1)

        # print("world_grids, c2w_single", world_grids.shape, c2w_single.shape)
        # c2w_single = 
        # camera_grids_w2c = camera_grids_w2c.unsqueeze(0)
        camera_grid_o3d = o3d.geometry.PointCloud()
        camera_grid_o3d.points = o3d.utility.Vector3dVector(camera_grids_w2c.squeeze(0).numpy())
        # camera_grid_o3d.paint_uniform_color([1, 0.706, 0])

        if rt:
            uv_pn = uv_rt
        else:
            uv_pn = uv_pn
        im_x = uv_pn[:,:, 0]
        im_y = uv_pn[:,:, 1]
        im_grid_pn = torch.stack([2 * im_x / (width - 1) - 1, 2 * im_y / (height - 1) - 1], dim=-1)

        # mask_z = camera_grids_w2c[:,:,2]<1e-3

        # mask_z = camera_grids_w2c[:,:,2]<1e-3
        if rt:
            mask_z = camera_grids_w2c[:,:,2]>0
        else:
            mask_z = camera_grids_w2c[:,:,2]<1e-3



        print("camera_grids_w2c", camera_grids_w2c.shape)
        mask = im_grid_pn.abs() <= 1
        print("maskz, mask", mask_z.shape, mask.shape)
        mask = (mask.sum(dim=-1) == 2) & (mask_z)
        # print("mask", mask.shape)

        im_grid_pn = im_grid_pn.unsqueeze(2)
        print("im", im.shape, im_grid_pn.shape)
        data_im = F.grid_sample(im, im_grid_pn, align_corners=True, mode='bilinear', padding_mode='zeros')
        print("data_im", data_im.shape)

        #print("data_im, mask", data_im.shape, mask.shape)
        data_im[mask.unsqueeze(1).unsqueeze(-1).repeat(1, data_im.shape[1], 1,1) == False] = 0

        data_im = data_im.squeeze().view(3, int(grid_size[0]/sfactor), int(grid_size[1]/sfactor), int(grid_size[2]/sfactor)).permute(1,2,3,0)
        
        #Visualize three planes

        # yz = torch.mean(data_im, dim=0)
        # plt.imshow(yz.numpy())
        # plt.show()

        # xz = torch.mean(data_im, dim=1)
        # plt.imshow(xz.numpy())
        # plt.show()


        xy = torch.mean(data_im, dim=2)
        # print("xy", xy.shape)
        # plt.imshow(xy.numpy())
        # plt.show()


        # for i in range(5):
        #     xy = data_im[:,:,10*i, :]
        #     print("xy", xy.shape)
        #     plt.imshow(xy.numpy())
        #     plt.show()

        # xy = torch.mean(data_im, dim=2)
        # print("xy", xy.shape)
        # plt.imshow(xy.numpy())
        # plt.show()

        # Now try indexing into the formed grid with xy, xz and yz coordinates of the source cameras
        print("all_c2w[ref_camera]", all_c2w[ref_camera])
        coords = get_coords(all_c2w[ref_camera], focal, img_size[0], img_size[1])

        # print("coords", coords.shape)
        # print("torch. max min coords", torch.max(coords[:,0]), torch.min(coords[:,0]))
        # print("torch. max min coords", torch.max(coords[:,1]), torch.min(coords[:,1]))
        # print("torch. max min coords", torch.max(coords[:,2]), torch.min(coords[:,2]))

        print("coords", coords.shape)
        print("data_im", data_im.shape)


        # coords, mag_original = contract_samples(coords, order=float('inf'))
        # print("min max coords before", torch.min(coords), torch.max(coords))

        xy_coords = coords.view(-1,3)[:,[0,1]]
        # print("xy, xy coords", xy.shape, xy_coords.shape)
        xy = xy.permute(2,1,0).unsqueeze(0)
        xy_coords = xy_coords.unsqueeze(0).unsqueeze(2)
        #2 is the contraction size of the cube 
        # xy_coords = xy_coords/2

        # print("min max xy_coords after", torch.min(xy_coords), torch.max(xy_coords))

        # yz_coords = coords.view(-1,3)[:,[1,2]]
        # print("xy, xy yz_coords", xy.shape, yz_coords.shape)
        # yz_coords = yz_coords.unsqueeze(0).unsqueeze(2)
        # yz = yz.permute(2,1,0).unsqueeze(0)
        # print("xy, xy coords", xy.shape, yz_coords.shape)

        # xz_coords = coords.view(-1,3)[:,[0,2]]
        # print("xy, xy yz_coords", xy.shape, xz_coords.shape)
        # xz_coords = xz_coords.unsqueeze(0).unsqueeze(2)
        # xz = xz.permute(2,1,0).unsqueeze(0)
        # print("xy, xy coords", xy.shape, xz_coords.shape)


        data_abs = F.grid_sample(xy, xy_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
        # data_abs = F.grid_sample(yz, yz_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
        #data_abs = F.grid_sample(xz, xz_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
        all_imgs = data_abs.squeeze(-1).squeeze(0).permute(1,0).reshape(480,640,65,3).numpy()

        img_1 = all_imgs[:,:,64,:]

        # plt.imshow(img_1)
        # plt.show()

        # print("data_abs", data_abs.shape)

        if draw_camera_grid:
            camera_grid_o3d.colors = o3d.utility.Vector3dVector(data_im.view(-1,3).numpy())
            things_to_draw.append(camera_grid_o3d)

        all_data_im.append(data_im.view(-1,3).unsqueeze(0))
    all_data_im = torch.cat((all_data_im), dim=0)
    
    print("all_data_im", all_data_im.shape)
    data_im = torch.mean(all_data_im, dim=0)
    print("data_im", data_im.shape)
    #draw stuff here
    #don't draw camera grid here
    # world_grid_o3d.paint_uniform_color((0,1,0)) 
    # things_to_draw.append(world_grid_o3d)

    print("data_im", torch.max(data_im), torch.min(data_im))
    if not draw_camera_grid:
        world_grid_o3d.colors = o3d.utility.Vector3dVector(data_im.view(-1,3).numpy())
        things_to_draw.append(world_grid_o3d)

    
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=5)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=np.array([0., 0., 0.]))
    sphere.paint_uniform_color((1, 0, 0))
    # coordinate_frame.paint_uniform_color((0,1,0))
    things_to_draw.append(sphere)
    things_to_draw.append(coordinate_frame)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)

    
    for i, C2W in enumerate(all_c2w):
        if i in ref_cameras:
            s = 0.1
        else:
            s = 0.05
        fig.plot_transform(A2B=C2W, s=s, strict_check=False)

    print("all_c2w[ref_camera]", all_c2w[ref_camera])
    if draw_camera_grid:
        if spherical:
            plot_rays_spherical(all_c2w[ref_camera], focal, img_size[0], img_size[1], fig, RTs=RTs, asset_pose_inv= asset_pose_inv, c2w_first=all_c2w[0])
        else:
            plot_rays(all_c2w[ref_camera], focal, img_size[0], img_size[1], fig, RTs=RTs, asset_pose_inv= asset_pose_inv, c2w_first=all_c2w[0])
    else:
        if spherical:
            plot_rays_spherical(all_c2w[ref_camera], focal, img_size[0], img_size[1], fig, RTs=RTs, asset_pose_inv= asset_pose_inv, c2w_first=all_c2w[0])
        else:
            plot_rays(all_c2w[ref_camera], focal, img_size[0], img_size[1], fig, RTs=RTs, asset_pose_inv= asset_pose_inv)
    
    fig.show()

def inv_transform_c2w(c2w):
    norm_pose_0 = [[ 7.14784828e-01, -3.56641552e-01, 6.01572484e-01, 6.02849754e-01],
        [ 6.99344443e-01,  3.64515616e-01, -6.14854223e-01, -6.16159694e-01],
        [-6.48906216e-18,  8.60194844e-01,  5.09965519e-01, 5.11048288e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
    norm_pose_0 = torch.FloatTensor(norm_pose_0).to(c2w.device)
    new_c2w = norm_pose_0 @ c2w
    return new_c2w

def get_coords(c2w, focal, W, H):
    focal = focal
    directions = get_ray_directions(H, W, focal) # (h, w, 3)
    c2w = torch.FloatTensor(c2w)[:3, :4]
    # rays_o, rays_d = get_rays(directions, c2w)
    rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)
    # ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*1.0))
    # rays_o = rays_o[ids, :]
    # rays_d = rays_d[ids, :]
    print("rays_o", rays_o.shape)

    rays_o = torch.FloatTensor(rays_o)
    rays_d = torch.FloatTensor(rays_d)

    # near = torch.full_like(rays_o[..., -1:], 0.3)
    # far = torch.full_like(rays_o[..., -1:], 8.0)
    near = 0.2
    far = 8.0
    t_vals, coords = sample_along_rays_vanilla(rays_o, rays_d, 64, near, far, randomized = True, lindisp = False )

    return coords

def rot_from_origin(c2w,rotation=10):
    rot = c2w[:3,:3]
    pos = c2w[:3,-1:]
    rot_mat = get_rotation_matrix(rotation)
    pos = torch.mm(rot_mat, pos)
    rot = torch.mm(rot_mat, rot)
    c2w = torch.cat((rot, pos), -1)
    return c2w

def get_rotation_matrix(rotation):
    #if iter_ is not None:
    #    rotation = self.near_c2w_rot * (self.smoothing_rate **(int(iter_/self.smoothing_step_size)))
    #else: 

    phi = (rotation*(np.pi / 180.))
    x = np.random.uniform(-phi, phi)
    y = np.random.uniform(-phi, phi)
    z = np.random.uniform(-phi, phi)
    
    rot_x = torch.Tensor([
                [1,0,0],
                [0,np.cos(x),-np.sin(x)],
                [0,np.sin(x), np.cos(x)]
                ])
    rot_y = torch.Tensor([
                [np.cos(y),0,-np.sin(y)],
                [0,1,0],
                [np.sin(y),0, np.cos(y)]
                ])
    rot_z = torch.Tensor([
                [np.cos(z),-np.sin(z),0],
                [np.sin(z),np.cos(z),0],
                [0,0,1],
                ])
    rot_mat = torch.mm(rot_x, torch.mm(rot_y, rot_z))
    return rot_mat

def get_rays_mvs(H, W, focal, c2w):
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
    ys, xs = ys.reshape(-1), xs.reshape(-1)

    dirs = torch.stack([(xs-W/2)/focal, (ys-H/2)/focal, torch.ones_like(xs)], -1) # use 1 instead of -1
    rays_d = dirs @ c2w[:3,:3].t() # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)
    print("rays_o", rays_o.shape, rays_d.shape)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d

def plot_rays(c2w, focal, W, H, fig, RTs = None, asset_pose_inv= None, c2w_first=None):
    focal = focal
    directions = get_ray_directions(H, W, focal) # (h, w, 3)

    # c2w = np.eye(4)
    # c2w_near = rot_from_origin(torch.FloatTensor(c2w))
    # c2w_near = c2w_near[:3, :4]
    
    c2w = convert_pose(c2w)
    c2w = torch.FloatTensor(c2w)[:3, :4]

    # print("c2w", c2w.shape, c2w_near.shape)
    # print("c2w", c2w, c2w_near)

    
    # rays_o, rays_d = get_rays(directions, c2w)
    rays_o, rays_d = get_rays_mvs(H,W, focal, c2w)
    #rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)
    #rays_o_near, view_dirs, rays_d_near, radii = get_rays(directions, c2w_near, output_view_dirs=True, output_radii=True)

    # print("rays_o", rays_o.shape, rays_o_near.shape)
    ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*1.0))
    
    # print("rays", rays_o.shape, rays_d.shape, rays_o_near.shape, rays_d_near.shape)
    rays_o = torch.FloatTensor(rays_o[ids, :])[:200]
    rays_d = torch.FloatTensor(rays_d[ids, :])[:200]
    near = torch.full_like(rays_o[..., -1:], 0.1)
    far = torch.full_like(rays_o[..., -1:], 2.5)

    t_vals, coords = sample_along_rays_vanilla(rays_o, rays_d, 64, near, far, randomized = True, lindisp = False)
    coords = coords.view(-1,3)

    # rays_o_near = torch.FloatTensor(rays_o_near[ids, :])[:200]
    # rays_d_near = torch.FloatTensor(rays_d_near[ids, :])[:200]
    # near = torch.full_like(rays_o_near[..., -1:], 0.1)
    # far = torch.full_like(rays_d_near[..., -1:], 2.5)

    # t_vals, coords_near = sample_along_rays_vanilla(rays_o_near, rays_d_near, 64, near, far, randomized = True, lindisp = False)
    # coords_near = coords_near.view(-1,3)

    # print("coords", coords.shape, coords_near.shape)
    # print("torch. max min coords", torch.max(coords[:,0]), torch.min(coords[:,0]))
    # print("torch. max min coords", torch.max(coords[:,1]), torch.min(coords[:,1]))
    # print("torch. max min coords", torch.max(coords[:,2]), torch.min(coords[:,2]))
    # # coords, mag_original = contract_samples(coords)
    # coords = inverse_contract_samples(coords, mag_original)
    # coords = coords[:2000,:]
    fig.plot(coords, c=(0.0, 1.0, 0.0))
    #fig.plot(coords_near, c=(1.0, 0.0, 0.0))
    

def plot_rays_spherical(c2w, focal, W, H, fig, RTs = None, asset_pose_inv= None, c2w_first=None):
    focal = focal
    directions = get_ray_directions(H, W, focal) # (h, w, 3)

    # c2w = torch.FloatTensor(c2w)
    # c2w = inv_transform_c2w(c2w)
    # c2w = c2w[:3, :4]

    c2w = torch.FloatTensor(c2w)[:3, :4]
    # rays_o, rays_d = get_rays(directions, c2w)
    rays_o, view_dirs, rays_d, radii = get_rays(directions, c2w, output_view_dirs=True, output_radii=True)
    # ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*1.0))
    # rays_o = rays_o[ids, :]
    # rays_d = rays_d[ids, :]

    far = intersect_sphere(torch.FloatTensor(rays_o), torch.FloatTensor(rays_d))

    rays_o = rays_o[:200]
    rays_d = rays_d[:200]

    near = torch.full_like(rays_o[..., -1:], 1e-4)
    far = intersect_sphere(rays_o, rays_d)


    obj_t_vals, obj_samples = sample_along_rays(
        rays_o=rays_o,
        rays_d=rays_d,
        num_samples=64,
        near = near,
        far = far,
        randomized=True,
        lindisp=False,
        in_sphere=True,
    )

    bg_t_vals, bg_samples, bg_samples_linear = sample_along_rays(
        rays_o=rays_o,
        rays_d=rays_d,
        num_samples=64,
        near=near,
        far=far,
        randomized=True,
        lindisp=False,
        in_sphere=False,
    )

    print("torch. max min coords", torch.max(obj_samples[:,0]), torch.min(obj_samples[:,0]))
    print("torch. max min coords", torch.max(obj_samples[:,1]), torch.min(obj_samples[:,1]))
    print("torch. max min coords", torch.max(obj_samples[:,2]), torch.min(obj_samples[:,2]))

    print("torch. max min coords", torch.max(bg_samples_linear[:,0]), torch.min(bg_samples_linear[:,0]))
    print("torch. max min coords", torch.max(bg_samples_linear[:,1]), torch.min(bg_samples_linear[:,1]))
    print("torch. max min coords", torch.max(bg_samples_linear[:,2]), torch.min(bg_samples_linear[:,2]))

    coords = torch.cat((obj_samples, bg_samples[:,:,:3]), dim=0)
    #coords, mag_original = contract_samples(coords)
    #coords = inverse_contract_samples(coords, mag_original)
    # print("coords", coords.shape)
    fig.plot(obj_samples.view(-1,3), c=(0.0, 1.0, 0.0))
    fig.plot(bg_samples[:,:,:3].view(-1,3), c=(1.0, 0.0, 0.0))
    fig.plot(bg_samples_linear[:,:,:3].view(-1,3), c=(1.0, 0.0, 0.0))


def draw_bboxes_mpl_glow(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    color_ground = color
    n_lines = 8
    diff_linewidth = 1.05
    alpha_value = 0.03
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
    for i, j in zip(range(4), range(4, 8)):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        for n in range(1, n_lines+1):
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]], color=color_ground, linewidth=1, marker='o')
            plt.plot([img_pts[i][0], img_pts[j][0]], [img_pts[i][1], img_pts[j][1]],
                marker='o',
                linewidth=1+(diff_linewidth*n),
                alpha=alpha_value,
                color=color_ground)
    return img

def get_camera_frustum(img_size, focal, C2W, frustum_length=1, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / focal) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / focal) * 2.)
    # print("hfov", hfov, vfov)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    #print("frustum_points", frustum_points)
    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    # C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    #print("frustum_points afters", frustum_points)
    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

import colorsys
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors


