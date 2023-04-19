import open3d as o3d
import numpy as np
import trimesh as tm
model = '/home/zubair/srn_car_model/02958343/1c53bc6a3992b0843677ee89898ae463/model_normalized.obj'
from pyvista import themes

my_theme = themes.DefaultTheme()
# my_theme.color = 'black'
my_theme.lighting = True
my_theme.background = 'white'
# mesh = o3d.io.read_triangle_mesh(model)
# mesh.compute_vertex_normals()

# # Fit to unit cube.
# mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
#             center=mesh.get_center())
# print('Displaying input mesh ...')
# o3d.visualization.draw([mesh, *unitcube])

# voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds( mesh, voxel_size=0.01, min_bound=[-1,-1, -1], max_bound=[1,1,1])(
#     mesh, voxel_size=0.005)
# print('Displaying voxel grid ...')
# o3d.visualization.draw([voxel_grid, *unitcube])

import pyvista as pv
pv.set_plot_theme(my_theme)
mesh = pv.read(model)


# cpos = [
#     (7.656346967151718, -9.802071079151158, -11.021236183314311),
#     (0.2224512272564101, -0.4594554282112895, 0.5549738359311297),
#     (-0.6279216753504941, -0.7513057097368635, 0.20311105371647392),
# ]
# voxels = pv.voxelize(mesh, density=0.01, check_surface=False)


# Initialize the plotter object with four sub plots
# pl = pv.Plotter(shape=(2, 2))

# # Second subplot show the voxelized repsentation of the mesh with voxel size of 0.01. We remove the surface check as the mesh has small imperfections
# pl.subplot(0, 1)
# voxels = pv.voxelize(mesh, density=0.005, check_surface=False)
texture = pv.read_texture('/home/zubair/srn_car_model/02958343/1c53bc6a3992b0843677ee89898ae463/model_normalized.mtl')

p = pv.Plotter()
# p.add_mesh(voxels, scalars='vtkOriginalCellIds')
p.add_mesh(mesh, texture=texture)
p.add_bounding_box()


def my_plane_func(normal, origin):
    slc = mesh.slice(normal=normal, origin=origin)
    arrows = slc.glyph(orient='vectors', scale="scalars", factor=0.01)
    p.add_mesh(arrows, name='arrows')

p.add_plane_widget(my_plane_func)
p.show()

# voxels.plot(scalars='vtkOriginalCellIds')
# We add the voxels as a new mesh, add color and show their edges
# pl.add_mesh(voxels, color=True, show_edges=True)

# # Third subplot shows the voxel representation using cones 
# pl.subplot(1,0)
# glyphs = voxels.glyph(factor=1e-3, geom=pv.Cone())
# pl.add_mesh(glyphs)

# # Forth subplot shows the voxels together with a contour showing the per voxel distance to the mesh
# pl.subplot(1,1)
# # Calculate the distance between the voxels and the mesh. Add the results as a new scalar to the voxels
# voxels.compute_implicit_distance(mesh, inplace=True)
# # Create a contour representing the calculated distance
# contours = voxels.contour(6, scalars="implicit_distance")
# # Add the voxels and the contour with different opacity to show both
# pl.add_mesh(voxels, opacity=0.25, scalars="implicit_distance")
# pl.add_mesh(contours, opacity=0.5, scalars="implicit_distance")


# # Link all four views so all cameras are moved at the same time
# pl.link_views()
# # Set camera start position
# pl.camera_position = 'xy'
# # Show everything
# pl.show()