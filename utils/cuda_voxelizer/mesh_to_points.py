import os
import meshio
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_path)
ply_path = r"H:\Projects\FS5\cuda_voxelizer\1.5x1.5_closed_bottom.ply"
cuda_voxelizer_path = r".\cuda_voxelizer.exe"
voxelSize = 1

meshio_mesh = meshio.read(ply_path)
max_size = max(meshio_mesh.points.max(axis=0) - meshio_mesh.points.min(axis=0))
mesh_centre = (meshio_mesh.points.max(axis=0) + meshio_mesh.points.min(axis=0))/2
max_size = int(np.ceil(max_size / voxelSize))

cmd_str = "{} -f {} -s {} -solid -o obj_points".format(cuda_voxelizer_path, ply_path, str(max_size))
print(cmd_str)
os.system(cmd_str)

pointcloud_filename = ply_path+"_{}_pointcloud.obj".format(str(max_size))
print("reading "+os.path.basename(pointcloud_filename)+"...")
points = np.loadtxt(pointcloud_filename, usecols=(1,2,3)) * voxelSize
points_centre = (points.max(axis=0) + points.min(axis=0))/2

csv_filename = ply_path+"_voxelSize={}.csv".format(str(voxelSize))
print("generating "+os.path.basename(csv_filename)+"...")
points = points + mesh_centre - points_centre
np.savetxt(csv_filename, points, fmt='%.8f', delimiter=',')
print(csv_filename)