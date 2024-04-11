import open3d as o3d
import numpy as np

# Define vertices of the cube
vertices = np.array([
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5]
])

# Define the indices of the cube's faces
faces = np.array([
    [0, 1, 3],
    [0, 2, 3],
    [1, 5, 3],
    [5, 7, 3],
    [4, 5, 7],
    [4, 6, 7],
    [0, 4, 2],
    [4, 6, 2],
    [0, 1, 4],
    [1, 5, 4],
    [2, 3, 6],
    [3, 7, 6]
])

# Create Open3D mesh object
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])
