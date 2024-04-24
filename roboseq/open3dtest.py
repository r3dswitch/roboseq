import open3d as o3d
import numpy as np
import time

if __name__ == "__main__":
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)
    points = o3d.io.read_point_cloud("pcd.ply")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(frame)
    vis.add_geometry(points)

    start_time = time.time()

    while time.time()-start_time < 5000:
        points = o3d.io.read_point_cloud("pcd.ply")
        vis.add_geometry(points, reset_bounding_box=False)
        vis.update_geometry(points)
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(points, reset_bounding_box=False)

    vis.destroy_window()
