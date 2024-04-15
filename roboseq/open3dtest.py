import open3d as o3d
import numpy as np
import time

if __name__ == "__main__":
    points = o3d.io.read_point_cloud("/home/soumya_mondal/Desktop/Projects/Thesis/roboseq/pcd.ply")

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(points)
    vis.add_geometry(frame)

    start_time = time.time()

    while time.time()-start_time < 10:
        vis.update_geometry(points)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()
