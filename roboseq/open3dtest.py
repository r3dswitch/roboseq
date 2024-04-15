import open3d as o3d
import numpy as np
import time

if __name__ == "__main__":
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0)
    points = o3d.io.read_point_cloud("/home/soumya_mondal/Desktop/Projects/Thesis/roboseq/pcd.ply")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.add_geometry(frame)
    # vis.add_geometry(points)

    start_time = time.time()

    while time.time()-start_time < 50:
        points = o3d.io.read_point_cloud("/home/soumya_mondal/Desktop/Projects/Thesis/roboseq/pcd.ply")
        vis.add_geometry(frame)
        vis.add_geometry(points)
        vis.update_geometry(points)
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(points)
        time.sleep(0.01)

    vis.destroy_window()
