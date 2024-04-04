import numpy as np
import open3d as o3d


class PointcloudVisualizer:
    def __init__(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        # self.vis.register_key_callback(self.vis.get_key(), self.vis.update_renderer())

    def add_geometry(self, cloud):
        self.vis.add_geometry(cloud)

    def update(self, cloud):
        # Your update routine
        self.vis.update_geometry(cloud)
        self.vis.update_renderer()
        self.vis.poll_events()


if __name__ == "__main__":
    visualizer = PointcloudVisualizer()
    cloud = o3d.io.read_point_cloud(
        "/home/soumya.mondal/Desktop/Projects/DexterousHands/assets/laptop.ply"
    )
    visualizer.add_geometry(cloud)
    visualizer.update(cloud)
