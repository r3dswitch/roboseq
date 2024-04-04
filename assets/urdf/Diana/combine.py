import urdfpy
import os


def load_urdf_and_combine(
    main_urdf_path, sub_urdf_path, link_name, xyz=[0, 0, 0], rpy=[0, 0, 0]
):
    main_robot = urdfpy.URDF.load(main_urdf_path)
    sub_robot = urdfpy.URDF.load(sub_urdf_path)

    # Modify the pose of the sub_robot link relative to the main_robot
    pose = urdfpy.Pose(xyz=xyz, rpy=rpy)

    # Add the sub_robot as a new link to the main_robot
    main_robot.links.append(
        urdfpy.Link(
            name=link_name,
            visual=sub_robot.visual,
            collision=sub_robot.collision,
            inertial=sub_robot.inertial,
            origin=pose,
        )
    )

    return main_robot


if __name__ == "__main__":
    main_urdf_path = "/home/soumya.mondal/Desktop/Projects/Arnie/arnie/models/diana_v2_with_shadowhand/diana_v2.urdf"
    sub_urdf_path = "/home/soumya.mondal/Desktop/Projects/Arnie/arnie/models/diana_v2_with_shadowhand/shadowhand.urdf"
    combined_urdf_path = "/home/soumya.mondal/Desktop/Projects/Arnie/arnie/models/diana_v2_with_shadowhand/diana_with_hand.urdf"

    print(os.path.isfile(sub_urdf_path))
    # Load URDF files and combine them
    combined_robot = load_urdf_and_combine(
        main_urdf_path,
        sub_urdf_path,
        link_name="shadowhand",
        xyz=[0, 0, 0],
        rpy=[0, 0, 0],
    )

    # Save the combined URDF model to a new file
    combined_robot.export_xml(combined_urdf_path)
