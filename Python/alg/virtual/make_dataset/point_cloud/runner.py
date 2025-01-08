import os

from alg.virtual.make_dataset.point_cloud.annotate_img import annotate_img_pixel
from alg.virtual.make_dataset.point_cloud.annotate_pcd import reconstruct_point_cloud

object_dirs = [
    "object_1",
    "object_2",
    "object_3",
]

for i in range(380, 1000):
    os.system(
        r"C:\Users\ryout\AppData\Local\ov\pkg\isaac-sim-4.2.0\isaac-sim.bat --no_window --/omni/replicator/script=C:\Users\ryout\Documents\main_desk\omniverse_driver\alg\virtual\make_dataset\camera\world.py"
    )

    os.system(
        r"C:\Users\ryout\AppData\Local\ov\pkg\isaac-sim-4.2.0\isaac-sim.bat --no_window --/omni/replicator/script=C:\Users\ryout\Documents\main_desk\omniverse_driver\alg\virtual\make_dataset\camera\object_1.py"
    )

    os.system(
        r"C:\Users\ryout\AppData\Local\ov\pkg\isaac-sim-4.2.0\isaac-sim.bat --no_window --/omni/replicator/script=C:\Users\ryout\Documents\main_desk\omniverse_driver\alg\virtual\make_dataset\camera\object_2.py"
    )

    os.system(
        r"C:\Users\ryout\AppData\Local\ov\pkg\isaac-sim-4.2.0\isaac-sim.bat --no_window --/omni/replicator/script=C:\Users\ryout\Documents\main_desk\omniverse_driver\alg\virtual\make_dataset\camera\object_3.py"
    )
    for j, object_dir in enumerate(object_dirs):
        annotate_img_pixel(object_dir, annotation_value=j + 1)

    reconstruct_point_cloud(
        "world", "object_1", "object_2", "object_3", f"data_{i}.ply"
    )
