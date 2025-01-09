import os
import time
import matplotlib.pyplot as plt

from .annotate_img import annotate_single_prefab_images
from .annotate_pcd import reconstruct_point_cloud

def process_directories(base_path):
    i = 0
    times = []
    while i < 10:
        dir_name = f"iteration_{i}"
        dir_path = os.path.join(base_path, dir_name)
        while not os.path.exists(dir_path):
            print(f"Directory {dir_name} not found. Waiting for directory to be created...")
            time.sleep(1)
        if not is_processed(dir_path):
            start_time = time.time()
            time.sleep(5)
            annotate_single_prefab_images(dir_path)
            reconstruct_point_cloud(f"pcd_{i}.ply", dir_path)
            mark_as_processed(dir_path)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Processed directory {dir_name}")
        i += 1

    plot_times(times)

def is_processed(directory):
    return os.path.exists(os.path.join(directory, 'processed.txt'))

def mark_as_processed(directory):
    with open(os.path.join(directory, 'processed.txt'), 'w') as f:
        f.write('processed')

def plot_times(times):
    plt.plot(range(1, len(times) + 1), times)
    plt.xlabel('Number of Directories Processed')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time per Directory')
    plt.show()

if __name__ == "__main__":
    base_path = '../data_images'
    process_directories(base_path)