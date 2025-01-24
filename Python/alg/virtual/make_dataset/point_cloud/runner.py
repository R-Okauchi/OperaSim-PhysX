import os
import time
import datetime
import matplotlib.pyplot as plt
import shutil
import csv

from .annotate_img import annotate_single_prefab_images
from .annotate_pcd import reconstruct_point_cloud

def process_directories(base_path, output_dir):
    i = 0
    times = []
    start_time = time.time()  # 全体の開始時間を記録
    while i < 1000:
        dir_name = f"iteration_{i}"
        dir_path = os.path.join(base_path, dir_name)
        while not os.path.exists(dir_path):
            print(f"Directory {dir_name} not found. Waiting for directory to be created...")
            time.sleep(1)
        time.sleep(5)
        
        # annotate_single_prefab_imagesにリトライ機能を追加
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                annotate_single_prefab_images(dir_path)
                reconstruct_point_cloud(f"pcd_{i}.ply", dir_path, output_dir)
                success = True
                break  # 成功したらループを抜ける
            except Exception as e:
                print(f"Failed to process {dir_name} on attempt {attempt + 1}. Error: {e}")

        if success:
            print(f"Processed directory {dir_name} after {attempt + 1} attempts.")
            shutil.rmtree(dir_path)
            end_time = time.time()
            times.append(end_time - start_time)  # 全体の経過時間を記録
            i += 1
        else:
            print(f"Failed to process {dir_name} after {max_retries} attempts.")
            error_dir_path = f"{dir_path}_error"
            os.rename(dir_path, error_dir_path)

    plot_times(times)

def plot_times(times):
    plt.plot(range(1, len(times) + 1), times)
    plt.xlabel('Number of Directories Processed')
    plt.ylabel('Cumulative Processing Time (seconds)')
    plt.title('Cumulative Processing Time per Directory')
    
    # 現在の日時を取得してファイル名に追加
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cumulative_processing_times_{current_time}.png"
    plt.savefig(filename)
    plt.show()

    # 時間情報をCSVファイルに保存
    csv_filename = f"processing_times_{current_time}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Directory Index', 'Cumulative Processing Time (seconds)'])
        for index, time in enumerate(times, start=1):
            writer.writerow([index, time])

if __name__ == "__main__":
    base_path = '../data_images'
    output_dir = '../dataset'
    process_directories(base_path, output_dir)