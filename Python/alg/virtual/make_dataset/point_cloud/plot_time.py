import csv
import matplotlib.pyplot as plt
import datetime

def plot_from_csv(csv_filename):
    times = []
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダーをスキップ
        for row in reader:
            times.append(float(row[1]))

    plt.plot(range(1, len(times) + 1), times)
    plt.xlabel('Number of Directories Processed')
    plt.ylabel('Cumulative Processing Time (seconds)')
    plt.title('Cumulative Processing Time per Directory')
    
    # 現在の日時を取得してファイル名に追加
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cumulative_processing_times_from_csv_{current_time}.png"
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    csv_filename = 'processing_times_YYYYMMDD_HHMMSS.csv'  # 実際のCSVファイル名に置き換えてください
    plot_from_csv(csv_filename)
    