from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
from functools import reduce

def extract_all_loss_scalars_from_logs(root_dir, output_dir="exported_csvs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个子目录（即每个实验日志文件夹）
    for subdir in os.listdir(root_dir):
        subpath = os.path.join(root_dir, subdir)
        if not os.path.isdir(subpath):
            continue  # 忽略非目录项

        # 查找该目录下的 .tfevents 文件
        event_file = None
        for f in os.listdir(subpath):
            if "tfevents" in f:
                event_file = os.path.join(subpath, f)
                break

        if not event_file or not os.path.isfile(event_file):
            print(f"[跳过] 未找到 .tfevents 文件: {subpath}")
            continue

        print(f"[处理中] {subdir} --> {event_file}")

        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            scalar_tags = ea.Tags().get('scalars', [])
            # loss_tags = [tag for tag in scalar_tags if 'loss' in tag.lower()]

            if not scalar_tags:
                print(f"[跳过] 无 loss 标签: {subdir}")
                continue

            dfs = []
            for tag in scalar_tags:
                events = ea.Scalars(tag)
                df = pd.DataFrame(events)[['step', 'value']]
                df = df.rename(columns={'value': tag.replace("/", "_")})
                dfs.append(df)

            merged_df = reduce(lambda left, right: pd.merge(left, right, on='step', how='outer'), dfs)
            merged_df = merged_df.sort_values(by='step')

            output_csv_path = os.path.join(output_dir, f"{subdir}.csv")
            merged_df.to_csv(output_csv_path, index=False)
            print(f"[完成] 已保存: {output_csv_path}")
        except Exception as e:
            print(f"[错误] 处理 {subdir} 失败: {e}")

# 使用方法
logs_root = "/data-shared/server01/data1/haochuan/CharacterRecords2025May-03/Logs/"
extract_all_loss_scalars_from_logs(logs_root)