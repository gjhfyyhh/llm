import json
import random
import os

# 获取项目根目录的路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 读取原始数据集
subset_anno_path = os.path.join(root_dir, 'data', 'egoschema', 'subset_anno.json')
lavila_subset_path = os.path.join(root_dir, 'data', 'egoschema', 'lavila_subset.json')

print(f"Reading from {subset_anno_path}")
with open(subset_anno_path, 'r') as f:
    subset_anno = json.load(f)

print(f"Reading from {lavila_subset_path}")
with open(lavila_subset_path, 'r') as f:
    lavila_subset = json.load(f)

# 随机选择50个样本
selected_ids = random.sample(list(subset_anno.keys()), 30)

# 创建新的数据集
miniset_anno = {vid_id: subset_anno[vid_id] for vid_id in selected_ids}
lavila_miniset = {vid_id: lavila_subset[vid_id] for vid_id in selected_ids}

# 确保输出目录存在
output_dir = os.path.join(root_dir, 'data', 'egoschema')
os.makedirs(output_dir, exist_ok=True)

# 保存新的数据集
miniset_anno_path = os.path.join(output_dir, 'miniset_anno.json')
lavila_miniset_path = os.path.join(output_dir, 'lavila_miniset.json')

print(f"Saving to {miniset_anno_path}")
with open(miniset_anno_path, 'w') as f:
    json.dump(miniset_anno, f, indent=4)

print(f"Saving to {lavila_miniset_path}")
with open(lavila_miniset_path, 'w') as f:
    json.dump(lavila_miniset, f, indent=4)

print(f"Created miniset with {len(selected_ids)} samples") 