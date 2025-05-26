import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

def parse_log_file(log_file):
    results = {}
    current_video_id = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # 提取视频ID
            if "Start to process" in line:
                current_video_id = line.split()[-1].strip()
                results[current_video_id] = {
                    "video_id": current_video_id,
                    "answer": None,
                    "label": None,
                    "corr": None,
                    "count_frame": None,
                    "get_ans_step": None
                }
            
            # 提取结果信息
            if current_video_id and "Finished video:" in line:
                # 格式例如: "Finished video: video_id/answer/label (line xxx)"
                parts = line.split(":")[-1].strip().split("/")
                if len(parts) >= 3:
                    # 处理可能带有行号的标签
                    answer = parts[1].strip()
                    label = parts[2].split('(')[0].strip()  # 移除可能的行号信息
                    try:
                        results[current_video_id]["answer"] = int(answer)
                        results[current_video_id]["label"] = int(label)
                        results[current_video_id]["corr"] = 1 if answer == label else 0
                    except ValueError as e:
                        print(f"Warning: 无法解析答案或标签: {line}")
                        continue
                    
            # 提取使用的帧数
            if "frame_descriptions" in line:
                if current_video_id and "count_frame" not in results[current_video_id]:
                    results[current_video_id]["count_frame"] = 0
                results[current_video_id]["count_frame"] = results[current_video_id].get("count_frame", 0) + 1
            
            # 提取答案获取步骤
            if "get_ans_step" not in results.get(current_video_id, {}):
                if "1_s_r" in line:
                    results[current_video_id]["get_ans_step"] = "1_s_r"
                elif "2_s_r" in line:
                    results[current_video_id]["get_ans_step"] = "2_s_r"
                elif "3_s_r" in line:
                    results[current_video_id]["get_ans_step"] = "3_s_r"
                elif "4_s_r" in line:
                    results[current_video_id]["get_ans_step"] = "4_s_r"
                elif "5_s_r" in line:
                    results[current_video_id]["get_ans_step"] = "5_s_r"
                elif "final_direct_qa" in line:
                    results[current_video_id]["get_ans_step"] = "final_direct_qa"
                elif "post_s_r" in line:
                    results[current_video_id]["get_ans_step"] = "post_s_r"
    
    # 移除未完成的视频记录
    results = {k: v for k, v in results.items() if all(v.values())}
    return results

def main():
    parser = argparse.ArgumentParser(description='Convert log file to json format')
    parser.add_argument('--log_file', type=str, required=True, help='Input log file path')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory path')
    parser.add_argument('--dataset', type=str, default='egoschema_subset', help='Dataset name for output path')
    args = parser.parse_args()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_file = os.path.join(output_dir, f"{timestamp}.json")
    
    # 解析日志文件
    results = parse_log_file(args.log_file)
    
    # 保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Converted {len(results)} video results")
    print(f"Output saved to: {output_file}")
    
    # 打印统计信息
    total = len(results)
    if total > 0:
        correct = sum(1 for v in results.values() if v['corr'] == 1)
        print(f"\nStatistics:")
        print(f"Total videos processed: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {(correct/total)*100:.2f}%")
        
        # 按步骤统计
        step_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for v in results.values():
            step = v['get_ans_step']
            step_stats[step]['total'] += 1
            if v['corr'] == 1:
                step_stats[step]['correct'] += 1
        
        print("\nResults by step:")
        for step, stats in step_stats.items():
            acc = (stats['correct'] / stats['total']) * 100
            print(f"{step}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
    
    print(f"\ntimestamp: {timestamp}")

if __name__ == "__main__":
    main() 