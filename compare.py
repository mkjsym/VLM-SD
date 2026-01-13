import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_jsonl(file_path, model_label):
    """
    JSONL 파일을 읽어서 필요한 메트릭을 추출하여 DataFrame으로 변환합니다.
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            choice = entry['choices'][0]
            
            # 기본 정보 추출
            q_id = entry['question_id']
            
            # 메트릭 추출 (리스트인 경우 합계 또는 평균 계산)
            new_tokens = sum(choice.get('new_tokens', [0]))
            wall_time = sum(choice.get('wall_time', [0]))
            
            # Acceptance Length (Speculative Decoding 효율성 지표)
            acc_len_list = choice.get('acceptance_length', [])
            mean_acc_len = sum(acc_len_list) / len(acc_len_list) if acc_len_list else 0
            
            # TPS 계산 (ZeroDivisionError 방지)
            tps = new_tokens / wall_time if wall_time > 0 else 0
            
            data.append({
                'question_id': q_id,
                'model': model_label,
                'new_tokens': new_tokens,
                'wall_time': wall_time,
                'tps': tps,
                'mean_acceptance_length': mean_acc_len
            })
            
    return pd.DataFrame(data)

def compare_results(file1, file2, label1="Model A", label2="Model B"):
    """
    두 개의 결과 파일을 비교하고 요약 통계 및 그래프를 출력합니다.
    """
    # 데이터 로드
    df1 = parse_jsonl(file1, label1)
    df2 = parse_jsonl(file2, label2)
    
    # 데이터 합치기
    df_all = pd.concat([df1, df2])
    
    # 1. 요약 통계 출력
    print(f"=== Performance Summary: {label1} vs {label2} ===")
    summary = df_all.groupby('model')[['tps', 'wall_time', 'mean_acceptance_length']].mean()
    print(summary)
    print("\n" + "="*50 + "\n")

    # 2. Question ID 기준 1:1 비교 (Speedup 계산)
    # 두 모델의 결과를 question_id로 병합
    merged_df = pd.merge(df1, df2, on='question_id', suffixes=(f'_{label1}', f'_{label2}'))
    
    # Speedup 계산 (Model B TPS / Model A TPS)
    merged_df['speedup_ratio'] = merged_df[f'tps_{label2}'] / merged_df[f'tps_{label1}']
    
    avg_speedup = merged_df['speedup_ratio'].mean()
    print(f"=== Speedup Analysis ===")
    print(f"Average Speedup ({label2} / {label1}): {avg_speedup:.2f}x")
    print(f"Total Samples Compared: {len(merged_df)}")
    
    # 3. 시각화
    plt.figure(figsize=(12, 5))
    
    # TPS Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='model', y='tps', data=df_all)
    plt.title('Tokens Per Second (TPS) Distribution')
    plt.ylabel('Tokens / Sec')
    
    # Acceptance Length Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x='model', y='mean_acceptance_length', data=df_all)
    plt.title('Mean Acceptance Length Distribution')
    plt.ylabel('Avg Accepted Tokens per Step')
    
    plt.tight_layout()
    plt.show()

    # Speedup Histogram (만약 비교 데이터가 있다면)
    if not merged_df.empty:
        plt.figure(figsize=(8, 4))
        sns.histplot(merged_df['speedup_ratio'], kde=True, bins=20)
        plt.axvline(x=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
        plt.axvline(x=avg_speedup, color='g', linestyle='--', label=f'Mean Speedup ({avg_speedup:.2f}x)')
        plt.title(f'Speedup Distribution ({label2} over {label1})')
        plt.xlabel('Speedup Ratio')
        plt.legend()
        plt.show()

# === 사용 예시 ===
# 실제 파일 경로로 교체하여 실행하세요.
# 파일이 없다면 예시 데이터를 생성하여 테스트할 수 있습니다.

file_path_a = "/home/youngmin/workspace/VLM-SD/vispec.jsonl"
file_path_b = "/home/youngmin/workspace/VLM-SD/aircache.jsonl"
compare_results(file_path_a, file_path_b, label1="Baseline", label2="Ours (Speculative)")
