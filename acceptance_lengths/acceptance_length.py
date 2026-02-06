import json
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드
jsonl_file_path = 'results/coco/hivis_d6.jsonl' 

# 데이터를 저장할 리스트
all_acceptance_lengths = []

try:
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue # 빈 줄 무시
            try:
                data = json.loads(line)
                # acceptance_length 추출 (choices 리스트의 첫 번째 요소 내부에 있음)
                acc_len = data['choices'][0]['acceptance_length']
                all_acceptance_lengths.append(acc_len)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(f"Error parsing line: {e}")
                continue
except FileNotFoundError:
    print(f"'{jsonl_file_path}' 파일을 찾을 수 없습니다. 같은 폴더에 데이터 파일을 위치시켜주세요.")
    exit()

# 2. 데이터 가공
if not all_acceptance_lengths:
    print("데이터가 없습니다.")
    exit()

# 가장 긴 리스트의 길이 찾기
max_len = max(len(seq) for seq in all_acceptance_lengths)

# 데이터를 행렬로 변환 (부족한 부분은 NaN으로 채움)
padded_data = np.full((len(all_acceptance_lengths), max_len), np.nan)

for i, seq in enumerate(all_acceptance_lengths):
    padded_data[i, :len(seq)] = seq

# 평균 및 표준편차 계산 (NaN 무시)
means = np.nanmean(padded_data, axis=0)
stds = np.nanstd(padded_data, axis=0)
phases = np.arange(max_len)

# 3. 시각화
plt.figure(figsize=(12, 6))

# 개별 데이터 그리기 (투명도 낮게 설정하여 분포 확인)
for seq in all_acceptance_lengths:
    plt.plot(seq, color='gray', alpha=0.1, linewidth=1)

# 평균선 그리기
plt.plot(phases, means, color='red', linewidth=2, label='Average Acceptance Length')

# 표준편차 영역 칠하기
plt.fill_between(phases, means - stds, means + stds, color='red', alpha=0.1, label='Standard Deviation')

# 그래프 스타일 설정
plt.title('Acceptance Length per Phase', fontsize=16)
plt.xlabel('Phase (Turn Index)', fontsize=12)
plt.ylabel('Acceptance Length', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()

# 변경된 부분: 화면 출력 대신 파일로 저장
# 원하는 파일명과 확장자(.png, .pdf, .svg 등)를 지정하세요.
output_filename = 'acceptance_length_plot.png'
plt.savefig(output_filename, dpi=300) # dpi=300은 고해상도 설정을 위함

print(f"그래프가 '{output_filename}' 파일로 저장되었습니다.")
