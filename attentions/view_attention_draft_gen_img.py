import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def plot_transposed_heatmap(data, row_indices, col_indices, title, save_path):
    """
    [수정됨] X, Y축이 반전된 히트맵을 그리는 함수
    - data: (N_Image, N_Gen) 형태의 2D 배열 (Transposed)
    - row_indices: Y축 라벨 (이미지 토큰 인덱스)
    - col_indices: X축 라벨 (생성된 텍스트 토큰 인덱스)
    """
    if data.size == 0:
        print(f"Skipping empty data slice: {title}")
        return

    # 1. 데이터 스케일링 (1/3승)
    data_to_plot = np.power(data, 1/3)

    # 2. 그림 크기 동적 계산 (가로로 긴 형태 지향)
    # y_len: 이미지 토큰 개수 (Rows), x_len: 생성 토큰 개수 (Cols)
    y_len, x_len = data.shape
    
    # 셀 하나당 크기 배정
    cell_width = 0.5   # 생성 토큰(X축) 간격
    cell_height = 0.4  # 이미지 토큰(Y축) 간격
    
    width = max(12, x_len * cell_width)
    height = max(8, y_len * cell_height)
    
    # 너무 크면 제한
    if width > 60: width = 60
    if height > 40: height = 40

    plt.figure(figsize=(width, height))

    # 3. 히트맵 그리기
    mask = (data == 0)

    heatmap_args = {
        "data": data_to_plot,
        "cmap": "viridis",
        "cbar_kws": {'label': 'Attention Score ^ (1/3)', 'shrink': 0.8},
        "mask": mask,
        "xticklabels": col_indices, # X축: 생성 토큰 인덱스
        "yticklabels": row_indices  # Y축: 이미지 토큰 인덱스
    }
    
    sns.heatmap(**heatmap_args)

    # 4. 축 설정
    plt.xlabel(f"Generated Token Indices [Count: {x_len}]", fontsize=14, labelpad=15)
    plt.ylabel(f"Image Token Indices (Active) [Count: {y_len}]", fontsize=14, labelpad=15)
    
    # X축 (Generated Index) 라벨 간격 조정
    if x_len > 50:
        step = 5
        plt.xticks(ticks=np.arange(0, x_len, step) + 0.5, labels=col_indices[::step], rotation=90, fontsize=9)
    else:
        plt.xticks(rotation=90, fontsize=9)
        
    # Y축 (Image Index) 라벨 간격 조정
    if y_len > 50:
         step = 5
         plt.yticks(ticks=np.arange(0, y_len, step) + 0.5, labels=row_indices[::step], rotation=0, fontsize=9)
    else:
        plt.yticks(rotation=0, fontsize=9)

    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # -----------------------------------------------------------
    # 1. 파일 로드
    # -----------------------------------------------------------
    attn_file = "attentions/attention_maps/q1/draft_attn_dog.npy"
    token_file = "attentions/attention_maps/q1/generated_tokens_dog.json"

    if not os.path.exists(attn_file):
        print(f"Error: {attn_file} not found.")
        return

    print(f"Loading Draft Model Attention Map from {attn_file}...")
    attn_map = np.load(attn_file)
    total_len = attn_map.shape[0]

    # -----------------------------------------------------------
    # 2. 인덱스 범위 계산
    # -----------------------------------------------------------
    gen_token_texts = []
    if os.path.exists(token_file):
        with open(token_file, 'r', encoding='utf-8') as f:
            token_info = json.load(f)
            gen_token_texts = [item['token_text'] for item in token_info]
    
    len_gen = len(gen_token_texts)
    
    LEN_SYSTEM = 35
    LEN_INSTRUCTION = 17 

    idx_system_end = LEN_SYSTEM
    idx_gen_start = total_len - len_gen
    idx_instr_start = idx_gen_start - LEN_INSTRUCTION
    
    idx_image_start = idx_system_end
    idx_image_end = idx_instr_start

    image_range = (idx_image_start, idx_image_end)
    gen_range = (idx_gen_start, total_len)

    print(f"\n[Ranges] Image: {image_range}, Generated: {gen_range}")

    # -------------------------------------------------------
    # 3. 데이터 처리
    # -------------------------------------------------------
    print("\nProcessing Generated vs Image Attention...")

    # (1) Raw Data 슬라이싱 [Rows: Generated, Cols: Image]
    raw_attn = attn_map[gen_range[0]:gen_range[1], image_range[0]:image_range[1]]
    
    if raw_attn.size == 0:
        return

    # (2) 유효한 Image Token(Column) 필터링
    col_max_values = raw_attn.max(axis=0) 
    valid_col_indices = np.where(col_max_values > 0)[0] 

    if len(valid_col_indices) == 0:
        print("No attention found.")
        return

    # 필터링 수행: (Generated, Filtered_Image)
    filtered_attn = raw_attn[:, valid_col_indices]

    # (3) [핵심 변경] 데이터 전치 (Transpose) -> (Filtered_Image, Generated)
    # 이제 행(Row)이 이미지, 열(Column)이 생성된 텍스트가 됩니다.
    transposed_attn = filtered_attn.T

    # (4) 라벨 준비
    # Y축 라벨: 살아남은 Image Token의 절대 인덱스
    y_axis_indices = valid_col_indices + image_range[0]
    
    # X축 라벨: Generated Token의 절대 인덱스 (텍스트 대신 숫자 사용)
    x_axis_indices = np.arange(gen_range[0], gen_range[1])

    print(f"Shape after transpose: {transposed_attn.shape}")
    print(f" - Rows (Image): {len(y_axis_indices)}")
    print(f" - Cols (Gen): {len(x_axis_indices)}")

    # -------------------------------------------------------
    # 4. 시각화 실행
    # -------------------------------------------------------
    plot_title = "Transposed Attention: Active Image Tokens (Row) vs Generated Indices (Col)"
    save_filename = "draft_attn_Transposed_Image_vs_Gen.png"

    plot_transposed_heatmap(
        data=transposed_attn,
        row_indices=y_axis_indices,  # Y축: Image Index
        col_indices=x_axis_indices,  # X축: Gen Index
        title=plot_title,
        save_path=save_filename
    )

    print("\nVisualization complete.")

if __name__ == "__main__":
    main()
