import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def plot_general_heatmap(attn_data, x_range, y_range, title, save_path, 
                          x_token_labels=None, y_token_labels=None, 
                          x_axis_name="Key", y_axis_name="Query"):
    """
    [범용] Attention Map을 시각화합니다.
    """
    # 1. 데이터 슬라이싱
    slice_data = attn_data[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    
    if slice_data.size == 0:
        print(f"Skipping empty data slice: {title}")
        return

    # 데이터 크기 확인 (X, Y 길이)
    y_len, x_len = slice_data.shape
    is_square_shape = (x_len == y_len) # 정사각형 여부 판별

    # 2. 데이터 스케일링 (Power Transform 1/3승)
    data_to_plot = np.power(slice_data, 1/3)

    # 3. 그림 크기 동적 계산
    base_size = 10
    
    # 텍스트 라벨이 있는 경우, 라벨 개수에 맞춰 크기 확장 계산
    calc_width = base_size
    calc_height = base_size
    
    if x_token_labels: calc_width = max(base_size, len(x_token_labels) * 0.25)
    if y_token_labels: calc_height = max(base_size, len(y_token_labels) * 0.25)
    
    # [수정됨] 정사각형 데이터일 경우 가로/세로 크기를 통일
    if is_square_shape:
        max_dim = max(calc_width, calc_height)
        width = max_dim
        height = max_dim
    else:
        width = calc_width
        height = calc_height

    # 너무 커지지 않도록 최대 크기 제한 (비율 유지)
    max_limit = 60
    if width > max_limit or height > max_limit:
        scale_factor = max_limit / max(width, height)
        width *= scale_factor
        height *= scale_factor

    plt.figure(figsize=(width, height))
    
    # 4. 히트맵 그리기
    heatmap_args = {
        "data": data_to_plot,
        "cmap": "viridis",
        "cbar_kws": {'label': 'Attention Score ^ (1/3)', 'shrink': 0.8},
        # [수정됨] 데이터가 정사각형이면 각 셀(Cell)을 정사각형으로 강제
        "square": is_square_shape
    }
    
    if x_token_labels:
        curr_labels = x_token_labels
        if len(curr_labels) > slice_data.shape[1]:
            curr_labels = curr_labels[:slice_data.shape[1]]
        heatmap_args["xticklabels"] = curr_labels

    if y_token_labels:
        curr_labels = y_token_labels
        if len(curr_labels) > slice_data.shape[0]:
            curr_labels = curr_labels[:slice_data.shape[0]]
        heatmap_args["yticklabels"] = curr_labels

    sns.heatmap(**heatmap_args)
    
    # 5. 축 설정
    x_title = f"{x_axis_name} [{x_range[0]}:{x_range[1]}]"
    if x_token_labels:
        x_title += " (Text)"
        plt.xticks(rotation=90, fontsize=10)
    else:
        x_title += " (Idx)"
        
    plt.xlabel(x_title, fontsize=14, labelpad=15)

    y_title = f"{y_axis_name} [{y_range[0]}:{y_range[1]}]"
    if y_token_labels:
        y_title += " (Text)"
        plt.yticks(rotation=0, fontsize=10)
    else:
        y_title += " (Idx)"
        
    plt.ylabel(y_title, fontsize=14, labelpad=15)

    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    # -----------------------------------------------------------
    # 1. 파일 로드 (Draft Model Attention & Generated Tokens)
    # -----------------------------------------------------------
    attn_file = "attentions/attention_maps/q4/draft_attn_dog.npy"  # [중요] Draft Model Attention Map
    token_file = "attentions/attention_maps/q4/generated_tokens_dog.json"

    if not os.path.exists(attn_file):
        print(f"Error: {attn_file} not found. Please run inference first.")
        return

    print(f"Loading Draft Model Attention Map from {attn_file}...")
    attn_map = np.load(attn_file)
    total_len = attn_map.shape[0]
    print(f"Draft Map Shape: {attn_map.shape} (Total Tokens: {total_len})")

    # -----------------------------------------------------------
    # 2. 인덱스 범위 동적 계산 (Compressed Image Logic)
    # -----------------------------------------------------------
    # 가정: Text(System, Instruction, Generated) 길이는 Target과 동일하고
    #      나머지 공간은 모두 '압축된 이미지'가 차지한다.
    
    # (A) Generated Tokens 로드
    gen_token_texts = []
    if os.path.exists(token_file):
        with open(token_file, 'r', encoding='utf-8') as f:
            token_info = json.load(f)
            gen_token_texts = [item['token_text'] for item in token_info]
    else:
        print("Warning: generated_tokens.json not found.")
    
    len_gen = len(gen_token_texts) # 생성된 토큰 개수

    # (B) 고정된 텍스트 길이 정의 (Target Model 기준)
    # System: 0 ~ 35 (35 tokens)
    # Instruction: 1983 ~ 2000 (17 tokens)
    LEN_SYSTEM = 35
    LEN_INSTRUCTION = 17 

    # (C) Draft Model 범위 역산
    # 구조: [System] -> [Compressed Image] -> [Instruction] -> [Generated]
    
    idx_system_end = LEN_SYSTEM
    
    # Generated는 맨 뒤에 위치
    idx_gen_start = total_len - len_gen
    
    # Instruction은 Generated 바로 앞에 위치
    idx_instr_start = idx_gen_start - LEN_INSTRUCTION
    
    # Image는 System 끝과 Instruction 시작 사이
    idx_image_start = idx_system_end
    idx_image_end = idx_instr_start

    ranges = {
        "System": (0, idx_system_end),
        "Image": (idx_image_start, idx_image_end),
        "Instruction": (idx_instr_start, idx_gen_start),
        "Generated": (idx_gen_start, total_len)
    }

    print("\n[Draft Model Index Ranges Calculated]")
    for key, val in ranges.items():
        print(f" - {key}: {val} (Length: {val[1] - val[0]})")
    
    if ranges["Image"][1] <= ranges["Image"][0]:
        print("\n[Warning] Calculated Image range is invalid (Size <= 0).")
        print("Check if 'draft_attn.npy' matches the 'generated_tokens.json'.")

    # -------------------------------------------------------
    # 3. 시각화 수행
    # -------------------------------------------------------
    region_order = ["System", "Image", "Instruction", "Generated"]
    
    print("\nStarting visualization...")
    
    # (1) 모든 조합 시각화
    for q_name in region_order:
        for k_name in region_order:
            # Causal Mask check
            if ranges[k_name][0] > ranges[q_name][0]:
                continue 

            y_labels = gen_token_texts if q_name == "Generated" else None
            x_labels = gen_token_texts if k_name == "Generated" else None
            
            rel_type = "Self-Attention" if q_name == k_name else "Cross-Attention"
            plot_title = f"[{rel_type}] Query: {q_name} -> Key: {k_name}"
            filename = f"draft_attn_{q_name}_vs_{k_name}.png"
            
            plot_general_heatmap(
                attn_map,
                x_range=ranges[k_name],
                y_range=ranges[q_name],
                title=plot_title,
                save_path=filename,
                x_token_labels=x_labels,
                y_token_labels=y_labels,
                x_axis_name=f"Key ({k_name})",
                y_axis_name=f"Query ({q_name})"
            )

    # (2) Full Map 저장
    print("Saving Full Draft Attention Map...")
    plot_general_heatmap(
        attn_map,
        x_range=(0, total_len),
        y_range=(0, total_len),
        title="Full Draft Attention Map",
        save_path="draft_attn_FULL_MAP.png",
        x_axis_name="Key Index",
        y_axis_name="Query Index"
    )

    print("\nAll visualizations complete.")

if __name__ == "__main__":
    main()
