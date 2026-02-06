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
    - x_token_labels/y_token_labels: 축의 눈금(Tick) 라벨 (토큰 텍스트)
    - x_axis_name/y_axis_name: 축의 전체 이름 (Label)
    """
    # 1. 데이터 슬라이싱
    # Y축: Query (행), X축: Key (열)
    slice_data = attn_data[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    
    # 데이터가 비어있으면 스킵 (유효하지 않은 구간 등)
    if slice_data.size == 0:
        return

    # 데이터 크기 확인 (X, Y 길이)
    y_len, x_len = slice_data.shape
    is_square_shape = (x_len == y_len) # 정사각형 여부 판별

    # -------------------------------------------------------------------------
    # [수정됨] 데이터 스케일링 방식 변경
    # 기존 Log Scale -> Power Transform (1/3승)
    # Attention Score는 0~1 사이의 값이므로, 1/3승을 하면 작은 값이 크게 부각됨.
    # -------------------------------------------------------------------------
    data_to_plot = np.power(slice_data, 1/3)

    # 3. 그림 크기 동적 계산
    # 기본 크기
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

    # 너무 커지지 않도록 최대 크기 제한 (정사각형일 경우 비율 유지하며 제한)
    max_limit = 60
    if width > max_limit or height > max_limit:
        scale_factor = max_limit / max(width, height)
        width *= scale_factor
        height *= scale_factor

    plt.figure(figsize=(width, height))
    
    # 4. 히트맵 그리기 옵션 설정
    heatmap_args = {
        "data": data_to_plot,
        "cmap": "viridis",
        # 컬러바 라벨 변경
        "cbar_kws": {'label': 'Attention Score ^ (1/3)', 'shrink': 0.8},
        # [수정됨] 데이터가 정사각형이면 각 셀(Cell)을 정사각형으로 강제 -> 전체 맵도 정사각형이 됨
        "square": is_square_shape 
    }
    
    # X축 눈금 라벨 (Ticks)
    if x_token_labels:
        # 슬라이스 크기에 맞게 라벨 자르기
        curr_labels = x_token_labels
        if len(curr_labels) > slice_data.shape[1]:
            curr_labels = curr_labels[:slice_data.shape[1]]
        heatmap_args["xticklabels"] = curr_labels
    else:
        # 라벨이 없으면 자동 눈금
        pass

    # Y축 눈금 라벨 (Ticks)
    if y_token_labels:
        curr_labels = y_token_labels
        if len(curr_labels) > slice_data.shape[0]:
            curr_labels = curr_labels[:slice_data.shape[0]]
        heatmap_args["yticklabels"] = curr_labels
    else:
        pass

    # 히트맵 생성
    ax = sns.heatmap(**heatmap_args)
    
    # 5. 축 스타일 및 이름 설정
    # X축 이름
    x_title_text = f"{x_axis_name} [{x_range[0]} : {x_range[1]}]"
    if x_token_labels:
        x_title_text += " - (Text Tokens)"
        plt.xticks(rotation=90, fontsize=10) # 텍스트가 있으면 회전
    else:
        x_title_text += " - (Indices)"
        
    plt.xlabel(x_title_text, fontsize=14, labelpad=15)

    # Y축 이름
    y_title_text = f"{y_axis_name} [{y_range[0]} : {y_range[1]}]"
    if y_token_labels:
        y_title_text += " - (Text Tokens)"
        plt.yticks(rotation=0, fontsize=10)
    else:
        y_title_text += " - (Indices)"
        
    plt.ylabel(y_title_text, fontsize=14, labelpad=15)

    # 제목 설정
    plt.title(title, fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    # 1. 파일 로드
    attn_file = "attentions/attention_maps/q4/target_attn_ski.npy" # 또는 draft_attn.npy
    token_file = "attentions/attention_maps/q4/generated_tokens_ski.json"

    if not os.path.exists(attn_file):
        print(f"Error: {attn_file} not found.")
        return

    print(f"Loading attention map from {attn_file}...")
    attn_map = np.load(attn_file)
    print(f"Full Map Shape: {attn_map.shape}")

    # 2. 토큰 정보 로드 (Generated 영역용)
    gen_token_texts = []
    if os.path.exists(token_file):
        print(f"Loading token info from {token_file}...")
        with open(token_file, 'r', encoding='utf-8') as f:
            token_info = json.load(f)
            # 딕셔너리 리스트에서 텍스트만 추출
            gen_token_texts = [item['token_text'] for item in token_info]
    else:
        print("Warning: Generated token file not found. Visualizations will use indices.")

    # 3. 인덱스 범위 정의
    # 기본값 (토큰 파일이 없을 경우 대비)
    gen_start = 2000
    if token_info:
        gen_start = token_info[0]['global_index']
    
    gen_end = attn_map.shape[0] # 맵 끝까지
    
    ranges = {
        "System": (0, 35),
        "Image": (35, 1983),
        "Instruction": (1983, gen_start),
        "Generated": (gen_start, gen_end)
    }

    # 영역 순서 (순회용)
    region_order = ["System", "Image", "Instruction", "Generated"]
    
    print("\nStarting batch visualization for all combinations...")
    
    # 4. 이중 루프로 모든 조합 시각화 (Query x Key)
    # Query (Y축, 보는 주체)
    for q_name in region_order:
        # Key (X축, 보이는 대상)
        for k_name in region_order:
            
            # [Causal Masking 체크]
            if ranges[k_name][0] > ranges[q_name][0]:
                continue 

            # 라벨 준비 (Generated 영역인 경우에만 텍스트 라벨 적용)
            y_labels = gen_token_texts if q_name == "Generated" else None
            x_labels = gen_token_texts if k_name == "Generated" else None
            
            # 제목 및 파일명 설정
            rel_type = "Self-Attention" if q_name == k_name else "Cross-Attention"
            plot_title = f"[{rel_type}] Query: {q_name} -> Key: {k_name}"
            filename = f"attn_{q_name}_vs_{k_name}.png"
            
            # 시각화 실행
            plot_general_heatmap(
                attn_map,
                x_range=ranges[k_name],
                y_range=ranges[q_name],
                title=plot_title,
                save_path=filename,
                x_token_labels=x_labels,
                y_token_labels=y_labels,
                x_axis_name=f"Key ({k_name})",   # X축 이름
                y_axis_name=f"Query ({q_name})"  # Y축 이름
            )

    # 5. 전체 맵 (Full Map) 저장
    print("Saving Full Attention Map...")
    plot_general_heatmap(
        attn_map,
        x_range=(0, attn_map.shape[0]),
        y_range=(0, attn_map.shape[0]),
        title="Full Attention Map (All Regions)",
        save_path="attn_FULL_MAP.png",
        x_axis_name="Key Index (All)",
        y_axis_name="Query Index (All)"
    )

    print("\nAll visualizations complete.")

if __name__ == "__main__":
    main()

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# def plot_general_heatmap(attn_data, x_range, y_range, title, save_path, 
#                           x_axis_name="Key", y_axis_name="Query"):
#     """
#     [범용] Attention Map을 시각화합니다.
#     - 텍스트 라벨 대신 인덱스를 그대로 사용합니다.
#     - 정사각형 데이터는 1:1 비율 유지, 직사각형 데이터는 화면에 꽉 차게 비율 조정.
#     """
#     # 1. 데이터 슬라이싱
#     # Y축: Query (행), X축: Key (열)
#     slice_data = attn_data[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    
#     # 데이터가 비어있으면 스킵
#     if slice_data.size == 0:
#         return

#     # 데이터 크기 확인
#     y_len, x_len = slice_data.shape
#     is_square_shape = (x_len == y_len) # 정사각형 여부 판별

#     # 2. 데이터 스케일링 (Power Transform)
#     # 작은 값을 부각시키기 위해 1/3승 적용
#     data_to_plot = np.power(slice_data, 1/3)

#     # 3. 그림 크기 및 비율 설정 (핵심 수정 사항)
#     if is_square_shape:
#         # [정사각형] 가로/세로 1:1 비율 유지
#         # 크기는 12x12로 고정하되, 데이터가 매우 작으면 줄임
#         fig_dim = 12
#         plt.figure(figsize=(fig_dim, fig_dim))
#         square_cell = True  # seaborn 옵션: 셀을 정사각형으로 강제
#     else:
#         # [직사각형] 비율이 깨지지 않도록 고정된 화면 비율(Wide) 사용
#         # square=False로 설정하여 히트맵이 figure 크기에 맞춰 늘어나게 함
#         plt.figure(figsize=(16, 8)) 
#         square_cell = False 

#     # 4. 히트맵 그리기 옵션 설정
#     heatmap_args = {
#         "data": data_to_plot,
#         "cmap": "viridis",
#         "cbar_kws": {'label': 'Attention Score ^ (1/3)', 'shrink': 0.8},
#         "square": square_cell,  # 정사각형일 때만 True, 아니면 False(채우기)
#         # xticklabels, yticklabels를 지정하지 않으면 자동으로 인덱스 간격을 조절하여 표시함
#         "xticklabels": "auto", 
#         "yticklabels": "auto"
#     }

#     # 히트맵 생성
#     ax = sns.heatmap(**heatmap_args)
    
#     # 5. 축 스타일 및 이름 설정
#     # X축 이름
#     x_title_text = f"{x_axis_name} [{x_range[0]} : {x_range[1]}]"
#     plt.xlabel(x_title_text, fontsize=14, labelpad=15)
#     plt.xticks(rotation=45, fontsize=10) # 인덱스 겹침 방지를 위해 약간 회전

#     # Y축 이름
#     y_title_text = f"{y_axis_name} [{y_range[0]} : {y_range[1]}]"
#     plt.ylabel(y_title_text, fontsize=14, labelpad=15)
#     plt.yticks(rotation=0, fontsize=10)

#     # 제목 설정
#     plt.title(title, fontsize=16, pad=20)
    
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved: {save_path}")

# def main():
#     # 1. 파일 로드
#     attn_file = "attentions/attention_maps/draft_attn_ski.npy" # 파일 경로 확인 필요
#     token_file = "attentions/attention_maps/generated_tokens_ski.json" # 인덱스 시작점 파악용

#     if not os.path.exists(attn_file):
#         print(f"Error: {attn_file} not found.")
#         return

#     print(f"Loading attention map from {attn_file}...")
#     attn_map = np.load(attn_file)
#     print(f"Full Map Shape: {attn_map.shape}")

#     # 2. 인덱스 경계 확인 (토큰 텍스트 로딩 로직은 제거하고 시작 인덱스만 확인)
#     gen_start = 2000 # 기본값
#     if os.path.exists(token_file):
#         try:
#             with open(token_file, 'r', encoding='utf-8') as f:
#                 token_info = json.load(f)
#                 if token_info:
#                     gen_start = token_info[0]['global_index']
#                     print(f"Found generated start index: {gen_start}")
#         except Exception as e:
#             print(f"Warning: Could not parse token file. Using default index 2000. ({e})")
    
#     gen_end = attn_map.shape[0]
    
#     # 3. 인덱스 범위 정의
#     ranges = {
#         "System": (0, 35),
#         "Image": (35, 1983),
#         "Instruction": (1983, gen_start),
#         "Generated": (gen_start, gen_end)
#     }

#     # 영역 순서
#     region_order = ["System", "Image", "Instruction", "Generated"]
    
#     print("\nStarting batch visualization for all combinations...")
    
#     # 4. 이중 루프로 모든 조합 시각화
#     for q_name in region_order:
#         for k_name in region_order:
            
#             # [Causal Masking 체크] Key가 Query보다 미래에 있으면 스킵
#             if ranges[k_name][0] > ranges[q_name][0]:
#                 continue 

#             # 제목 및 파일명 설정
#             rel_type = "Self-Attention" if q_name == k_name else "Cross-Attention"
#             plot_title = f"[{rel_type}] Query: {q_name} -> Key: {k_name}"
#             filename = f"attn_{q_name}_vs_{k_name}.png"
            
#             # 시각화 실행 (라벨 전달 인자 제거됨)
#             plot_general_heatmap(
#                 attn_map,
#                 x_range=ranges[k_name],
#                 y_range=ranges[q_name],
#                 title=plot_title,
#                 save_path=filename,
#                 x_axis_name=f"Key ({k_name})",
#                 y_axis_name=f"Query ({q_name})"
#             )

#     # 5. 전체 맵 (Full Map) 저장
#     print("Saving Full Attention Map...")
#     plot_general_heatmap(
#         attn_map,
#         x_range=(0, attn_map.shape[0]),
#         y_range=(0, attn_map.shape[0]),
#         title="Full Attention Map (All Regions)",
#         save_path="attn_FULL_MAP.png",
#         x_axis_name="Key Index (All)",
#         y_axis_name="Query Index (All)"
#     )

#     print("\nAll visualizations complete.")

# if __name__ == "__main__":
#     main()
