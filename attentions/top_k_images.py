import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def analyze_top_k_image_attention(attn_data, gen_tokens, img_range, gen_range, top_k=5, save_path="top_k_attention.json"):
    """
    [분석] 생성된 토큰들이 어떤 이미지 토큰을 가장 많이 참조했는지 Top-K 연산하여 출력/저장
    """
    print(f"\n[Analysis] Analyzing Top-{top_k} Image Attention per Generated Token...")
    
    # 1. 데이터 슬라이싱 (Query: Generated, Key: Image)
    # shape: (Generated 토큰 수, Image 토큰 수)
    attn_slice = attn_data[gen_range[0]:gen_range[1], img_range[0]:img_range[1]]
    
    if attn_slice.size == 0:
        print("Error: Sliced attention map is empty.")
        return

    n_gen, n_img = attn_slice.shape
    print(f"Analysis Shape: {n_gen} generated tokens x {n_img} image tokens")

    # 2. Top-K 연산
    # np.argsort: 값의 크기 순으로 인덱스 반환 (오름차순)
    # axis=1 (행/Query 기준)로 정렬 후, 뒤에서 k개를 뽑고, 다시 뒤집어서(내림차순) 정렬
    top_k_indices_local = np.argsort(attn_slice, axis=1)[:, -top_k:][:, ::-1]
    
    # 해당 인덱스의 실제 Score 값 가져오기
    # 행 인덱스 배열 생성 (0, 1, 2... n_gen)을 (n_gen, 1) 형태로 만듦
    row_indices = np.arange(n_gen)[:, None]
    top_k_scores = attn_slice[row_indices, top_k_indices_local]

    # 3. 결과 정리 및 출력
    results = []
    
    print("-" * 80)
    print(f"{'Gen Token':<20} | Top-{top_k} Image Token Indices (Global) & Scores")
    print("-" * 80)

    for i in range(n_gen):
        # 현재 생성 토큰의 텍스트
        token_text = gen_tokens[i] if i < len(gen_tokens) else f"Gen_Idx_{i}"
        
        # Local Index (0 ~ 1948) -> Global Index (35 ~ 1983) 변환
        global_img_indices = top_k_indices_local[i] + img_range[0]
        scores = top_k_scores[i]
        
        # 저장용 데이터 구조 생성
        entry = {
            "gen_token_idx": int(gen_range[0] + i),
            "gen_token_text": token_text,
            "top_k_image_indices": global_img_indices.tolist(),
            "top_k_scores": scores.tolist()
        }
        results.append(entry)

        # 콘솔 출력 (가독성을 위해 3개까지만 줄여서 보여주거나 전체 보여줌)
        idx_str = ", ".join([str(idx) for idx in global_img_indices])
        score_str = ", ".join([f"{s:.4f}" for s in scores])
        
        # 너무 길면 출력 생략
        if i < 10 or i > n_gen - 5: 
            print(f"{token_text:<20} | Idx: [{idx_str}]")
            print(f"{'':<20} | Val: [{score_str}]")
            if i == 9: print("... (skipping middle tokens) ...")

    # 4. JSON 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("-" * 80)
    print(f"Top-K analysis saved to: {save_path}")
    print("-" * 80)


def plot_general_heatmap(attn_data, x_range, y_range, title, save_path, 
                          x_token_labels=None, y_token_labels=None, 
                          x_axis_name="Key", y_axis_name="Query"):
    """
    [범용] Attention Map 시각화 함수 (기존 유지)
    """
    slice_data = attn_data[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    if slice_data.size == 0: return

    y_len, x_len = slice_data.shape
    is_square_shape = (x_len == y_len)
    data_to_plot = np.power(slice_data, 1/3) # Power Transform

    # 크기 계산
    base_size = 10
    calc_width = max(base_size, len(x_token_labels) * 0.25) if x_token_labels else base_size
    calc_height = max(base_size, len(y_token_labels) * 0.25) if y_token_labels else base_size

    if is_square_shape:
        width = height = max(calc_width, calc_height)
    else:
        width, height = calc_width, calc_height

    max_limit = 60
    if max(width, height) > max_limit:
        scale = max_limit / max(width, height)
        width *= scale
        height *= scale

    plt.figure(figsize=(width, height))
    
    heatmap_args = {
        "data": data_to_plot,
        "cmap": "viridis",
        "cbar_kws": {'label': 'Attention Score ^ (1/3)', 'shrink': 0.8},
        "square": is_square_shape
    }
    
    if x_token_labels:
        curr = x_token_labels[:slice_data.shape[1]]
        heatmap_args["xticklabels"] = curr
    if y_token_labels:
        curr = y_token_labels[:slice_data.shape[0]]
        heatmap_args["yticklabels"] = curr

    sns.heatmap(**heatmap_args)
    
    plt.xlabel(f"{x_axis_name} [{x_range[0]}:{x_range[1]}]", fontsize=14)
    plt.ylabel(f"{y_axis_name} [{y_range[0]}:{y_range[1]}]", fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    
    if x_token_labels: plt.xticks(rotation=90, fontsize=10)
    if y_token_labels: plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap: {save_path}")

def main():
    # 1. 파일 경로 설정
    base_dir = "attentions/attention_maps/q4"  # 폴더 경로 수정 필요 시 변경
    attn_file = os.path.join(base_dir, "target_attn_ski.npy")
    token_file = os.path.join(base_dir, "generated_tokens_ski.json")

    if not os.path.exists(attn_file):
        print(f"Error: {attn_file} not found.")
        return

    print(f"Loading attention map from {attn_file}...")
    attn_map = np.load(attn_file)
    print(f"Full Map Shape: {attn_map.shape}")

    # 2. 토큰 정보 로드
    gen_token_texts = []
    gen_start = 2000 # Default
    if os.path.exists(token_file):
        print(f"Loading token info from {token_file}...")
        with open(token_file, 'r', encoding='utf-8') as f:
            token_info = json.load(f)
            gen_token_texts = [item['token_text'] for item in token_info]
            if token_info:
                gen_start = token_info[0]['global_index']
    else:
        print("Warning: Token file not found. Using indices only.")

    gen_end = attn_map.shape[0]

    # 3. 인덱스 범위 정의
    ranges = {
        "System": (0, 35),
        "Image": (35, 1983),       # 여기가 분석 대상 (Key)
        "Instruction": (1983, gen_start),
        "Generated": (gen_start, gen_end) # 여기가 분석 주체 (Query)
    }

    # =================================================================
    # [NEW] Top-K Attention Analysis (Generated -> Image)
    # =================================================================
    analyze_top_k_image_attention(
        attn_data=attn_map,
        gen_tokens=gen_token_texts,
        img_range=ranges["Image"],     # Key Range
        gen_range=ranges["Generated"], # Query Range
        top_k=100,                       # 상위 5개 추출
        save_path="top_100_image_attention.json"
    )
    # =================================================================

    # 4. 히트맵 시각화 (기존 로직)
    print("\nStarting heatmap visualization...")
    
    # 보고 싶은 특정 조합만 그리려면 아래 리스트를 수정하세요.
    # 예: check_pairs = [("Generated", "Image")]
    region_order = ["System", "Image", "Instruction", "Generated"]
    
    for q_name in region_order:
        for k_name in region_order:
            if ranges[k_name][0] > ranges[q_name][0]: continue 

            y_labels = gen_token_texts if q_name == "Generated" else None
            x_labels = gen_token_texts if k_name == "Generated" else None
            
            rel_type = "Self-Attention" if q_name == k_name else "Cross-Attention"
            plot_title = f"[{rel_type}] Query: {q_name} -> Key: {k_name}"
            filename = f"attn_{q_name}_vs_{k_name}.png"
            
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

    print("\nAll tasks complete.")

if __name__ == "__main__":
    main()
