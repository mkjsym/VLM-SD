import torch
import requests
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import seaborn as sns

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
local_model_path = "/data/youngmin/models/llava-v1.6-vicuna-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {local_model_path}...")
processor = LlavaNextProcessor.from_pretrained(local_model_path)
model = LlavaNextForConditionalGeneration.from_pretrained(
    local_model_path,
    quantization_config=quantization_config,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)

# ==========================================
# 2. 데이터 준비
# ==========================================
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKhJ9vY-WJviH34cgDfbG2Hn_cBf0t5BBmaWrmH--NzBO3pjGP6hjV7pb8s958ug9K7p6iR-3vz6nlw7c4i5ZdMw"

try:
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    target_size = (336, 336)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    print(f"Input Image Size: {image.size}")
except Exception as e:
    print(f"이미지 로드 실패: {e}")
    exit()

prompt_text = "What's the content of the image? explain in very detail."
prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt_text} ASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# ==========================================
# 3. Forward Pass
# ==========================================
print("Calculating Attention Map...")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# ==========================================
# 4. 시각화 및 값 출력 (Debug 코드 추가됨)
# ==========================================
def visualize_selected_cls_to_compressed_image(
    model, processor, inputs, outputs, 
    threshold_ratio=0.9, 
    save_filename="selected_cls_to_compressed_image_debug.png"
):
    # 1. 인덱스 설정
    input_ids = inputs.input_ids[0]
    image_token_id = model.config.image_token_index
    image_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    
    if len(image_indices) == 0: return
    
    img_start = image_indices[0].item()
    img_end = image_indices[-1].item() + 1
    cls_index = img_start 
    text_start = img_end
    text_end = len(input_ids)
    
    # 2. Attention Score 추출 (keep_topk 로직 반영)
    last_layer_attn_all = outputs.attentions[-1]
    
    # (Head 평균)
    avg_attention = last_layer_attn_all[0].mean(dim=0) 
    
    # (CLS 토큰의 Attention)
    cls_token_attention = avg_attention[cls_index]
    
    # (이미지 토큰들에 대한 점수)
    image_token_scores = cls_token_attention[image_indices]

    # =========================================================
    # [추가된 부분] CLS Token Attention 값 출력 (디버깅용)
    # =========================================================
    print("\n" + "="*50)
    print(" [Debug] CLS Token (<image_start>) Attention Stats")
    print("="*50)
    
    # 텐서 값을 Python Float으로 변환하여 출력
    min_val = image_token_scores.min().item()
    max_val = image_token_scores.max().item()
    mean_val = image_token_scores.mean().item()
    std_val = image_token_scores.std().item()
    
    print(f"1. Overall Statistics (over {len(image_token_scores)} image tokens):")
    print(f"   - Min  : {min_val:.6e}")
    print(f"   - Max  : {max_val:.6e}")
    print(f"   - Mean : {mean_val:.6e}")
    print(f"   - Std  : {std_val:.6e}")
    
    # 상위 10개 값 출력
    topk_vals, topk_inds = torch.topk(image_token_scores, k=min(10, len(image_token_scores)))
    print(f"\n2. Top-{len(topk_vals)} Highest Attention Scores:")
    for rank, (val, idx) in enumerate(zip(topk_vals.tolist(), topk_inds.tolist())):
        print(f"   Rank {rank+1}: {val:.6e} (Image Token Index: {idx})")
        
    if max_val - min_val < 1e-9:
        print("\n[Warning] Attention Score의 변화가 거의 없습니다 (Uniform). 시각화 시 단색으로 보일 수 있습니다.")
    print("="*50 + "\n")
    # =========================================================

    # (텍스트 토큰들에 대한 점수)
    text_attn_to_img = avg_attention[text_start:text_end, image_indices]

    # ------------------------------------------------
    # Step A: 압축 및 다운샘플링
    # ------------------------------------------------
    cls_score_np = image_token_scores.cpu().float().numpy().reshape(1, -1) 
    text_score_np = text_attn_to_img.cpu().float().numpy()                 
    
    combined_raw_map = np.vstack([cls_score_np, text_score_np])
    
    clip_patch_grid = 24 
    tokens_per_patch = clip_patch_grid * clip_patch_grid 
    
    if combined_raw_map.shape[1] >= tokens_per_patch:
        compact_img_attn = combined_raw_map[:, -tokens_per_patch:] 
    else:
        compact_img_attn = combined_raw_map

    target_grid = 12 
    reshaped_attn = compact_img_attn.reshape(compact_img_attn.shape[0], clip_patch_grid, clip_patch_grid)
    
    downsampled_list = []
    for i in range(reshaped_attn.shape[0]):
        resized = cv2.resize(reshaped_attn[i], (target_grid, target_grid), interpolation=cv2.INTER_AREA)
        downsampled_list.append(resized.flatten())
    
    compressed_map_all = np.array(downsampled_list)

    # ------------------------------------------------
    # Step B: 중요 텍스트 토큰 선별
    # ------------------------------------------------
    text_self_attn = avg_attention[text_start:text_end, text_start:text_end].cpu().float().numpy()
    max_score = np.max(text_self_attn)
    row_max_values = np.max(text_self_attn, axis=1)
    
    selected_text_indices_local = np.where(row_max_values >= max_score * threshold_ratio)[0]
    rows_to_plot = [0] + (1 + selected_text_indices_local).tolist()
    
    final_attn_matrix = compressed_map_all[rows_to_plot, :]

    # ------------------------------------------------
    # Step C: Row-wise Normalization
    # ------------------------------------------------
    normalized_matrix = np.zeros_like(final_attn_matrix)
    for i in range(final_attn_matrix.shape[0]):
        row = final_attn_matrix[i]
        row_min, row_max = row.min(), row.max()
        if row_max - row_min > 1e-9:
            normalized_matrix[i] = (row - row_min) / (row_max - row_min)
        else:
            normalized_matrix[i] = row

    # ------------------------------------------------
    # Step D: 시각화
    # ------------------------------------------------
    y_labels = []
    y_labels.append("★ <image_start> (CLS)")
    
    for local_idx in selected_text_indices_local:
        global_idx = text_start + local_idx
        token_id = input_ids[global_idx]
        token_str = processor.tokenizer.decode([token_id]).replace(' ', ' ')
        if token_str == "": token_str = "[UNK]"
        if token_str == "\n": token_str = "[\\n]"
        y_labels.append(token_str)

    num_cols = target_grid * target_grid 
    num_rows = len(y_labels)
    
    fig_width = max(12, num_cols * 0.15)
    fig_height = max(5, num_rows * 0.6)
    
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    
    sns.heatmap(
        normalized_matrix,
        xticklabels=False,
        yticklabels=y_labels,
        cmap="viridis",
        square=True,
        cbar=True,
        cbar_kws={'label': 'Relative Attention (Row-normalized)', 'shrink': 0.8},
        ax=ax,
        vmin=0.0, vmax=1.0
    )
    
    ax.set_title(f"Attention (with Values Debug): [ImageStart + Queries] -> Image", fontsize=16)
    ax.set_xlabel(f"Compressed Image Region ({target_grid}x{target_grid})", fontsize=14)
    ax.set_ylabel("Source Tokens", fontsize=14)
    ax.tick_params(axis='y', rotation=0, labelsize=12)

    plt.tight_layout()
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Success] 저장 완료: {save_filename}")

# 실행
visualize_selected_cls_to_compressed_image(model, processor, inputs, outputs, threshold_ratio=0.8)
