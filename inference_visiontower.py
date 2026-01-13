import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import requests
import os
import numpy as np
import cv2

# 1. 모델 로드
model_id = "/data/youngmin/models/llava-v1.6-vicuna-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf" 
print("Loading model...")
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager"
).to("cuda")
processor = LlavaNextProcessor.from_pretrained(model_id)

# 2. 이미지 준비
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKhJ9vY-WJviH34cgDfbG2Hn_cBf0t5BBmaWrmH--NzBO3pjGP6hjV7pb8s958ug9K7p6iR-3vz6nlw7c4i5ZdMw"
try:
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
except Exception as e:
    exit()

inputs = processor(text="", images=image, return_tensors="pt").to("cuda")

# 3. Vision Tower 실행
pixel_values = inputs.pixel_values
b, n, c, h, w = pixel_values.shape
pixel_values_reshaped = pixel_values.view(b * n, c, h, w)

with torch.no_grad():
    outputs = model.vision_tower(pixel_values_reshaped, output_attentions=True)
attentions = outputs.attentions 

# ==========================================
# 4. [핵심] Log Scale + Layer 15 시각화
# ==========================================
save_dir = "attention_log_scale"
os.makedirs(save_dir, exist_ok=True)

patch_idx = 0 # Global View

# [솔루션 1] 너무 깊은 층(20~) 말고 "중간층"인 15번 레이어를 선택하세요.
# CLIP 계열은 12~16번 레이어에서 형태(Shape) 정보가 가장 뚜렷합니다.
target_layer_idx = 15 
layer_attn = attentions[target_layer_idx][patch_idx] # [Heads, Seq, Seq]
num_heads = layer_attn.shape[0]

print(f"Visualizing Layer {target_layer_idx} with LOG SCALE...")

# 배경 이미지 준비
input_tensor = pixel_values_reshaped[patch_idx]
img_np = input_tensor.detach().cpu().numpy().transpose(1, 2, 0)
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
img_uint8 = (img_np * 255).astype(np.uint8)
img_h, img_w, _ = img_uint8.shape
img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

for head_idx in range(num_heads):
    # 1. Attention Score 추출
    head_attn = layer_attn[head_idx]
    cls_attn = head_attn[0, 1:] # [Seq-1]
    
    # 2. Grid 변환
    num_tokens = cls_attn.shape[0]
    grid_size = int(num_tokens**0.5)
    attn_map = cls_attn.reshape(grid_size, grid_size).detach().cpu().float().numpy()
    
    # 3. [솔루션 2] Log Scale 적용 (핵심!)
    # 아주 작은 값(1e-6)을 더한 뒤 로그를 취해 낮은 값(고양이)을 증폭시킵니다.
    attn_log = np.log(attn_map + 1e-6)
    
    # 4. 리사이즈 (Log 적용된 맵을 리사이즈)
    attn_resized = cv2.resize(attn_log, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    
    # 5. 정규화 (로그 스케일 기준 Min-Max)
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    
    # 대비(Contrast)를 더 높이기 위해 제곱을 한 번 더 해줄 수도 있음 (선택 사항)
    # attn_norm = attn_norm ** 2 
    
    attn_uint8 = (attn_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
    
    # 6. 오버레이 및 저장
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    
    save_path = os.path.join(save_dir, f"layer{target_layer_idx}_head{head_idx}_log.png")
    cv2.imwrite(save_path, overlay)
    
    # [디버깅] 실제 값의 범위가 얼마나 컸는지 출력
    raw_max = attn_map.max()
    raw_min = attn_map.min()
    print(f"Head {head_idx}: Saved. (Raw Val Range: {raw_min:.5f} ~ {raw_max:.5f})")

print("\n저장 완료. 'attention_log_scale' 폴더를 확인하세요.")
