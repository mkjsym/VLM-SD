"""
Generate a single answer with local models.

Usage:
python3 gen_single_answer.py \
    --base-model-path /path/to/base/model \
    --spec-model-path /path/to/spec/model \
    --image-file "path/to/image.jpg" \
    --prompt "Describe this image."
"""

import argparse
import os
import re
import sys
import torch
import numpy as np
import gc
from PIL import Image
from transformers import LlavaNextImageProcessor
import inspect 

# -----------------------------------------------------------------------------
# 환경 및 경로 설정
# -----------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# vispec 모듈 로딩을 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from vispec.model.utils import prepare_logits_processor
except ImportError:
    sys.path.append(os.getcwd())
    from vispec.model.utils import prepare_logits_processor

# -----------------------------------------------------------------------------
# 유틸리티 함수
# -----------------------------------------------------------------------------
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        import requests
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_model(args):
    """지정된 옵션에 따라 SpecModel을 로드합니다."""
    print(f"Loading model from {args.base_model_path}...")
    
    if args.use_ours == 1:
        from vispec.model.spec_model_ours import SpecModel
    elif args.use_ours == 2:
        from vispec.model.spec_model_hivis import SpecModel
    elif args.use_medusa:
        from vispec.model.spec_model_medusa import SpecModel
    else:
        from vispec.model.spec_model import SpecModel

    # 공통 인자 설정
    kwargs = {
        "base_model_path": args.base_model_path,
        "spec_model_path": args.spec_model_path,
        "total_token": args.total_token,
        "depth": args.depth,
        "top_k": args.top_k,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    
    if args.use_ours in [1, 2]:
        kwargs["num_q"] = args.num_q

    return SpecModel.from_pretrained(**kwargs)

# -----------------------------------------------------------------------------
# Attention Score 저장 함수 (Target/Draft 자동 감지)
# -----------------------------------------------------------------------------
def save_attention_score(model, input_ids, pixel_values, image_sizes, save_path="attn_score.npy"):
    # (이전과 동일한 코드 유지 - 주석 처리는 아래 메인 로직에서 수행)
    import gc
    import inspect
    
    print(f"Extracting attention scores to save at {save_path}...")
    torch.cuda.empty_cache()
    gc.collect()

    target_model = model.base_model if hasattr(model, "base_model") else model
    
    print("Moving model to CPU for extraction...")
    target_model = target_model.to('cpu')
    
    input_ids_cpu = input_ids.to('cpu')
    pixel_values_cpu = pixel_values.to('cpu') if pixel_values is not None else None
    image_sizes_cpu = image_sizes.to('cpu') if image_sizes is not None else None

    forward_params = inspect.signature(target_model.forward).parameters
    kwargs = {"output_attentions": True}
    is_draft_model = "pixel_values" not in forward_params
    
    if not is_draft_model:
        kwargs["input_ids"] = input_ids_cpu
        if pixel_values_cpu is not None: kwargs["pixel_values"] = pixel_values_cpu
        if image_sizes_cpu is not None: kwargs["image_sizes"] = image_sizes_cpu
        if "return_dict" in forward_params: kwargs["return_dict"] = True
        args_list = [] 
    else:
        if hasattr(target_model, "config") and hasattr(target_model.config, "image_token_index"):
            img_token_idx = target_model.config.image_token_index
        else:
            img_token_idx = 32000
        image_mask = (input_ids_cpu == img_token_idx)
        if hasattr(target_model, "embed_tokens"):
            inputs_embeds = target_model.embed_tokens(input_ids_cpu)
        else:
            raise AttributeError("Draft model does not have 'embed_tokens' layer.")
        args_list = [inputs_embeds] 
        kwargs["input_ids"] = input_ids_cpu
        kwargs["image_mask"] = image_mask

    try:
        with torch.inference_mode():
            outputs = target_model(*args_list, **kwargs)
        
        last_layer_attn = None
        if isinstance(outputs, tuple):
            last_layer_attn = outputs[1]
        elif hasattr(outputs, "attentions"):
            last_layer_attn = outputs.attentions[-1]
        else:
            if isinstance(outputs, tuple) and len(outputs) > 1:
                 last_layer_attn = outputs[-1] if len(outputs) > 2 else outputs[1]

        if last_layer_attn is None:
            raise ValueError("Could not find attention tensor in outputs.")

        attn_score_avg = torch.mean(last_layer_attn, dim=1).squeeze(0)
        attn_data = attn_score_avg.float().numpy()
        np.save(save_path, attn_data)
        print(f"Successfully saved attention score to '{save_path}' (Shape: {attn_data.shape})")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass

def prepare_inputs(tokenizer, image_file, prompt, device, model_path):
    image = load_image(image_file)
    qs = prompt
    if '<image>' not in qs: qs = '<image>\n' + qs
    text_prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {qs} ASSISTANT:"
    try:
        processor = LlavaNextImageProcessor.from_pretrained(model_path)
    except Exception:
        processor = LlavaNextImageProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    inputs = processor(image, return_tensors='pt')
    return {
        "input_ids": tokenizer(text_prompt, return_tensors='pt').input_ids.to(device),
        "pixel_values": inputs['pixel_values'].to(device, dtype=torch.float16),
        "image_sizes": inputs['image_sizes'].to(device),
        "attention_mask": None 
    }

def expand_input_ids(input_ids, num_image_tokens, image_token_idx=32000):
    input_ids_list = input_ids[0].tolist()
    new_input_ids = []
    for token in input_ids_list:
        if token == image_token_idx:
            new_input_ids.extend([image_token_idx] * num_image_tokens)
        else:
            new_input_ids.append(token)
    return torch.tensor([new_input_ids], device=input_ids.device, dtype=torch.long)

# -----------------------------------------------------------------------------
# [추가] 토큰 인덱스 정보 분석 함수
# -----------------------------------------------------------------------------
def print_token_indices(input_ids, input_len, image_token_idx, tokenizer):
    """
    전체 시퀀스에서 각 파트(System Prompt, Image, Instruction, Generated)의 인덱스 범위를 출력합니다.
    """
    ids = input_ids[0].cpu().tolist()
    total_len = len(ids)

    # 이미지 토큰의 위치 찾기
    image_indices = [i for i, x in enumerate(ids) if x == image_token_idx]

    print("\n" + "="*20 + " Token Index Analysis " + "="*20)
    print(f"Total Sequence Length (Query/Key axis size): {total_len}")
    print(f"Input Prompt Length: {input_len}")
    
    if not image_indices:
        print("[Warning] No image tokens found in the sequence.")
        # 이미지가 없는 경우 단순 프롬프트와 생성으로만 구분
        print(f"1. Prompt Region:     [0 : {input_len}]")
        print(f"2. Generated Tokens:  [{input_len} : {total_len}]")
    else:
        # 이미지가 존재하는 경우 (LLaVA 구조)
        # 구조: [System Prompt + "USER: "] -> [Image Tokens] -> ["\n" + Instruction + "ASSISTANT:"] -> [Generated]
        
        img_start = image_indices[0]
        img_end = image_indices[-1] + 1
        
        print(f"\n[Index Ranges for Query & Key Axes]")
        
        # 1. System Prompt (이미지 앞부분)
        print(f"1. System Prompt & Header: [0 : {img_start}]")
        print(f"   - Text snippet: {tokenizer.decode(ids[:min(20, img_start)], skip_special_tokens=True)} ...")

        # 2. Image Tokens
        print(f"2. Image Tokens:           [{img_start} : {img_end}]")
        print(f"   - Count: {img_end - img_start} tokens")

        # 3. Instruction (이미지 뒷부분 ~ 프롬프트 끝)
        print(f"3. User Instruction:       [{img_end} : {input_len}]")
        print(f"   - Text: {tokenizer.decode(ids[img_end:input_len], skip_special_tokens=False)}")

        # 4. Generated Tokens (프롬프트 끝 ~ 전체 끝)
        print(f"4. Generated Answer:       [{input_len} : {total_len}]")
        print(f"   - Text snippet: {tokenizer.decode(ids[input_len:min(input_len+20, total_len)], skip_special_tokens=True)} ...")

    print("="*60 + "\n")

# -----------------------------------------------------------------------------
# 메인 추론 로직
# -----------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(args):
    # 1. 모델 로드
    model = load_model(args)
    tokenizer = model.get_tokenizer()
    model.eval()

    # 2. 입력 준비
    print("Processing input...")
    device = next(model.parameters()).device
    model_inputs = prepare_inputs(tokenizer, args.image_file, args.prompt, device, args.base_model_path)
    image_token_idx = getattr(model.base_model.config, "image_token_index", 32000)

    # 3. 추론 실행
    print("Generating answer...")
    torch.cuda.synchronize()
    
    try:
        output_ids, new_token, idx, accp_len = model.specgenerate(
            **model_inputs,
            temperature=args.temperature,
            log=True,
            return_acceptance_len=True,
        )

    except ValueError as e:
        error_msg = str(e)
        match = re.search(r"features\s+(\d+)", error_msg)
        
        if match:
            num_features = int(match.group(1))
            model_inputs['input_ids'] = expand_input_ids(
                model_inputs['input_ids'], num_features, image_token_idx
            )
            output_ids, new_token, idx, accp_len = model.specgenerate(
                **model_inputs,
                temperature=args.temperature,
                log=True,
                return_acceptance_len=True,
            )
        else:
            raise e

    torch.cuda.synchronize()

    # 4. 결과 디코딩
    input_len = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:] 

    temp_ids = generated_ids.clone()
    temp_ids[temp_ids > tokenizer.vocab_size] = 0
    output = tokenizer.decode(temp_ids, spaces_between_special_tokens=False)
    
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for t in special_token: output = output.replace(t, "")
        else:
            output = output.replace(special_token, "")
    
    output = output.strip().replace("Assistant:", "", 1).strip()

    print("\n" + "=" * 20 + " FINAL OUTPUT " + "=" * 20)
    print(output)
    print("=" * 54)
    print(f"New tokens: {new_token}")
    print(f"Acceptance length info: {accp_len}")

    # -------------------------------------------------------------------------
    # [수정됨] Attention Score 추출 중단 및 인덱스 정보 출력
    # -------------------------------------------------------------------------
    
    # 시각화용 전체 토큰 시퀀스 구성
    viz_input_ids = torch.cat([model_inputs['input_ids'], generated_ids.unsqueeze(0)], dim=1)

    # 1) 토큰 인덱스 정보 콘솔 출력
    print_token_indices(viz_input_ids, input_len, image_token_idx, tokenizer)

    # -------------------------------------------------------------------------
    # [추가] 생성된 토큰 정보(인덱스, 텍스트) JSON 저장
    # -------------------------------------------------------------------------
    print("Saving generated token info to 'generated_tokens.json'...")
    
    # 토큰 ID를 실제 텍스트(subword)로 변환
    gen_token_strs = tokenizer.convert_ids_to_tokens(generated_ids)
    
    token_info_list = []
    for i, (t_id, t_text) in enumerate(zip(generated_ids.tolist(), gen_token_strs)):
        token_info_list.append({
            "relative_index": i,                # 생성된 토큰 내에서의 순서 (0부터 시작)
            "global_index": input_len + i,      # 전체 시퀀스(프롬프트 포함)에서의 절대 인덱스
            "token_id": t_id,                   # 토큰 정수 ID
            "token_text": t_text                # 토큰 텍스트 (예: " The", " dog")
        })

    import json
    with open("generated_tokens.json", "w", encoding="utf-8") as f:
        json.dump(token_info_list, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved info for {len(token_info_list)} tokens.")

    # 메모리 확보
    del output_ids, generated_ids, temp_ids, gen_token_strs
    torch.cuda.empty_cache()

    print("Skipping Attention Score extraction (commented out).")

    # [주석 처리됨] 1) Target Model
    if hasattr(model, 'base_model'):
        save_attention_score(
            model.base_model, 
            viz_input_ids, 
            model_inputs['pixel_values'], 
            model_inputs['image_sizes'],
            save_path="target_attn.npy"
        )

    # [주석 처리됨] 2) Draft Model
    draft_model = None
    if hasattr(model, 'draft_model'):
        draft_model = model.draft_model
    elif hasattr(model, 'spec_layer'): 
        draft_model = model.spec_layer
    elif hasattr(model, 'spec_model'): 
        draft_model = model.spec_model
    
    if draft_model is not None:
        try:
            save_attention_score(
                draft_model, 
                viz_input_ids, 
                model_inputs['pixel_values'], 
                model_inputs['image_sizes'],
                save_path="draft_attn.npy"
            )
        except Exception as e:
            print(f"Could not save Draft Model attention: {e}")
    else:
        print("Draft model attribute not found. Skipping draft attention save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model Configuration
    parser.add_argument("--spec-model-path", type=str, default="/data2/youngmin/checkpoints/fine_q4/state_20")
    parser.add_argument("--base-model-path", type=str, default="/data2/youngmin/models/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--use-ours", type=int, default=1, help="1: SpecModelOurs, 2: SpecModelHiVis")
    parser.add_argument("--use-medusa", type=bool, default=False)
    
    # Speculative Decoding Parameters
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-q", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-8bit", action="store_true")

    # Input Configuration
    parser.add_argument("--image-file", type=str, default="./ski.jpg")
    parser.add_argument("--prompt", type=str, default="What is the image? Please describe in very detailed.")

    args = parser.parse_args()

    run_inference(args)
