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
from PIL import Image
from transformers import LlavaNextImageProcessor

# -----------------------------------------------------------------------------
# 환경 및 경로 설정
# -----------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def prepare_inputs(tokenizer, image_file, prompt, device, model_path):
    """이미지와 텍스트 입력을 처리하여 텐서로 변환합니다."""
    image = load_image(image_file)
    
    qs = prompt
    if '<image>' not in qs:
        qs = '<image>\n' + qs
    
    # LLaVA v1.6 Chat Template
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
    """단일 <image> 토큰을 실제 이미지 특징 개수만큼 복제하여 확장합니다."""
    input_ids_list = input_ids[0].tolist()
    new_input_ids = []
    
    for token in input_ids_list:
        if token == image_token_idx:
            new_input_ids.extend([image_token_idx] * num_image_tokens)
        else:
            new_input_ids.append(token)
            
    return torch.tensor([new_input_ids], device=input_ids.device, dtype=torch.long)

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
    
    # 이미지 토큰 ID 확인
    image_token_idx = getattr(model.base_model.config, "image_token_index", 32000)

    # 3. 추론 실행 (Error-Driven Expansion 전략)
    # LLaVA-NeXT의 동적 토큰 개수를 맞추기 위해 1회 실패 후 재시도하는 방식을 사용
    print("Generating answer...")
    torch.cuda.synchronize()
    
    try:
        # [시도 1] 기본 입력(토큰 1개)으로 실행
        output_ids, new_token, idx, accp_len = model.specgenerate(
            **model_inputs,
            temperature=args.temperature,
            log=True,
            return_acceptance_len=True,
        )

    except ValueError as e:
        # [예외 처리] 토큰 개수 불일치 에러 발생 시, 필요한 개수를 파싱하여 확장 후 재시도
        error_msg = str(e)
        match = re.search(r"features\s+(\d+)", error_msg)
        
        if match:
            num_features = int(match.group(1))
            model_inputs['input_ids'] = expand_input_ids(
                model_inputs['input_ids'], num_features, image_token_idx
            )
            
            # [시도 2] 확장된 입력으로 재실행
            output_ids, new_token, idx, accp_len = model.specgenerate(
                **model_inputs,
                temperature=args.temperature,
                log=True,
                return_acceptance_len=True,
            )
        else:
            raise e

    torch.cuda.synchronize()

    # 4. 결과 디코딩 및 출력
    input_len = model_inputs["input_ids"].shape[1]
    output_ids = output_ids[0][input_len:] # 프롬프트 부분 제거

    output_ids[output_ids > tokenizer.vocab_size] = 0
    output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
    
    # 특수 토큰 제거
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model Configuration
    parser.add_argument("--spec-model-path", type=str, default="/data/youngmin/checkpoints/fine/state_20")
    parser.add_argument("--base-model-path", type=str, default="/data/youngmin/models/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--use-ours", type=int, default=1, help="1: SpecModelOurs, 2: SpecModelHiVis")
    parser.add_argument("--use-medusa", type=bool, default=False)
    
    # Speculative Decoding Parameters
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-8bit", action="store_true") # 기본값 False로 변경 (필요시 True)

    # Input Configuration
    parser.add_argument("--image-file", type=str, default="./doggos.jpg")
    parser.add_argument("--prompt", type=str, default="What is the image? Please describe in very detailed.")

    args = parser.parse_args()

    run_inference(args)
