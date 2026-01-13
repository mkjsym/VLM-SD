import torch
import os
import argparse
from tqdm import tqdm

def inspect_checkpoint(file_path):
    print(f"\n[Inspect] Loading file: {file_path}")
    try:
        data = torch.load(file_path, map_location="cpu")
        
        if not isinstance(data, dict):
            print(f"Warning: Data is not a dict, it is {type(data)}")
            return

        print(f"{'Key':<25} | {'Type':<15} | {'Shape/Value':<30}")
        print("-" * 75)
        
        for key, value in data.items():
            if torch.is_tensor(value):
                info = str(list(value.shape))
                dtype = str(value.dtype).replace("torch.", "")
                print(f"{key:<25} | Tensor({dtype}) | {info:<30}")
            elif isinstance(value, list):
                info = f"List (len={len(value)})"
                # 리스트 내부 첫 번째 요소 타입 확인
                if len(value) > 0:
                    info += f", item_type={type(value[0])}"
                print(f"{key:<25} | List            | {info:<30}")
            else:
                print(f"{key:<25} | {type(value).__name__:<15} | {str(value)[:30]:<30}")
        
        # 특정 키(attentions, hidden_state)에 대한 추가 세부 정보
        if "attentions" in data:
            attn = data["attentions"]
            print("\n[Detail] 'attentions' stats:")
            print(f" - Is None? {attn is None}")
            if torch.is_tensor(attn):
                print(f" - Shape: {attn.shape}")
                print(f" - Min/Max: {attn.min():.4f} / {attn.max():.4f}")
        
        if "hidden_state" in data:
            print(f"\n[Detail] 'hidden_state' shape: {data['hidden_state'].shape}")

    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 확인하고 싶은 데이터 디렉토리 경로 (사용자 설정에 맞게 변경)
    parser.add_argument("--datapath", type=str, default="/home/youngmin/workspace/ViSpec/datasets/llava_pretrain_gen/llava_pretrain_gen_0_67999_mufp16/0")
    parser.add_argument("--num_files", type=int, default=3, help="검사할 파일 개수")
    args = parser.parse_args()

    if not os.path.exists(args.datapath):
        print(f"Error: Path {args.datapath} does not exist.")
    else:
        # 디렉토리 내 .ckpt 파일 찾기
        files = [f for f in os.listdir(args.datapath) if f.endswith(".ckpt")]
        files.sort() # 순서대로 정렬
        
        print(f"Found {len(files)} files in {args.datapath}")
        
        # 지정된 개수만큼만 확인
        for i in range(min(args.num_files, len(files))):
            file_path = os.path.join(args.datapath, files[i])
            inspect_checkpoint(file_path)
