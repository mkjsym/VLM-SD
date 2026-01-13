import os
from datasets import load_dataset

def load_data():
    # 1. (수정) "coco"가 아닌 원래의 "HuggingFaceM4/COCO"로 되돌립니다.
    dataset = load_dataset("HuggingFaceM4/COCO", split="test")

    # 2. (수정) "image_id"가 아닌 원래의 "cocoid"로 되돌립니다.
    imgid_indices = {d["imgid"]: idx for idx, d in enumerate(dataset)}
    
    filtered_dataset = dataset.select(imgid_indices.values())
    return filtered_dataset.shuffle(seed=42).select(range(0, 100))

data = load_data()
print("데이터 로드 시도 중...")
