import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

# ---------------------------------------------------------
# 1. 데이터 로드 함수
# ---------------------------------------------------------
def load_data_standard_json(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] 파일이 존재하지 않습니다: {file_path}")
        return []

    print(f"[Info] 파일 로드 중... ({file_path})")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Info] 성공! 총 {len(data)}개의 토큰 데이터를 로드했습니다.")
        return data
        
    except json.JSONDecodeError as e:
        print(f"[Error] JSON 파싱 실패: {e}")
        return []
    except Exception as e:
        print(f"[Error] 알 수 없는 오류: {e}")
        return []

# ---------------------------------------------------------
# 2. 분석 및 시각화 로직 (수정됨: 두 개의 범례 모두 표시)
# ---------------------------------------------------------
def analyze_attention_trend(data, save_path="attention_result.png"):
    if not data:
        print("[Error] 분석할 데이터가 없습니다.")
        return

    rows = []
    print("[Info] 데이터 분석 및 변환 중...")
    
    for step, item in enumerate(data):
        if not isinstance(item, dict): continue
        if 'gen_token_text' not in item or 'top_k_image_indices' not in item: continue

        t_text = item['gen_token_text'].replace('▁', '').strip()
        t_idx = item.get('gen_token_idx', step)
        
        indices = np.array(item['top_k_image_indices'])
        scores = np.array(item['top_k_scores'])
        
        if len(scores) == 0 or np.sum(scores) == 0: continue
            
        weighted_avg_idx = np.sum(indices * scores) / np.sum(scores)
        
        for img_idx, score in zip(item['top_k_image_indices'], item['top_k_scores']):
            rows.append({
                "Step": step,
                "Token": t_text,
                "Global_Token_Idx": t_idx,
                "Image_Index": img_idx,
                "Score": score,
                "Weighted_Avg": weighted_avg_idx
            })

    if not rows:
        print("[Error] 시각화할 데이터가 없습니다.")
        return

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # 시각화 설정
    # ---------------------------------------------------------
    plt.figure(figsize=(18, 8))
    
    # 1. 배경: 개별 Attention Point (Scatter)
    ax = sns.scatterplot(
        data=df, 
        x="Step", 
        y="Image_Index", 
        size="Score", 
        hue="Score",
        sizes=(20, 200),
        alpha=0.6,
        palette="viridis",
        legend="brief" # Seaborn 범례 활성화
    )

    # 2. 전경: 가중 평균 흐름선 (Line)
    # line_plot 변수에 라인 객체를 저장해둡니다.
    line_plot, = plt.plot(
        df['Step'].unique(), # Step 기준 정렬이 필요할 수 있으나 df가 순서대로라면 OK
        df.groupby('Step')['Weighted_Avg'].first(), # 각 Step별 평균값 하나만 추출
        color='red', 
        linewidth=2, 
        linestyle='--', 
        label='Focus Center'
    )
    
    # ---------------------------------------------------------
    # [핵심 수정] 범례 2개(Scatter, Line) 동시에 띄우기
    # ---------------------------------------------------------
    
    # (1) Seaborn Scatter 범례 설정 및 이동
    #     - 위치를 (1.01, 0.85)로 설정하여 Line 범례 아래에 오게 합니다.
    try:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 0.85), title="Attn Score")
    except:
        pass # 구버전 Seaborn 대비

    # (2) Scatter 범례를 '아티스트'로 추가하여 고정 (이걸 안 하면 다음 legend 호출 때 사라짐)
    if ax.get_legend():
        ax.add_artist(ax.get_legend())

    # (3) Line Plot(Focus Center) 범례 새로 생성
    #     - 위치를 (1.01, 1.0)으로 설정하여 가장 위에 오게 합니다.
    plt.legend(
        handles=[line_plot], 
        labels=['Focus Center'], 
        loc='upper left', 
        bbox_to_anchor=(1.01, 1.0),
        frameon=True  # 테두리 표시
    )

    # ---------------------------------------------------------
    # 축 및 기타 설정
    # ---------------------------------------------------------
    # X축 라벨을 위해 중복 제거된 데이터프레임 생성
    df_sorted = df[['Step', 'Token']].drop_duplicates().sort_values('Step')
    
    plt.xticks(
        ticks=df_sorted['Step'],
        labels=df_sorted['Token'],
        rotation=90,
        fontsize=9
    )
    
    plt.title("Attention Trace Analysis(Top-20)", fontsize=16)
    plt.ylabel("Image Token Index", fontsize=12)
    plt.xlabel("Generated Sequence", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 범례가 잘리지 않도록 레이아웃 조정
    plt.tight_layout()

    # 파일 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Success] 그래프가 성공적으로 저장되었습니다: {save_path}")

    # ---------------------------------------------------------
    # 통계 출력
    # ---------------------------------------------------------
    print("\n[Analysis] Image Index Distribution by Key Tokens")
    print("-" * 60)
    
    target_keywords = ["image", "The", "dep"] 
    
    for keyword in target_keywords:
        subset = df[df['Token'].str.contains(keyword, case=False, na=False)]
        
        if not subset.empty:
            avg_loc = subset['Image_Index'].mean()
            std_loc = subset['Image_Index'].std()
            top_locs = subset.groupby('Image_Index')['Score'].sum().sort_values(ascending=False).head(5).index.tolist()
            
            print(f"Token containing '{keyword}':")
            print(f"  - Avg Location: {avg_loc:.1f} (±{std_loc:.1f})")
            print(f"  - Top 5 Indices: {top_locs}")
            print("-" * 60)

# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    input_file_path = "/home/youngmin/workspace/VLM-SD/attentions/top_20_image_attention.json" 
    output_image_path = "/home/youngmin/workspace/VLM-SD/attentions/attention_analysis_result_20.png"

    data = load_data_standard_json(input_file_path)
    
    if data:
        analyze_attention_trend(data, save_path=output_image_path)
