import json
import numpy as np
from typing import Dict, Any

def analyze_json_file(file_path: str) -> Dict[str, Any]:
    """
    주어진 경로의 JSON 파일을 분석하여 성능 지표를 계산합니다.
    서로 다른 JSON 구조를 자동으로 처리합니다.

    Args:
        file_path (str): 분석할 JSON 파일의 경로.

    Returns:
        Dict[str, Any]: 계산된 성능 지표를 담은 딕셔너리.
    """
    total_tokens = 0
    total_time = 0.0
    all_acceptance_lengths = []
    generation_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    generation_count += 1
                    
                    if data.get('choices'):
                        choice = data['choices'][0]
                        
                        # --- 핵심 수정 로직 ---
                        # 'new_tokens' 값이 0보다 크면 그 값을 사용 (e.g., Speculative 모델)
                        # 그렇지 않으면 'idxs' 값을 토큰 수로 사용 (e.g., Baseline 모델)
                        new_tokens_val = choice.get('new_tokens', [0])[0]
                        if new_tokens_val > 0:
                            effective_tokens = new_tokens_val
                        else:
                            effective_tokens = choice.get('idxs', [0])[0]
                        # ---------------------

                        total_tokens += effective_tokens
                        total_time += choice.get('wall_time', [0.0])[0]
                        
                        # 'acceptance_length' 키가 없는 경우를 대비해 .get() 사용
                        all_acceptance_lengths.extend(choice.get('acceptance_length', []))

                except json.JSONDecodeError:
                    print(f"경고: '{file_path}' 파일의 다음 라인에서 JSON 파싱 오류: {line.strip()}")
                except (IndexError, KeyError) as e:
                    print(f"경고: '{file_path}' 파일의 데이터 구조 오류: {e}, 데이터: {line.strip()}")

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None

    # 최종 지표 계산
    tps = total_tokens / total_time if total_time > 0 else 0
    avg_acceptance_length = np.mean(all_acceptance_lengths) if all_acceptance_lengths else 0

    return {
        "file_name": file_path,
        "generation_count": generation_count,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tps": tps,
        "avg_acceptance_length": avg_acceptance_length,
    }

def print_results(stats: Dict[str, Any]):
    """분석 결과를 깔끔한 형식으로 출력합니다."""
    if not stats:
        return
    
    print(f"--- 📄 {stats['file_name']} 분석 결과 ---")
    print(f"총 생성 요청 수: {stats['generation_count']} 건")
    print(f"총 생성 토큰 수: {stats['total_tokens']} 개")
    print(f"총 소요 시간: {stats['total_time']:.2f} 초")
    print(f"**처리량 (TPS)**: {stats['tps']:.2f} 토큰/초")
    print(f"**평균 수락 길이**: {stats['avg_acceptance_length']:.2f}")
    print("-" * 35)

if __name__ == "__main__":
    # --- 파일 경로 설정 ---
    # 비교할 파일들의 경로를 여기에 입력하세요.
    # 사용자가 제공한 새로운 형식의 파일
    baseline_file = "/data/youngmin/results/AR_coco/llava-1.5-0.0.jsonl" 
    # 기존 형식의 파일
    test_file = "/data/youngmin/results/SD_coco/llava-1.5-0.0_depth5.jsonl"
    # ---------------------

    print("=" * 35)
    print("성능 비교 분석 시작")
    print("=" * 35)

    baseline_stats = analyze_json_file(baseline_file)
    test_stats = analyze_json_file(test_file)

    if baseline_stats:
        print_results(baseline_stats)
    if test_stats:
        print_results(test_stats)

    if baseline_stats and test_stats:
        if baseline_stats['tps'] > 0:
            speed_improvement = ((test_stats['tps'] - baseline_stats['tps']) / baseline_stats['tps']) * 100
        else:
            speed_improvement = float('inf')

        print("\n--- 📊 최종 비교 결과 ---")
        print(f"**평균 수락 길이**: Baseline {baseline_stats['avg_acceptance_length']:.2f} -> Test {test_stats['avg_acceptance_length']:.2f}")
        print(f"**처리량 (TPS)**: Baseline {baseline_stats['tps']:.2f} -> Test {test_stats['tps']:.2f}")
        
        if speed_improvement >= 0:
            print(f"🚀 **속도 개선율: +{speed_improvement:.2f}%**")
        else:
            print(f"🐢 **속도 저하율: {speed_improvement:.2f}%**")
        print("=" * 35)
