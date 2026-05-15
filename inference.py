"""
inference.py — Qwen3-VL LoRA 추론 실행 스크립트
================================================
사용법:
    # 단일 이미지
    python inference.py --image ./bill.jpg

    # 여러 이미지
    python inference.py --image ./img1.jpg ./img2.jpg ./img3.jpg

    # 모델 경로 직접 지정
    python inference.py --image ./bill.jpg --model ./my-lora-adapter

    # 결과 저장 폴더 및 프롬프트 파일 지정
    python inference.py --image ./bill.jpg --output ./results --prompt ./prompt.txt

    # torch.compile 활성화 (GPU 메모리 여유 있을 때 속도 향상)
    python inference.py --image ./bill.jpg --compile
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

from PIL import Image

from model import load_model, run_inference, preprocess_image, print_timings


# ──────────────────────────────────────────────────────────────
# 기본값
# ──────────────────────────────────────────────────────────────

DEFAULT_MODEL      = "wish418/unsloth-inference"
DEFAULT_OUTPUT     = "./results"
DEFAULT_PROMPT     = "./prompt.txt"
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_SIZE   = 1024


# ──────────────────────────────────────────────────────────────
# 결과 파싱 & 저장
# ──────────────────────────────────────────────────────────────

def parse_json_output(raw_text: str) -> dict | None:
    """
    모델 출력에서 JSON을 파싱합니다.
    마크다운 코드블록(```json ... ```)이 포함된 경우에도 처리합니다.

    Returns:
        파싱 성공 시 dict, 실패 시 None
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def save_result(
    result: dict | None,
    raw_output: str,
    image_path: str,
    output_dir: str,
) -> str:
    """
    결과를 JSON 파일로 저장합니다.

    - 파싱 성공 시: 구조화된 dict를 저장
    - 파싱 실패 시: {"_raw_output": ...} 형태로 원본 텍스트를 저장

    Returns:
        저장된 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    stem      = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{stem}_{timestamp}.json")

    payload = result if result is not None else {"_raw_output": raw_output}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return save_path


# ──────────────────────────────────────────────────────────────
# 단일 이미지 처리
# ──────────────────────────────────────────────────────────────

def process_image(model, tokenizer, img_path: str, prompt: str, args) -> None:
    print(f"\n{'='*60}")
    print(f"처리 중: {img_path}")
    print("="*60)

    # 이미지 열기
    try:
        image = Image.open(img_path).convert("RGB")
        print(f"원본 크기: {image.size}")
    except Exception as e:
        print(f"❌ 이미지 열기 실패: {e}")
        return

    # 전처리
    image = preprocess_image(image, max_size=args.max_size)
    print(f"리사이즈 후: {image.size}")

    # 추론
    print("추론 중...")
    raw_output, timings = run_inference(
        model, tokenizer, image, prompt,
        max_new_tokens=args.max_tokens,
    )

    # 타이밍 출력
    print_timings(timings)

    # JSON 파싱
    result = parse_json_output(raw_output)

    # 콘솔 출력
    print("\n[결과]")
    if result is not None:
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("⚠️  JSON 파싱 실패 — raw 출력:")
        print(raw_output)

    # 저장
    saved_path = save_result(result, raw_output, img_path, args.output)
    print(f"\n✅ 저장 완료: {saved_path}")


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL LoRA 추론 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", nargs="+", required=True,
        metavar="PATH",
        help="추론할 이미지 경로 (1개 이상)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        metavar="PATH",
        help="LoRA 어댑터 또는 베이스 모델 경로",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        metavar="PATH",
        help="프롬프트 텍스트 파일 경로",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        metavar="DIR",
        help="결과 JSON 저장 폴더",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help="최대 생성 토큰 수",
    )
    parser.add_argument(
        "--max-size", type=int, default=DEFAULT_MAX_SIZE,
        metavar="PX",
        help="이미지 리사이즈 기준 (긴 변 최대 픽셀)",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="torch.compile 적용 (첫 실행 느리지만 이후 속도 향상)",
    )
    args = parser.parse_args()

    # 프롬프트 파일 로드
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        print(f"❌ 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
        return
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    print(f"[프롬프트] {prompt_path} ({len(prompt)} chars)")

    # 모델 로드 (한 번만)
    model, tokenizer = load_model(args.model, use_compile=args.compile)

    # 이미지별 처리
    for img_path in args.image:
        process_image(model, tokenizer, img_path, prompt, args)

    print(f"\n\n모든 처리 완료. 결과 폴더: {args.output}")


if __name__ == "__main__":
    main()