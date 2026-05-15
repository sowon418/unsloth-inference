"""
model.py — Qwen3-VL 모델 로드 및 추론 로직
"""

import time
import torch
from pathlib import Path
from PIL import Image

import os
#os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

from unsloth import FastVisionModel

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(messages):
        """qwen_vl_utils 미설치 시 이미지 리스트만 반환하는 폴백"""
        images = []
        for msg in messages:
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    images.append(content["image"])
        return images, []


# ──────────────────────────────────────────────────────────────
# 이미지 전처리
# ──────────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    긴 변 기준으로 이미지를 리사이즈합니다 (비율 유지).

    Args:
        image:    PIL Image 객체
        max_size: 긴 변의 최대 픽셀 수 (기본값 1024)

    Returns:
        리사이즈된 PIL Image 객체
    """
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


# ──────────────────────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────────────────────

def load_model(model_path: str, use_compile: bool = False):
    """
    Unsloth FastVisionModel을 로드합니다.

    Args:
        model_path:   LoRA 어댑터 또는 베이스 모델 경로
        use_compile:  torch.compile 적용 여부 (GPU 메모리 여유 있을 때 권장)

    Returns:
        (model, tokenizer) 튜플
    """
    print(f"[모델 로드] {model_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    FastVisionModel.for_inference(model)

    if use_compile:
        print("[torch.compile] 모델 컴파일 중... (첫 실행 시 시간이 걸립니다)")
        model = torch.compile(model, mode="reduce-overhead")

    print("✅ 모델 로드 완료")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────
# 단일 이미지 추론
# ──────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> tuple[str, dict]:
    """
    단일 이미지에 대해 추론을 수행합니다.

    Args:
        model:          로드된 모델
        tokenizer:      로드된 토크나이저
        image:          PIL Image 객체
        prompt:         시스템/유저 프롬프트 문자열
        max_new_tokens: 최대 생성 토큰 수
        temperature:    샘플링 온도
        top_p:          nucleus sampling p 값
        top_k:          top-k 샘플링 k 값

    Returns:
        (decoded_text, timings) 튜플
        - decoded_text: 모델 출력 문자열
        - timings:      구간별 소요 시간 딕셔너리
    """
    timings = {}

    # 1. 메시지 구성
    t0 = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt},
            ],
        }
    ]
    timings["1_message_build"] = time.perf_counter() - t0

    # 2. Chat template 적용
    t0 = time.perf_counter()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    timings["2_chat_template"] = time.perf_counter() - t0

    # 3. 이미지 전처리
    t0 = time.perf_counter()
    image_inputs, _ = process_vision_info(messages)
    timings["3_vision_preprocess"] = time.perf_counter() - t0

    # 4. 텍스트+이미지 → GPU 텐서
    t0 = time.perf_counter()
    inputs = tokenizer(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to("cuda")
    timings["4_tokenize_to_tensor"] = time.perf_counter() - t0

    # 5. 모델 generate
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )
    torch.cuda.synchronize()
    timings["5_model_generate"] = time.perf_counter() - t0

    # 6. 디코딩
    t0 = time.perf_counter()
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    timings["6_decode"] = time.perf_counter() - t0

    timings["_total"] = sum(v for k, v in timings.items() if not k.startswith("_"))
    return decoded, timings


# ──────────────────────────────────────────────────────────────
# 타이밍 출력 유틸
# ──────────────────────────────────────────────────────────────

def print_timings(timings: dict) -> None:
    total = timings["_total"]
    print(f"  {'총 소요시간':<30} {total:6.3f}s  100.0%")