# Qwen3-VL LoRA Inference

탄소 증빙 문서(전기요금, 도시가스, 운송장)에서 구조화된 데이터를 추출하는 Qwen3-VL LoRA 추론 스크립트입니다.

## 파일 구조

```
unsloth-inference/
├── image             # 샘플 이미지 폴더
├── inference.py      # 실행 진입점
├── model.py          # 모델 로드 & 추론 로직
├── prompt.txt        # 추출 지시 프롬프트
├── requirements.txt  # 패키지 의존성
└── README.md
```

## 요구 사항

- Python 3.10+
- CUDA 지원 GPU (VRAM 8GB 이상 권장)
- CUDA 12.4

## 사용법

### 기본 실행

```bash
# 단일 이미지
python inference.py --image ../image/gas.png

# 여러 이미지 한 번에 처리
python inference.py --image ./electric.jpg ./gas.png ./transport.png
```



## 출력 형식

각 이미지에 대해 `./results/{이미지명}_{타임스탬프}.json` 파일이 생성됩니다.

### 전기 / 가스 문서

```json
{
  "category": "electric",
  "confidence": 0.97,
  "document_number": null,
  "usage_amount": 1234,
  "usage_unit": "kWh",
  "billing_amount": 185000,
  "billing_period_start": "2024-11-01",
  "billing_period_end": "2024-11-30",
  "customer_number": "1234567890",
  "transport_details": null,
  "raw_text": "전기요금 청구서 고객번호 1234567890 ..."
}
```

### 운송 문서

```json
{
  "category": "transport",
  "confidence": 0.95,
  "document_number": "WB-20241115-001",
  "usage_amount": null,
  "usage_unit": null,
  "billing_amount": 350000,
  "billing_period_start": null,
  "billing_period_end": null,
  "customer_number": null,
  "transport_details": {
    "fuel_type": "경유",
    "distance_km": 120.5,
    "vehicle_number": "12가3456",
    "transport_date": "2024-11-15",
    "freight_amount": 350000,
    "origin": { "name": "인천항", "address": "인천광역시 중구 ..." },
    "destination": { "name": "물류센터", "address": "경기도 이천시 ..." },
    "waypoints": []
  },
  "raw_text": "운송장번호 WB-20241115-001 차량번호 12가3456 ..."
}
```


## 라이선스

MIT
