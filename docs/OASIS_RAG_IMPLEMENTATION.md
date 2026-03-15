# O.A.S.I.S. RAG Pipeline — 구현 현황 & Setup Guide

> **O.A.S.I.S.** — Offline AI Survival & First-aid Kit
> Raspberry Pi 5 위에서 완전 오프라인으로 동작하는 음성 응급처치 AI 디바이스

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [아키텍처 다이어그램](#2-아키텍처-다이어그램)
3. [구현 완료 항목](#3-구현-완료-항목)
4. [파일 구조](#4-파일-구조)
5. [RAG 파이프라인 상세](#5-rag-파이프라인-상세)
6. [API 레퍼런스](#6-api-레퍼런스)
7. [Setup Guide](#7-setup-guide)
8. [테스트 실행](#8-테스트-실행)
9. [하이퍼파라미터](#9-하이퍼파라미터)
10. [지식 문서 추가 방법](#10-지식-문서-추가-방법)
11. [성능 기준치](#11-성능-기준치)
12. [알려진 제약사항](#12-알려진-제약사항)

---

## 1. 프로젝트 개요

### 무엇을 만들었나

Pocket RAG 논문 기반의 **3-Stage Hybrid RAG 파이프라인**을 Python으로 구현하고,
기존 Node.js/TypeScript Whisplay 챗봇에 브릿지로 연결했습니다.

### 동작 흐름

```
사용자 음성
    → STT (Whisper)
    → OasisAdapter.ts  (Node.js)
        → RAG 서비스 HTTP 요청 (Python Flask, port 5001)
            → 3-Stage Hybrid Retrieval
            → 압축된 지식 컨텍스트 반환
        → 시스템 프롬프트 생성
    → LLM (gemma3:1b via Ollama)
    → 응답 텍스트
    → TTS
```

### 기술 스택

| 레이어 | 기술 |
|---|---|
| 임베딩 모델 | `thenlper/gte-small` (384차원, sentence-transformers) |
| 벡터 DB | FAISS `IndexFlatIP` (코사인 유사도, 인메모리) |
| 키워드 필터 | 커스텀 의료 키워드 사전 (317개 용어, 12 카테고리) |
| HTTP 서비스 | Flask 3.0, port 5001 |
| Node.js 브릿지 | axios, never-throw 패턴 |
| LLM | gemma3:1b (Ollama, 오프라인) |
| 타깃 하드웨어 | Raspberry Pi 5 (8GB RAM) |

---

## 2. 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────┐
│                    Node.js Process                       │
│                                                         │
│  ChatFlow.ts                                            │
│      │                                                  │
│      ▼                                                  │
│  OasisAdapter.ts ──────────────────────────────────┐   │
│      │  getSystemPromptFromOasis(query)             │   │
│      │                                              │   │
│      ├─[1] ragRetrieve()                            │   │
│      │       oasis-rag-client.ts (axios, 5s timeout)│   │
│      │       └─ POST http://localhost:5001/retrieve  │   │
│      │                                              │   │
│      ├─[2] matchProtocolLocal()  (RAG 실패 시)      │   │
│      │       30개 하드코딩 프로토콜 매처             │   │
│      │                                              │   │
│      └─[3] ""  (모두 실패 시)                       │   │
│                                                     │   │
└─────────────────────────────────────────────────────┘   │
                                                          │
┌─────────────────────────────────────────────────────────┘
│                Python Flask Process (port 5001)
│
│  service.py
│      │
│      ▼
│  Retriever.retrieve(query)
│      │
│      ├── Stage 1: Lexical Pre-filtering
│      │       medical_keywords.py (detect + expand)
│      │       keyword_map (역인덱스) 조회
│      │       최대 50개 후보 청크 선별
│      │
│      ├── Stage 2: Semantic Re-ranking
│      │       gte-small로 쿼리 임베딩
│      │       FAISS reconstruct → dot product
│      │       hybrid_score = 0.6*cosine + 0.4*lexical
│      │       score >= 0.10 필터 → Top-3
│      │
│      └── Stage 3: Context Compression
│              compressor.py
│              쿼리 키워드 히트 문장만 유지
│              최소 2문장, 최소 60% 토큰 보존
│              [Section Header] + 압축 본문 반환
│
│  data/rag_index/      (인덱스 아티팩트)
│      chunks.faiss
│      metadata.json
│      keyword_map.json
│
│  data/knowledge/      (지식 원본 마크다운)
│      severe_bleeding.md
│      cpr_adult.md
│      choking_adult.md
│      anaphylaxis.md
```

---

## 3. 구현 완료 항목

### Phase 1 — 문서 준비

- [x] `python/oasis-rag/document_chunker.py` — 300토큰 청킹, 50토큰 오버랩, 마크다운 헤딩 추출
- [x] `python/oasis-rag/medical_keywords.py` — 317개 의료 키워드, 12 카테고리 분류
- [x] `data/knowledge/severe_bleeding.md` — WHO BEC Module 3 + TCCC/IFRC 기반
- [x] `data/knowledge/cpr_adult.md` — AHA 2020 / ERC 2021 기준
- [x] `data/knowledge/choking_adult.md` — ERC 2021 / ILCOR 기준
- [x] `data/knowledge/anaphylaxis.md` — WAO 2020 기준

### Phase 2 — RAG 파이프라인 코어

- [x] `python/oasis-rag/config.py` — 중앙 하이퍼파라미터 관리
- [x] `python/oasis-rag/indexer.py` — FAISS 인덱스 빌드 + 저장/로드
- [x] `python/oasis-rag/retriever.py` — 3-Stage Hybrid Retriever
- [x] `python/oasis-rag/compressor.py` — 선택적 컨텍스트 압축
- [x] `python/oasis-rag/service.py` — Flask REST API (port 5001)
- [x] `python/oasis-rag/requirements.txt`

### Phase 3 — 인덱싱 테스트

- [x] 4개 지식 문서 인덱싱 성공
- [x] `python/oasis-rag/test_retriever.py` — 5개 쿼리 기능 테스트

### Phase 4 — TypeScript 브릿지

- [x] `src/cloud-api/local/oasis-rag-client.ts` — never-throw axios 클라이언트
- [x] `src/core/OasisAdapter.ts` — 3-tier 폴백 시스템 프롬프트 생성기

### Phase 5 — 서비스 통합

- [x] `run_chatbot.sh` — RAG 서비스 자동 기동 + 헬스체크 포함
- [x] `startup.sh` — systemd 유닛 자동 생성 (`oasis-rag.service`, `chatbot.service`)
- [x] `index_knowledge.sh` — 지식 인덱스 재빌드 스크립트
- [x] `install_dependencies.sh` — Python RAG 의존성 설치 포함
- [x] `.env.template` — RAG 서비스 환경변수 추가

### Phase 6 — 테스트 & 검증

- [x] `python/oasis-rag/test_accuracy.py` — 30개 정확도 테스트 케이스
- [x] `python/oasis-rag/benchmark.py` — 레이턴시 벤치마크
- [x] `python/oasis-rag/run_all_tests.py` — 통합 테스트 러너 (Stage 0~3)

**최종 결과: ALL STAGES PASSED (정확도 100%, 총 소요 12.9s)**

---

## 4. 파일 구조

```
whisplay-ai-chatbot-1/
│
├── data/
│   ├── knowledge/                  # 지식 마크다운 문서 (여기에 추가)
│   │   ├── severe_bleeding.md
│   │   ├── cpr_adult.md
│   │   ├── choking_adult.md
│   │   └── anaphylaxis.md
│   └── rag_index/                  # 인덱스 아티팩트 (자동 생성, git ignore)
│       ├── chunks.faiss
│       ├── metadata.json
│       └── keyword_map.json
│
├── python/
│   └── oasis-rag/
│       ├── config.py               # 하이퍼파라미터 (여기서 튜닝)
│       ├── medical_keywords.py     # 의료 키워드 사전
│       ├── document_chunker.py     # 마크다운 → 청크
│       ├── indexer.py              # FAISS 인덱스 빌드/로드
│       ├── retriever.py            # 3-Stage Hybrid Retriever
│       ├── compressor.py           # 컨텍스트 압축
│       ├── service.py              # Flask API 서버
│       ├── requirements.txt
│       ├── test_retriever.py       # 기능 테스트 (5 쿼리)
│       ├── test_accuracy.py        # 정확도 테스트 (30 케이스)
│       ├── benchmark.py            # 레이턴시 벤치마크
│       └── run_all_tests.py        # 통합 테스트 러너
│
├── src/
│   ├── core/
│   │   └── OasisAdapter.ts         # RAG → 시스템 프롬프트 변환
│   └── cloud-api/local/
│       └── oasis-rag-client.ts     # Flask API HTTP 클라이언트
│
├── run_chatbot.sh                  # 개발/수동 실행
├── startup.sh                      # systemd 서비스 등록
├── index_knowledge.sh              # 지식 재인덱싱
├── install_dependencies.sh         # 의존성 설치
└── .env.template                   # 환경변수 템플릿
```

---

## 5. RAG 파이프라인 상세

### Stage 1 — Lexical Pre-filtering

쿼리에서 의료 키워드를 감지하고 역인덱스(`keyword_map`)로 후보 청크를 좁힘.

```python
# medical_keywords.py에서 쿼리 분석
detected = detect_keywords(query)    # 직접 매칭 키워드
expanded = expand_query(query)       # 카테고리 확장 키워드

# keyword_map 조회 → 후보 청크 ID 집합
for term in all_terms:
    candidate_set |= set(keyword_map.get(term, []))

# 최대 50개로 제한 (LEXICAL_CANDIDATE_POOL)
```

- 키워드 미매칭 시 전체 코퍼스 폴백
- 후보 청크는 쿼리-특화 lexical score 순으로 정렬

### Stage 2 — Hybrid Semantic Re-ranking

```
hybrid_score = 0.6 × cosine_similarity + 0.4 × query_lexical_score
```

- `cosine_similarity`: gte-small 임베딩 내적 (L2 정규화)
- `query_lexical_score`: 쿼리 확장 용어 중 청크에 존재하는 비율
  - 글로벌 키워드 밀도 방식 사용 시 모든 의료 문서가 동점이 되는 문제를 해결
- `score_threshold = 0.10` 미만 청크 제외
- Top-K = 3 청크 선택

### Stage 3 — Context Compression

```python
# 문장 단위 스코어링
sentence_score = keyword_hits × 1.0 + position_bonus(0.05 per position)

# 유지 조건: score > 0.0 (최소 1개 키워드 히트)
# 하한 보장: 최소 2문장 or 원본의 60% 이상 토큰
```

- 출력 형식: `[Section Header]\n압축된 문장들`
- 여러 청크는 `\n\n---\n\n`로 구분, 출처 태그 포함

### 시스템 프롬프트 구조

```
You are OASIS, an offline first-aid assistant.
You respond ONLY based on the REFERENCE below.
Rules:
1. Maximum 5 numbered steps. Plain text only.
2. Each step under 15 words.
3. If supplies unavailable, suggest alternatives.
4. Never diagnose. Never prescribe medication.
5. If unsure: Call emergency services immediately.
6. If panicking: Start with 'Take a deep breath. I will guide you.'

REFERENCE:
[Source 1: severe_bleeding.md]
[Direct Pressure -- Step 1]
Apply firm pressure with a clean cloth or bandage...

---

[Source 2: severe_bleeding.md]
[Tourniquet Application]
Apply tourniquet 5-7 cm above wound...
```

---

## 6. API 레퍼런스

### GET /health

```bash
curl http://localhost:5001/health
```

**200 OK (정상)**
```json
{
  "status": "ok",
  "index_ready": true,
  "chunk_count": 42,
  "model": "thenlper/gte-small"
}
```

**503 (인덱스 미로드)**
```json
{
  "status": "degraded",
  "index_ready": false,
  "error": "Index artifact not found. Run indexer.py first."
}
```

### POST /retrieve

```bash
curl -X POST http://localhost:5001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "patient bleeding from leg wont stop"}'
```

**Request body**
```json
{
  "query": "patient bleeding from leg wont stop",
  "top_k": 3,        // 선택 (기본값 3)
  "compress": true   // 선택 (기본값 true)
}
```

**200 OK**
```json
{
  "context": "[Source 1: severe_bleeding.md]\n...",
  "chunks": [
    {
      "source": "severe_bleeding.md",
      "section": "Tourniquet Application",
      "hybrid_score": 0.7234,
      "cosine_score": 0.8103,
      "lexical_score": 0.5500,
      "compressed_text": "..."
    }
  ],
  "stage1_candidates": 12,
  "stage2_passing": 3,
  "latency_ms": 87.4
}
```

### POST /index

지식 디렉토리를 재인덱싱합니다. 새 문서 추가 후 호출.

```bash
curl -X POST http://localhost:5001/index \
  -H "Content-Type: application/json" \
  -d '{"knowledge_dir": "data/knowledge"}'
```

**200 OK**
```json
{
  "status": "ok",
  "chunk_count": 42,
  "duration_s": 14.7,
  "index_dir": "data/rag_index"
}
```

---

## 7. Setup Guide

### 사전 요구사항

- Python 3.10+
- Node.js 20+
- Ollama (gemma3:1b 모델 설치됨)
- RAM: 전체 파이프라인 4.5GB 이내 (Pi5 8GB 기준 충분)

### 7.1 클린 설치 (Pi5 / 새 환경)

```bash
# 1. 저장소 클론
git clone <repo-url>
cd whisplay-ai-chatbot-1

# 2. 전체 의존성 설치 (Python + Node.js)
chmod +x install_dependencies.sh
./install_dependencies.sh

# 3. 환경변수 설정
cp .env.template .env
# .env 편집이 필요한 경우 (기본값으로도 동작):
#   OASIS_RAG_SERVICE_URL=http://localhost:5001
#   OASIS_RAG_TIMEOUT_MS=5000

# 4. 지식 인덱스 빌드
#    최초 실행 시 gte-small 모델 다운로드 (~90MB)
chmod +x index_knowledge.sh
./index_knowledge.sh

# 5. systemd 서비스 등록 (Pi5 상시 운영용)
chmod +x startup.sh
./startup.sh
```

### 7.2 개발 환경 (수동 실행)

터미널 3개를 사용하거나, `run_chatbot.sh`로 한 번에 실행.

```bash
# 터미널 1: Ollama
ollama serve

# 터미널 2: RAG 서비스 (인덱스가 없으면 먼저 ./index_knowledge.sh 실행)
cd python/oasis-rag
python service.py

# 터미널 3: 챗봇
./run_chatbot.sh
```

또는 한 번에:

```bash
./run_chatbot.sh
# Ollama → RAG 서비스 → 헬스체크(최대 30초 대기) → Node.js 챗봇 순으로 기동
```

### 7.3 Python 의존성만 별도 설치

```bash
cd python/oasis-rag
pip install -r requirements.txt

# Pi5에서 faiss-cpu 빌드 실패 시:
sudo apt install libopenblas-dev
pip install faiss-cpu --no-binary faiss-cpu
```

### 7.4 환경변수

`.env` 파일 또는 shell 환경에서 설정:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OASIS_RAG_SERVICE_URL` | `http://localhost:5001` | RAG 서비스 주소 |
| `OASIS_RAG_TIMEOUT_MS` | `5000` | HTTP 요청 타임아웃 (ms) |
| `OASIS_KNOWLEDGE_DIR` | `data/knowledge` | 지식 문서 디렉토리 |
| `OASIS_INDEX_DIR` | `data/rag_index` | 인덱스 아티팩트 저장 위치 |
| `OASIS_EMBED_MODEL` | `thenlper/gte-small` | 임베딩 모델 ID |

### 7.5 systemd 서비스 확인

```bash
sudo systemctl status oasis-rag    # RAG 서비스 상태
sudo systemctl status chatbot       # 챗봇 상태

# 로그 확인
journalctl -u oasis-rag -f
journalctl -u chatbot -f

# 재시작
sudo systemctl restart oasis-rag
```

---

## 8. 테스트 실행

### 통합 테스트 (권장)

모든 스테이지를 순서대로 실행. 인덱스를 새로 빌드하고 전체 검증.

```bash
cd python/oasis-rag
python run_all_tests.py
```

옵션:

```bash
python run_all_tests.py --skip-index   # 기존 인덱스 재사용
python run_all_tests.py --skip-bench   # 벤치마크 생략
python run_all_tests.py --iterations 5 # 벤치마크 반복 횟수
```

**예상 출력 (PC 기준)**:
```
  [PASS] Stage 0: Indexer
  [PASS] Stage 1: Retriever
  [PASS] Stage 2: Accuracy     30/30 passed  (100.0%  target >= 90%)
  [PASS] Stage 3: Benchmark    All latency targets met
  ALL STAGES PASSED
  Total wall time: 12.9s
```

### 개별 테스트

```bash
cd python/oasis-rag

# 기능 테스트 (5 쿼리, 예상 문서 확인)
python test_retriever.py

# 정확도 테스트 (30 케이스)
python test_accuracy.py
python test_accuracy.py --verbose   # 컨텍스트 미리보기 포함

# 레이턴시 벤치마크
python benchmark.py
python benchmark.py --iterations 20
python benchmark.py --quiet         # per-run 출력 억제
```

### 빠른 smoke test

```bash
cd python/oasis-rag

# 단일 쿼리 직접 테스트
python retriever.py "patient bleeding from arm, pressure not working"

# 서비스 헬스체크
curl http://localhost:5001/health

# 서비스 쿼리
curl -s -X POST http://localhost:5001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"HELP THERE IS SO MUCH BLOOD"}' | python -m json.tool
```

---

## 9. 하이퍼파라미터

`python/oasis-rag/config.py`에서 조정. 재인덱싱 없이 적용 가능한 항목과 재인덱싱이 필요한 항목이 다릅니다.

### 재인덱싱 불필요 (retriever 재생성만)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `ALPHA` | `0.6` | 시맨틱 가중치 (0=완전 lexical, 1=완전 semantic) |
| `SCORE_THRESHOLD` | `0.10` | 최소 hybrid 점수 (낮추면 더 많은 청크 통과) |
| `TOP_K` | `3` | LLM에 전달할 최대 청크 수 |
| `LEXICAL_CANDIDATE_POOL` | `50` | Stage 1 최대 후보 수 |
| `COMPRESS_ENABLED` | `True` | Stage 3 압축 활성화 |
| `COMPRESS_MIN_RATIO` | `0.60` | 압축 후 최소 토큰 보존 비율 |

### 재인덱싱 필요

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `CHUNK_SIZE` | `300` | 청크 토큰 크기 |
| `CHUNK_OVERLAP` | `50` | 청크 간 오버랩 토큰 |
| `EMBEDDING_MODEL` | `thenlper/gte-small` | 임베딩 모델 |

---

## 10. 지식 문서 추가 방법

1. `data/knowledge/` 에 `.md` 파일 추가

2. 헤딩 구조로 작성 (검색 품질에 중요):
   ```markdown
   # 문서 제목
   [DOMAIN_TAGS: keyword1, keyword2, ...]

   ## 섹션 제목

   내용...

   ### 서브섹션

   내용...
   ```

3. `DOMAIN_TAGS` 줄에 관련 의료 키워드 나열 (Stage 1 매칭에 사용)

4. 인덱스 재빌드:
   ```bash
   ./index_knowledge.sh
   # 또는 서비스 실행 중이면:
   curl -X POST http://localhost:5001/index \
     -H "Content-Type: application/json" \
     -d '{}'
   ```

5. 새 키워드가 필요하다면 `python/oasis-rag/medical_keywords.py`의 `MEDICAL_TAXONOMY` 딕셔너리에 추가

---

## 11. 성능 기준치

### 벤치마크 결과 (PC / NVIDIA GPU 기준)

| 스테이지 | 평균 | 표준편차 | p95 | 목표 |
|---|---|---|---|---|
| Stage 1 Lexical | 0.0ms | 0.0ms | 0.1ms | < 5ms |
| Stage 2 Semantic | 14.3ms | 2.1ms | 18.4ms | < 500ms |
| Stage 3 Compress | 1.3ms | 0.3ms | 1.8ms | < 50ms |
| **Total Pipeline** | **16.4ms** | **2.3ms** | **20.1ms** | **< 2000ms** |

### Pi5 (CPU-only) 예상

- GPU 대비 3~8배 느림
- Stage 2 Semantic: ~100~400ms 예상
- Total: ~500~1500ms 예상 — 목표 이내

### 정확도 테스트 결과

| 카테고리 | 결과 | 케이스 수 |
|---|---|---|
| PHYSICAL_FIRST_AID | 20/20 (100%) | 20 |
| SAFETY (무해성 가드레일) | 5/5 (100%) | 5 |
| PANIC (대문자 패닉 쿼리) | 5/5 (100%) | 5 |
| **전체** | **30/30 (100%)** | **30** |

---

## 12. 알려진 제약사항

| 항목 | 내용 |
|---|---|
| 메모리 | gte-small 모델 ~90MB, FAISS 인메모리, 전체 ~500MB 이내 |
| 지식 범위 | 현재 4개 문서 (출혈, CPR, 기도폐쇄, 아나필락시스) |
| 언어 | 영어 지식 문서 + 영어 임베딩 (한국어 쿼리는 내부적으로 영어 키워드 매칭) |
| 진단 불가 | 설계상 의도: RAG는 프로토콜을 제공하며 진단/처방 생성 금지 |
| 최초 실행 | gte-small 모델 최초 다운로드 ~90MB (이후 오프라인) |
| Pi5 faiss | `pip install faiss-cpu` 실패 시 소스 빌드 필요 (`libopenblas-dev`) |

---

*구현 기간: 2026-03-15*
*기준 브랜치: `rag`*
