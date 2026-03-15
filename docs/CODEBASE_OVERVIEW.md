# Codebase Overview

O.A.S.I.S. (Offline AI Survival & First-aid Kit) — Raspberry Pi 5 기반 오프라인 응급처치 AI 디바이스.
Whisplay chatbot fork 기반, 3-Stage Hybrid RAG (Pocket RAG 논문) 구현.

**Validation status: 109/109 tests PASS**

---

## 서비스 시작 순서

```
1. ollama serve          # LLM (gemma3:1b)
2. cd python/oasis-rag && python app.py   # RAG Flask service (:5000)
3. npm start             # Node.js chatbot
```

---

## 디렉토리 구조

### Root

| 파일 | 설명 |
|------|------|
| `CLAUDE.md` | AI 어시스턴트용 프로젝트 컨텍스트 (현재 상태 기준) |
| `CODEBASE_OVERVIEW.md` | 이 파일 |
| `OASIS_RAG_IMPLEMENTATION.md` | RAG 구현 상세 설계 |
| `OASIS_RAG_VALIDATION_PLAN.md` | 검증 계획 (109개 테스트) |
| `RAG_Implement_plan.md` | RAG 구현 단계별 계획 |
| `deprecated/` | 사용 중단 파일 (LanceDB/Qdrant 기반 구버전) |

### `src/` (TypeScript — Node.js 핵심 로직)

**`src/core/`**
- `ChatFlow.ts` — 메인 대화 루프 (STT → RAG → LLM → TTS)
- `OasisAdapter.ts` — RAG 결과를 LLM 시스템 프롬프트로 변환
- `Knowledge.ts` — 시스템 프롬프트 / 컨텍스트 관리
- `StreamResponsor.ts` — LLM → TTS 스트리밍 처리

**`src/cloud-api/local/`**
- `oasis-rag-client.ts` — Python RAG 서비스 HTTP 클라이언트 (주 경로)
- `oasis-matcher-node.ts` — **[FALLBACK]** RAG 서비스 다운 시 사용
- `ollama-llm.ts` — Ollama LLM 인터페이스
- 기타 local ASR/TTS 클라이언트

**`src/cloud-api/`** — Gemini, OpenAI, Grok 등 클라우드 API 어댑터 (개발/테스트용)

**`src/test/`** — 테스트 유틸리티 (ollama-text-test.ts, oasis-test.ts 등)

### `python/oasis-rag/` (RAG 파이프라인 — 핵심)

| 파일 | 설명 |
|------|------|
| `app.py` | Flask HTTP 서버 (`:5000`) |
| `retriever.py` | 3-Stage Hybrid Retriever |
| `indexer.py` | FAISS 인덱스 빌더 |
| `chunker.py` | 문서 청킹 로직 |
| `compressor.py` | Stage 3: 컨텍스트 압축 |
| `medical_keywords.py` | 의료 도메인 키워드 택소노미 (역인덱스용) |
| `config.py` | 서비스 설정 |
| `validation/` | 검증 테스트 스위트 (109/109 PASS) |

**임베딩:** `thenlper/gte-small` (384차원, sentence-transformers)
**벡터DB:** FAISS (인메모리)
**인덱스 위치:** `data/rag_index/`

### `data/knowledge/` (지식베이스)

| 파일 | 출처 |
|------|------|
| `who_bec_module1~7.md` | WHO Basic Emergency Care 2018 (7개 모듈) |
| `who_bec_skills.md` | WHO BEC 술기 가이드 |
| `redcross_altitude.md` | Red Cross Wilderness — 고산병 |
| `redcross_bites_stings.md` | Red Cross Wilderness — 교상/자상/알레르기 |
| `redcross_bone_joint.md` | Red Cross Wilderness — 골절/관절 |
| `redcross_burns.md` | Red Cross Wilderness — 화상 |
| `redcross_cold_emergencies.md` | Red Cross Wilderness — 저체온/동상 |
| `redcross_heat_emergencies.md` | Red Cross Wilderness — 열사병/열탈진 |
| `redcross_lightning.md` | Red Cross Wilderness — 낙뢰 |
| `redcross_special.md` | Red Cross Wilderness — 특수 상황 |
| `redcross_submersion.md` | Red Cross Wilderness — 익수 |
| `redcross_wounds.md` | Red Cross Wilderness — 창상 |

총 317 청크 인덱싱됨.

### `python/` (기타 하드웨어 서비스)

- `chatbot-ui.py` — LCD 디스플레이 서비스 (Pillow)
- `oasis-matcher.py` — **[FALLBACK]** RAG 서비스 대체용 프로토콜 매처
- `speech-service/` — Whisper ASR, Piper TTS 서비스

### `docker/`

- `faster-whisper-http/` — 로컬 Whisper ASR Docker
- `piper-http/` — 로컬 Piper TTS Docker

### `scripts/`

- `convert_bec_pdf.py` — WHO BEC PDF → Markdown 변환
- `convert_wilderness_pdf.py` — Red Cross PDF → Markdown 변환
- `split_wilderness.py` — wilderness.md 주제별 분할
- `install_ollama.sh` — Ollama 설치

---

## 하드웨어 사양 (Pi5)

- RAM: 8GB (전체 파이프라인 4.5GB 이내 목표)
- LLM: gemma3:1b (Ollama)
- 임베딩: gte-small (384차원)
- 레이턴시: avg ~32ms / p95 ~50ms (PC 기준)
