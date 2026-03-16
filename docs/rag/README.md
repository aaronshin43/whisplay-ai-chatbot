# O.A.S.I.S. RAG 브랜치 인수인계

> **O.A.S.I.S.** = Offline Autonomous Safety Intelligence System
> 오프라인 응급처치 AI 디바이스 (Raspberry Pi 5 탑재용)

---

## 이 브랜치가 뭔가요?

`rag` 브랜치는 **O.A.S.I.S.의 핵심 두뇌**를 구현합니다.
사용자가 응급 상황을 음성으로 말하면, 디바이스가 WHO/Red Cross 공인 응급처치 매뉴얼에서 관련 정보를 찾아 LLM에게 전달하고, LLM이 5단계 행동 지침을 생성합니다.

**인터넷 없이, Raspberry Pi 5에서, 5초 이내 응답.**

```
사용자 음성 → STT → [RAG 검색 ~32ms] → [LLM 응답 ~5초] → TTS → 음성 출력
```

---

## 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                     Node.js Chatbot                         │
│   STT → ChatFlow.ts → OasisAdapter.ts → TTS                 │
│                           │                                 │
│              oasis-rag-client.ts (HTTP)                     │
└───────────────────────────┼─────────────────────────────────┘
                            │ localhost:5001
┌───────────────────────────┼─────────────────────────────────┐
│            Python Flask RAG Service                         │
│                           │                                 │
│   service.py  ←→  retriever.py  ←→  indexer.py             │
│                           │                                 │
│        [FAISS 벡터 인덱스 + keyword_map]                    │
│                           │                                 │
│              data/knowledge/ (27개 MD 파일)                 │
└───────────────────────────┼─────────────────────────────────┘
                            │ localhost:11434
┌───────────────────────────┼─────────────────────────────────┐
│            Ollama LLM Service                               │
│              qwen3.5:0.8b (선택 모델)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 빠른 시작 (로컬 개발)

### 사전 준비

```bash
# 1. Python 패키지 설치
cd python/oasis-rag
pip install -r requirements.txt

# 2. Ollama 설치 및 모델 다운로드
# https://ollama.com 에서 설치
ollama pull qwen3.5:0.8b    # 권장 모델 (1.0 GB)
ollama pull gemma3:1b        # 대안 모델 (815 MB)
```

### 서비스 시작 순서

**터미널 1 — Ollama LLM**
```bash
ollama serve
```

**터미널 2 — RAG Flask 서비스**
```bash
cd python/oasis-rag
python service.py
# → "RAG service ready on :5001 (281 chunks)" 출력 확인
```

**터미널 3 — Node.js 챗봇 (선택)**
```bash
npm start
```

### 서비스 상태 확인

```bash
# RAG 서비스 헬스체크
curl http://localhost:5001/health

# 응답 예시:
# {"status":"ok","index_ready":true,"chunk_count":281,"model":"thenlper/gte-small"}
```

---

## CLI로 직접 질문하기

RAG + LLM 전체 파이프라인을 터미널에서 바로 테스트할 수 있습니다.

```bash
cd python/oasis-rag

# 기본 모델(gemma3:1b)로 실행
python chat_test.py

# 다른 모델 지정
python chat_test.py --model qwen3.5:0.8b
python chat_test.py --model gemma3:4b
```

**실행 화면 예시:**
```
  O.A.S.I.S. Interactive Test CLI
  ─────────────────────────────────────────────
  [OK] RAG service  →  localhost:5001
  [OK] Ollama       →  localhost:11434  (qwen3.5:0.8b)

  Type your query below. Enter "quit" or Ctrl-C to exit.

OASIS> she collapsed not breathing no pulse

  [RAG] 3 chunk(s) found (28ms) | top: who_bec_skills_cpr.md (0.74)
        also: who_bec_module1_abcde.md (0.71), who_bec_module2_trauma.md (0.68)
  [LLM] qwen3.5:0.8b (5.3s)

1. Call emergency services (911/999/112) immediately.
2. Place the person on a firm, flat surface.
3. Begin chest compressions: push hard and fast on centre of chest.
4. Rate: 100-120 compressions per minute, depth 5-6 cm.
5. Continue 30:2 cycle (compressions:breaths) until help arrives.
```

**CLI에서 확인할 수 있는 정보:**
- `[RAG]` — 검색된 청크 수, 응답 시간, 소스 파일, 유사도 점수
- `[LLM]` — 사용 모델, 생성 시간
- 최종 응답 (5단계 행동 지침)

---

## 주요 파일 한눈에 보기

```
whisplay-ai-chatbot-1/
│
├── python/oasis-rag/              ← RAG 핵심 코드
│   ├── service.py                 ← Flask HTTP 서버 (:5001)
│   ├── retriever.py               ← 3-Stage 하이브리드 검색 엔진
│   ├── indexer.py                 ← FAISS 인덱스 빌더
│   ├── compressor.py              ← 컨텍스트 압축
│   ├── document_chunker.py        ← 문서 청킹
│   ├── medical_keywords.py        ← 의료 키워드 분류체계
│   ├── config.py                  ← 설정값 중앙화
│   ├── chat_test.py               ← CLI 대화 테스트 도구
│   └── validation/                ← 자동화 검증 스위트
│       ├── run_all.py             ← 전체 테스트 실행 (109개)
│       ├── test_llm_response.py   ← LLM+RAG 통합 테스트 (35개)
│       ├── compare_models.py      ← 모델 비교 도구
│       └── results/               ← 테스트 결과 JSON
│
├── data/
│   ├── knowledge/                 ← 응급처치 지식베이스 (27개 MD)
│   │   ├── who_bec_module*.md     ← WHO Basic Emergency Care
│   │   ├── who_bec_skills_*.md    ← WHO 술기 가이드
│   │   └── redcross_*.md          ← Red Cross 야외 응급처치
│   └── rag_index/                 ← 빌드 산출물 (.gitignore)
│       ├── chunks.faiss
│       ├── metadata.json
│       └── keyword_map.json
│
├── src/
│   ├── core/
│   │   ├── ChatFlow.ts            ← 메인 대화 루프
│   │   └── OasisAdapter.ts        ← RAG → LLM 시스템 프롬프트 변환
│   └── cloud-api/local/
│       └── oasis-rag-client.ts    ← RAG 서비스 HTTP 클라이언트
│
└── docs/rag/                      ← 이 문서들
```

---

## 다음 단계

- [아키텍처 상세](architecture.md) — RAG 3단계 파이프라인, Context Injection 패턴
- [테스트 가이드](testing.md) — 검증 테스트 실행 방법
- [현재 상태 및 로드맵](status.md) — 알려진 문제점, 앞으로 해야 할 일
