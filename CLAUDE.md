# OASIS Project Context

오프라인 응급처치 AI 디바이스 (Raspberry Pi 5) 소프트웨어.
Whisplay chatbot fork 기반, Pocket RAG 논문의 3-Stage Hybrid RAG 구현 완료.

## 현재 상태 (Phase 1 + Phase 2 완료)

- RAG 파이프라인: **완성** (Python Flask + FAISS + gte-small)
- 지식베이스: 33 문서, **374 청크** (Phase 2 신규 문서 4개 추가)
- 검증: **109/109 PASS** (정확도 100%, 안전 100%, 커버리지 100%, 레이턴시 avg ~25ms)
- TS 브릿지: `src/cloud-api/local/oasis-rag-client.ts` → RAG Flask 서비스 연결
- Context Injection: **22개 신호 통합** → `python/oasis-rag/context_injector.py` (단일 소스)
- 청킹: `SectionAwareChunker` (H3 경계 분리 + 최소 크기 병합)
- 임베딩: `text_with_prefix` (헤딩 컨텍스트 포함)
- 쿼리 분류: `query_classifier.py` + 신체부위 필터링

## 서비스 시작 순서

```bash
ollama serve                          # 1. LLM (gemma3:1b)
cd python/oasis-rag && python app.py  # 2. RAG Flask (:5001)
npm start                             # 3. Node.js chatbot
```

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `src/core/ChatFlow.ts` | 메인 대화 루프 (STT → RAG → LLM → TTS) |
| `src/core/OasisAdapter.ts` | RAG 결과 → LLM 시스템 프롬프트 변환 |
| `src/cloud-api/local/oasis-rag-client.ts` | RAG 서비스 HTTP 클라이언트 (주 경로) |
| `src/cloud-api/local/oasis-matcher-node.ts` | **[FALLBACK]** RAG 서비스 다운 시 프로토콜 매칭 |
| `python/oasis-rag/service.py` | RAG Flask 서버 |
| `python/oasis-rag/retriever.py` | 3-Stage Hybrid Retriever |
| `python/oasis-rag/indexer.py` | FAISS 인덱스 빌더 |
| `python/oasis-rag/medical_keywords.py` | 의료 키워드 택소노미 (역인덱스) |
| `python/oasis-rag/context_injector.py` | 22개 응급 상황 context injection 단일 소스 |
| `python/oasis-rag/query_classifier.py` | 쿼리 분류 (응급 유형, 신체부위, 중증도) |
| `python/oasis-rag/validation/run_all.py` | 전체 검증 실행 (109개 테스트) |

## RAG 아키텍처

```
Query
 → QueryClassifier (응급 유형, 신체부위 감지)
 → Stage 1: FAISS 벡터 검색 (gte-small 384차원, top-50)
 → Stage 2: Hybrid 재랭킹 (cosine 0.6 + BM25 0.4) + 신체부위 패널티 → top-4
 → Stage 3: 컨텍스트 압축 (compressor.py, 안전 문장 보존)
 → context_injector (22개 신호 기반 프로토콜 주입)
 → RetrievalResult (.chunks, .context, .latency_ms)
```

## 지식베이스

| 파일 | 출처 |
|------|------|
| `data/knowledge/who_bec_module1~5.md` | WHO Basic Emergency Care 2018 (모듈 1~5) |
| `data/knowledge/who_bec_quick_cards.md` | WHO BEC 빠른 참조 카드 |
| `data/knowledge/who_bec_skills_*.md` | WHO BEC 술기 가이드 (출혈, CPR, 기도, 상처 등) |
| `data/knowledge/who_bec_chest_pain_cardiac.md` | WHO BEC — 심장마비/흉통 |
| `data/knowledge/who_bec_poisoning_overdose.md` | WHO BEC — 중독/과다복용 |
| `data/knowledge/who_bec_seizure_epilepsy.md` | WHO BEC — 발작/간질 (신규) |
| `data/knowledge/who_bec_stroke.md` | WHO BEC — 뇌졸중 FAST (신규) |
| `data/knowledge/who_bec_diabetic_emergency.md` | WHO BEC — 당뇨 응급 (신규) |
| `data/knowledge/redcross_electric_shock.md` | Red Cross — 감전 (신규) |
| `data/knowledge/redcross_altitude.md` | Red Cross — 고산병 |
| `data/knowledge/redcross_bites_stings.md` | Red Cross — 교상/자상/알레르기 |
| `data/knowledge/redcross_bone_joint.md` | Red Cross — 골절/관절 |
| `data/knowledge/redcross_burns.md` | Red Cross — 화상 |
| `data/knowledge/redcross_cold_emergencies.md` | Red Cross — 저체온/동상 |
| `data/knowledge/redcross_heat_emergencies.md` | Red Cross — 열사병 |
| `data/knowledge/redcross_lightning.md` | Red Cross — 낙뢰 |
| `data/knowledge/redcross_special.md` | Red Cross — 특수 상황 |
| `data/knowledge/redcross_submersion.md` | Red Cross — 익수 |
| `data/knowledge/redcross_wounds.md` | Red Cross — 창상 |
| `data/knowledge/pediatric_emergency_consolidated.md` | 소아 응급 통합 |

인덱스: `data/rag_index/` (빌드 산출물, .gitignore 처리됨)

## 규칙

- 의료 조언 직접 생성 금지 — RAG로 검증된 매뉴얼만 참조
- Pi5 메모리 제한: 전체 파이프라인 4.5GB 이내
- LLM: gemma3:1b (Ollama)
- 임베딩: gte-small (sentence-transformers, 384차원)
- 벡터DB: FAISS (인메모리)
- 레이턴시 목표: PC < 200ms, Pi5 < 2000ms
