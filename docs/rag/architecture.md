# O.A.S.I.S. 아키텍처 상세

---

## 1. RAG 3-Stage 하이브리드 파이프라인

`retriever.py`가 구현하는 핵심 알고리즘입니다. Pocket RAG 논문 기반으로 Pi5의 제한된 리소스에 최적화했습니다.

```
Query
  │
  ▼
┌──────────────────────────────────────────┐
│  Stage 1: 의료 키워드 렉시컬 필터링        │
│  medical_keywords.py 역인덱스 검색        │
│  → 최대 50개 후보 청크                    │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Stage 2: 하이브리드 재랭킹               │
│  hybrid_score = 0.6×cosine + 0.4×lexical │
│  gte-small (384차원) 벡터 유사도          │
│  → 임계값(0.10) 이상 상위 3개 청크        │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  Stage 3: 컨텍스트 압축                   │
│  compressor.py 관련 문장만 추출           │
│  → 토큰 20~40% 절감                      │
└──────────────────┬───────────────────────┘
                   │
                   ▼
             RetrievalResult
           (context, chunks, latency_ms)
```

### 각 Stage 상세

**Stage 1 — 렉시컬 필터링**
- `medical_keywords.py`에 정의된 키워드 분류체계(Taxonomy)를 역인덱스로 구축
- 쿼리에서 의료 키워드 감지 → `keyword_map`에서 관련 청크 ID 목록 조회
- 키워드 매칭이 없으면 전체 FAISS 검색으로 폴백
- **목적:** 불필요한 벡터 유사도 계산 감소, Pi5 속도 최적화

**Stage 2 — 하이브리드 재랭킹**
```python
hybrid_score = 0.6 * cosine_similarity + 0.4 * lexical_overlap
```
- **cosine:** gte-small 임베딩 유사도 (384차원 L2-정규화 → IndexFlatIP)
- **lexical:** 쿼리-청크 간 단어 오버랩 비율
- 결합 점수로 의미적 + 어휘적 관련성을 동시에 반영

**Stage 3 — 컨텍스트 압축**
- 각 청크에서 쿼리와 관련 없는 문장 제거
- LLM의 컨텍스트 윈도우를 효율적으로 사용
- Pi5의 제한된 메모리와 속도를 고려한 설계

---

## 2. 인덱스 빌드 과정

```bash
cd python/oasis-rag
python indexer.py   # 최초 실행 또는 지식베이스 변경 시
```

```
data/knowledge/*.md
       │
       ▼ document_chunker.py
  청크 분할 (chunk_size=300, overlap=50)
       │
       ▼ sentence-transformers/gte-small
  384차원 임베딩 벡터 생성
       │
       ├──▶ data/rag_index/chunks.faiss      (FAISS IndexFlatIP)
       ├──▶ data/rag_index/metadata.json     (청크 텍스트, 소스, 섹션)
       └──▶ data/rag_index/keyword_map.json  (키워드 → 청크 ID 역인덱스)
```

**현재 인덱스 통계:**
- 문서 수: 27개 (WHO BEC 17개 + Red Cross 10개)
- 청크 수: 281개
- 임베딩 모델: `thenlper/gte-small` (384차원, ~33MB)

---

## 3. Context Injection 패턴

RAG가 정보를 찾아왔어도 0.8b 소형 모델은 컨텍스트를 무시하는 경우가 있습니다. 이를 해결하기 위해 **Context Injection** 패턴을 사용합니다.

### 동작 방식

```python
# chat_test.py / validation/test_llm_response.py 의 call_llm() 함수

q_lower = query.lower()

# 신호(Signal) 감지 → 프로토콜 주입
if any(sig in q_lower for sig in _CARDIAC_ARREST_SIGNALS):
    context = "CARDIAC ARREST PROTOCOL — ACT NOW:\n1. CALL..." + context

if any(sig in q_lower for sig in _BURN_SIGNALS):
    context = "⚠ BURN — MOST IMPORTANT FIRST ACTION:\n..." + context
```

RAG 컨텍스트 **앞에 prepend**하여 LLM이 프로토콜 지침을 먼저 읽도록 합니다.

### 구현된 신호 목록 (22종)

| 신호 변수 | 감지하는 상황 | 주입 내용 |
|-----------|--------------|-----------|
| `_CARDIAC_ARREST_SIGNALS` | 심정지, 호흡 없음 | CPR 30:2 프로토콜 |
| `_SPINAL_SIGNALS` | 척추 손상 의심 | 절대 이동 금지 |
| `_BURN_SIGNALS` | 화상 | 즉시 냉각 20분 |
| `_CHOKING_SIGNALS` | 기도 폐쇄 | 등 두드리기 + 하임리히 |
| `_SNAKEBITE_SIGNALS` | 뱀 교상 | 고정, 빨지 말 것 |
| `_HYPOTHERMIA_SIGNALS` | 저체온증 | 따뜻하게 (냉각 금지) |
| `_HEAT_STROKE_SIGNALS` | 열사병 | 즉시 냉각 |
| `_LIGHTNING_SIGNALS` | 낙뢰 | 낮게 웅크리기, 나무 금지 |
| `_FROSTBITE_SIGNALS` | 동상 | 따뜻한 물 재온 |
| `_PANIC_BLOOD_SIGNALS` | 대량 출혈 패닉 | 직접 압박 |
| `_NO_EPIPEN_SIGNALS` | 에피펜 없음 | 응급 호출 우선 |
| `_SEIZURE_SIGNALS` | 발작 | 억제 금지, 옆으로 |
| `_STROKE_SIGNALS` | 뇌졸중 | FAST 평가, 아스피린 금지 |
| `_DROWNING_SIGNALS` | 익수 | 구조 호흡 먼저 |
| `_POISONING_SIGNALS` | 중독 | 구토 유도 금지 |
| `_ELECTRIC_SHOCK_SIGNALS` | 감전 | 만지지 말 것 |
| `_INFANT_CPR_SIGNALS` | 영아 심정지 | 2손가락 압박 |
| `_EYE_CHEMICAL_SIGNALS` | 안구 화학물질 | 흐르는 물로 세척 |
| `_SHOCK_SIGNALS` | 쇼크 | 다리 올리기, 응급 호출 |
| `_FRACTURE_SIGNALS` | 골절 | 발견 상태 고정, 교정 금지 |
| `_ASTHMA_SIGNALS` | 천식 발작 | 앉히기, 누이지 말 것 |
| `_HEART_ATTACK_SIGNALS` | 심장마비 의심 | 아스피린, 응급 호출 |

### Context Injection 설계 원칙

1. **Prepend 원칙:** 대부분 RAG 컨텍스트 앞에 삽입 → LLM이 먼저 읽음
2. **우선순위 처리:** 여러 신호가 겹칠 때 더 위험한 쪽 우선 (예: 안구 화학 > 화상)
3. **비번호 형식:** 번호 매긴 주입은 LLM이 그 번호를 그대로 출력하는 경향 → 중요 지침은 비번호 문단으로 작성
4. **일관성:** `chat_test.py`와 `validation/test_llm_response.py` 양쪽 동일하게 유지

---

## 4. 지식베이스 구조

`data/knowledge/` 폴더의 모든 `.md` 파일이 인덱싱됩니다.

### 파일 헤더 형식

각 문서 상단에 메타데이터가 있어 검색 품질을 높입니다:

```markdown
# WHO BEC — CPR Protocol

**Source:** WHO Basic Emergency Care 2018
**Standard:** WHO / ICRC 2018
**Category:** Clinical Skills — Cardiopulmonary Resuscitation

[DOMAIN_TAGS: CPR, cardiac_arrest, chest_compressions, 30_2, ...]
```

- `[DOMAIN_TAGS: ...]` — BM25 렉시컬 검색 및 키워드 매핑에 사용
- 지식베이스 추가 시 이 형식을 따라야 Stage 1 필터링이 제대로 동작

### 현재 지식베이스

| 카테고리 | 파일 수 | 주요 내용 |
|----------|---------|-----------|
| WHO BEC 모듈 | 5개 | ABCDE, 외상, 호흡, 쇼크, 의식 변화 |
| WHO BEC 술기 | 11개 | CPR, 기도, 출혈, 화상, 고정, 창상 등 |
| Red Cross | 10개 | 낙뢰, 뱀교상, 골절, 화상, 익수, 열/냉 응급 등 |

### 지식베이스 추가 방법

```bash
# 1. data/knowledge/ 에 새 .md 파일 추가 (헤더 형식 준수)
# 2. 인덱스 재빌드
cd python/oasis-rag
python indexer.py

# 또는 서비스 실행 중 API로 재인덱스
curl -X POST http://localhost:5001/index
```

---

## 5. Flask API 상세

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | - | `{status, index_ready, chunk_count, model}` |
| `/retrieve` | POST | `{query, top_k?}` | `{context, chunks[], latency_ms, stage1_candidates, stage2_passing}` |
| `/index` | POST | - | `{status, chunk_count}` |

**`/retrieve` 응답 예시:**
```json
{
  "context": "CPR PROTOCOL:\n1. Call 911...\n\nSource: who_bec_skills_cpr.md",
  "chunks": [
    {
      "source": "who_bec_skills_cpr.md",
      "section": "Adult CPR — Step-by-Step",
      "hybrid_score": 0.74,
      "cosine_score": 0.71,
      "lexical_score": 0.80,
      "compressed_text": "Begin chest compressions..."
    }
  ],
  "latency_ms": 28.3,
  "stage1_candidates": 12,
  "stage2_passing": 3
}
```

---

## 6. TypeScript 브릿지

Node.js 챗봇과 Python RAG 서비스를 연결하는 부분입니다.

**`src/cloud-api/local/oasis-rag-client.ts`** — HTTP 클라이언트
- `isRagReady()` → 서비스 상태 확인
- `ragRetrieve(query)` → context 문자열 반환 (오류 시 빈 문자열)
- `ragRetrieveFull(query)` → 청크 메타데이터 포함 전체 응답

**`src/core/OasisAdapter.ts`** — 시스템 프롬프트 생성기
- RAG 컨텍스트를 받아 LLM용 시스템 프롬프트 조립
- RAG 불가 시 안전 폴백 프롬프트 반환 ("응급 서비스에 전화하세요")

**`src/core/ChatFlow.ts`** — 메인 대화 루프
```
STT → OasisAdapter.getSystemPromptFromOasis(query) → LLM → TTS
```

---

## 7. LLM 시스템 프롬프트

LLM에게 전달되는 시스템 프롬프트는 매우 엄격하게 제한되어 있습니다:

```
You are OASIS. A person needs first aid RIGHT NOW.

RULES YOU MUST FOLLOW:
- Your response is ONLY numbered steps 1 through 5.
- Do NOT write anything before "1."
- Each step is ONE sentence, maximum 12 words.
- Do NOT use asterisks, bold, markdown, or headers.
- Do NOT ask questions. Give commands only.

REFERENCE:
{RAG context + injections}

YOUR RESPONSE MUST START WITH "1." AND END AFTER STEP 5.
```

**설계 이유:**
- 실제 응급 상황에서 5단계 이하의 명확한 행동 지침이 필요
- TTS 출력을 위해 마크다운/서식 제거
- 소형 모델(0.8b)의 한계를 감안한 엄격한 포맷 제약

---

## 8. 하드웨어 제약

| 항목 | 제약 |
|------|------|
| 디바이스 | Raspberry Pi 5 |
| 메모리 | 전체 파이프라인 4.5GB 이내 |
| LLM | qwen3.5:0.8b (1.0GB, Ollama) |
| 임베딩 | gte-small (33MB) |
| 벡터DB | FAISS 인메모리 (CPU) |
| 레이턴시 목표 | RAG < 2000ms, LLM < 60초 |
| 인터넷 | 완전 오프라인 |
