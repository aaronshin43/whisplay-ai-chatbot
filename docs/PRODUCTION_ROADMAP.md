# O.A.S.I.S. 상용화 로드맵

> **목표**: RAG + LLM 성능을 오프라인 응급처치 AI 디바이스(Raspberry Pi 5)의 실제 상용화 수준까지 높인다.

---

## 현재 상태 (2026-03-17 기준)

| 항목 | 값 |
|------|-----|
| 검증 테스트 | 108/109 (99.1%) |
| 유일한 실패 | BRN-001 ("ice" 거짓 양성 — 기존 문제) |
| 청크 수 | 198개 |
| 지식베이스 | 17개 문서 (WHO BEC 2018 + Red Cross Wilderness) |
| 임베딩 | gte-small (384dim), FAISS in-메모리 |
| LLM | gemma3:1b (Ollama) |

### 발견된 치명적 구조 문제

1. **Context Injection 20개 프로덕션 누락**
   - `chat_test.py`에 22개 응급 상황별 context injection 존재
   - `OasisAdapter.ts`(실제 프로덕션 경로)에는 **spinal, seizure 2개만 구현**
   - burns, cardiac arrest, choking, hypothermia 등 20개 프로토콜이 실기기에서 누락

2. **service.py 동시성 버그**
   - `retriever.top_k`를 요청마다 직접 변경 → 동시 요청 시 레이스 컨디션

3. **gemma3:1b 프로덕션 부적합**
   - 22개 context injection이 존재하는 근본 이유: 모델이 RAG 컨텍스트만으로 올바른 프로토콜을 추출하지 못함
   - 3b-4b Q4 모델로 업그레이드 필요

---

## 타임라인 요약

```
Phase 1 ████░░░░░░░░░░░░░░░░  주 1-3    기반 안정화
Phase 2 ░░░░████████░░░░░░░░  주 4-7    RAG 강화 + KB 확장
Phase 3 ░░░░░░░░████████░░░░  주 8-11   LLM 품질 + 안전
Phase 4 ░░░░░░░░░░░░████░░░░  주 12-14  테스트 250+
Phase 5 ░░░░░░░░░░░░░░░████░  주 15-17  Pi5 최적화 + 배포
Phase 6 ░░░░░░░░░░░░░░░░░███→ 주 18+    안전성 검증 (지속)
```

**총 예상: 17-20주 (4-5개월)**

---

## Phase 1: 기반 안정화 (2-3주)

**목표**: 버그 수정, context injection 통합, 섹션 인식 청킹 도입

### 1.1 BRN-001 테스트 정의 수정 (15분)
- **문제**: `must_not_contain: ["ice"]`가 `"Do NOT apply ice"`(올바른 조언)를 거짓 양성으로 잡음
- **수정**: `["apply ice", "use ice"]`로 변경 (위험한 조언을 감지, 단어 자체가 아님)
- **효과**: 109/109 PASS 달성
- **파일**: `python/oasis-rag/validation/test_retrieval_accuracy.py` line ~118

### 1.2 Context Injection 통합 모듈 (1-2일)
- **문제**: 22개 signal이 `chat_test.py`, `test_llm_response.py`, `OasisAdapter.ts`에 중복 (대부분 누락)
- **수정**:
  1. `python/oasis-rag/context_injector.py` 신규 생성 — 22개 signal 단일 소스
  2. `service.py`의 `/retrieve` 엔드포인트에서 context injection 적용 후 반환
  3. `OasisAdapter.ts`의 하드코딩 제거 — Python에서 받은 enriched context 사용
  4. `chat_test.py`, `test_llm_response.py`도 `context_injector.py` import로 대체
- **효과**: 프로덕션 경로에서 20개 프로토콜 복구
- **성공 기준**: `OasisAdapter.ts`에 하드코딩 의료 내용 없음

### 1.3 service.py 동시성 버그 수정 (30분)
- **문제**: 공유 `_retriever` 인스턴스의 `top_k`, `compress`를 요청마다 직접 변경
- **수정**: 요청별 파라미터를 `retrieve()` 메서드 인자로 전달
- **파일**: `python/oasis-rag/service.py` lines 179-182

### 1.4 섹션 인식 청킹 (1-2일)
- **문제**: 300 토큰 슬라이딩 윈도우가 `### Broken Finger`와 `### Rib Fracture`를 같은 청크에 묶음
- **수정**:
  1. `SectionAwareChunker` 클래스 도입 in `document_chunker.py`
  2. H3 경계에서 분리, 단 80 토큰 미만 섹션은 같은 H2 부모 내에서 병합
  3. 각 청크에 `body_part`, `condition`, `section_path` 메타데이터 추가
  4. 인덱스 재빌드 후 전체 검증
- **이전 시도 실패 원인**: 모든 `###`에서 무조건 분리 → 20 토큰 미니 청크 547개 (임베딩 품질 저하)
- **이번 핵심**: 최소 크기 병합 + 신체부위 불일치 시 병합 금지
- **성공 기준**: "broken finger" 쿼리에서 chest/rib 관련 문장 0개

**Phase 1 완료 기준**: 109/109 PASS + anatomy mismatch 해결

---

## Phase 2: RAG 파이프라인 강화 (3-4주)

**목표**: context injection 없이 RAG만으로 모든 시나리오 커버 가능하게 만들기

### 2.1 쿼리 분류 레이어 (1-2일)
- **신규 파일**: `python/oasis-rag/query_classifier.py`
- 기존 `medical_keywords.py` taxonomy + 휴리스틱 규칙 활용
- 출력: `QueryClassification(emergency_type, body_part, severity, confidence)`
- 다중 부상 시 가장 위험한 상태 우선 처리
- **성공 기준**: 100개 쿼리 벤치마크에서 95%+ 정확도

### 2.2 신체부위 필터링 (1일)
- Stage 1에서 쿼리 신체부위와 불일치 청크에 패널티 적용
- `retriever.py`의 `_stage1_lexical` 또는 `_stage2_semantic` 수정
- **효과**: "broken finger" 쿼리 시 chest/rib/leg 관련 청크 우선순위 하락

### 2.3 Compressor 안전성 강화 (1일)
- `"Do NOT"`, `"Never"`, `"Avoid"` + 쿼리 키워드 포함 문장은 항상 보존 (현재 압축으로 제거될 수 있음)
- 처방 약물명 필터링 — LLM이 보기 전에 제거 (defense-in-depth)
- 번호 매긴 단계가 있는 출처는 번호 구조 유지
- **파일**: `python/oasis-rag/compressor.py`

### 2.4 지식베이스 확장 (1-2주)

현재 22개 context injection → RAG KB로 흡수하기 위한 필수 문서:

| 우선도 | 문서명 | 이유 |
|--------|--------|------|
| **높음** | `who_bec_seizure_epilepsy.md` | 발작 프로토콜 — injection으로만 존재 |
| **높음** | `who_bec_stroke.md` | 뇌졸중 FAST — injection으로만 존재 |
| **높음** | `who_bec_choking_airway.md` | 기도 폐쇄 상세 지침 부실 |
| **높음** | `who_bec_poisoning.md` | 중독 대응 — injection으로만 존재 |
| **높음** | `redcross_electric_shock.md` | 감전 — injection으로만 존재 |
| **높음** | `who_bec_diabetic_emergency.md` | 당뇨 응급 — AMS-002 근거 부족 |
| **높음** | `who_bec_cardiac_arrest.md` | 심정지 — injection으로만 존재 |
| **중간** | `who_bec_heart_attack.md` | 심장마비 (심정지와 구분) |
| **중간** | `who_bec_infant_child_cpr.md` | 소아 CPR 차이점 |
| **중간** | `redcross_eye_injuries.md` | 화학물질 눈 손상 |
| **중간** | `who_bec_shock_first_aid.md` | 응급처치 수준 쇼크 관리 |
| **낮음** | `who_bec_drug_overdose.md` | AMS-005 부분 커버 |
| **낮음** | `redcross_marine_stings.md` | 해양 생물 교상 |

모든 문서는 WHO BEC 2018, IFRC, Red Cross 공개 자료 기반으로 작성.

### 2.5 임베딩에 헤딩 컨텍스트 포함 (15분)
- **현재**: `indexer.py`에서 `m["text"]`만 임베딩
- **변경**: `m["text_with_prefix"]` (헤딩 포함)로 변경
- 임베딩 벡터에 섹션 의미 포함 → "Broken Finger" 청크와 "Rib Fracture" 청크 벡터 거리 증가
- **파일**: `python/oasis-rag/indexer.py` line ~149

### 2.6 하이퍼파라미터 재조정
KB 확장 후 (예상 ~40+ 문서, 400+ 청크):
- `CHUNK_SIZE`: 300 → 200-250 토큰 (더 정밀한 청크)
- `CHUNK_OVERLAP`: 50 → 30-40 토큰
- `TOP_K`: 3 → 4-5
- `ALPHA`: 0.6/0.4 비율 재검증
- `SCORE_THRESHOLD`: 재조정

**Phase 2 완료 기준**: context injection 없이 200+ 테스트 통과, 모든 시나리오 KB로 커버

---

## Phase 3: LLM 품질 & 안전 (3-4주)

### 3.1 LLM 모델 평가 및 업그레이드

**Pi5 메모리 예산 (8GB 총):**
```
OS + 시스템:      ~1.5GB
Flask + FAISS:    ~1.2GB
STT + TTS:        ~0.5GB
─────────────────────────
LLM 여유:         ~4.8GB
```

**후보 모델:**

| 모델 | 메모리 | Pi5 응답 | 지시 준수 | 추천 |
|------|--------|---------|----------|------|
| gemma3:1b (현재) | ~1.5GB | 3-5s | 낮음 | ❌ 프로덕션 부적합 |
| **phi-4-mini Q4** | ~2.8GB | 7-10s | 높음 | ✅ 1순위 |
| **gemma3:4b Q4** | ~3.0GB | 8-12s | 높음 | ✅ 2순위 |
| qwen2.5:3b Q4 | ~2.4GB | 6-8s | 중간 | 🟡 대안 |

**레이턴시 목표 수정**: PC < 5초, **Pi5 < 12초** (3b+ 모델 기준)

### 3.2 구조화된 출력 검증
- 응답이 번호 목록 포맷을 따르는지 검증 (regex: `^\d+\.\s.+`)
- 실패 시 1회 재시도 (더 엄격한 프롬프트)
- 2회 실패 시 context injection 텍스트를 직접 반환
- **신규 파일**: `python/oasis-rag/response_validator.py`

### 3.3 환각 검출
- 응답의 각 단계가 RAG 컨텍스트에 근거하는지 확인
- 키워드 오버랩으로 "grounding score" 계산
- medical_keywords에 속하는 행동 중 컨텍스트에 없는 것은 차단
- **신규 파일**: `python/oasis-rag/hallucination_detector.py`

### 3.4 처방 약물명 RAG 수준 차단
- `compressor.py`에서 처방약 언급 사전 제거 (LLM이 보기 전)
- defense-in-depth: KB에 임상 맥락으로 약물명이 있더라도 LLM까지 전달 안 됨

---

## Phase 4: 테스트 스위트 확장 (2-3주)

| 카테고리 | 현재 | 목표 |
|---------|------|------|
| 검색 정확도 | 47개 | 100+ |
| 안전성 | 8개 | 30+ |
| LLM 응답 품질 | 35개 | 60+ |
| E2E 통합 | 0개 | 30+ |
| **총계** | **109개** | **250+** |

**추가 테스트 항목:**
- 신체부위별 해부학 정확도 (손가락 vs 팔 vs 다리)
- 다중 부상 우선순위 분류
- 위험한 민간요법 차단 (버터 바르기, 발작 시 입에 물건 넣기)
- 범위 초과 요청 거절 (수술, 봉합, 약물 처방)
- 한국어/영어 혼용 쿼리

---

## Phase 5: Pi5 최적화 & 배포 (2-3주)

- 실기기 메모리 프로파일링
- 시작 시간 30초 이내 (ONNX 최적화, mmap FAISS)
- systemd 서비스 유닛 (부팅 후 60초 이내 Ready)
- 오프라인 복원력 (인덱스 손상 시 자동 재빌드, 워치독)

---

## Phase 6: 안전성 검증 프레임워크 (지속)

- **안전성 매트릭스**: 전체 지원 시나리오 → 출처 문서 + 테스트 ID 매핑 (감사 가능한 아티팩트)
- **레드팀 테스트**: 50개 적대적 쿼리 (혼합 상태, 오타, 유해 조언 유도)
- **범위 초과 탐지**: cosine < 0.5 → "응급 서비스에 전화하세요" 응답
- **배포 후 모니터링**: 모든 쿼리/응답 로컬 로깅, 주간 검토

---

## 현재 완료된 개선 사항

- ✅ 프롬프트: 5줄 고정 → 최대 7스텝, `"6."` stop 토큰 제거
- ✅ 이중 system prompt 제거 (`ollama-llm.ts`)
- ✅ Source diversity 필터 도입 (`MAX_PER_SOURCE=2`) — TRM-002 수정
- ✅ `COMPRESS_MIN_RATIO` 0.60 → 0.40 — 불필요 문장 강제 포함 감소
- ✅ 108/109 PASS (원본 107/109 대비 개선)
