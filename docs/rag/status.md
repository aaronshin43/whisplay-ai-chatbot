# 현재 상태 및 로드맵

---

## 현재 검증 결과 (2026-03-16 기준)

| 테스트 | 결과 | 목표 |
|--------|------|------|
| RAG 파이프라인 (109개) | **109/109 (100%)** ✅ | 109/109 |
| LLM+RAG 통합 (35개) | **34/35 (97.1%)** ✅ | 33/35+ |
| CONTENT 정확도 | **35/35 (100%)** ✅ | - |
| FORMAT 준수 | **35/35 (100%)** ✅ | - |
| SAFE (안전성) | **34/35 (97.1%)** ✅ | 100% 목표 |
| RAG 평균 레이턴시 | **~32ms (PC)** ✅ | <200ms |

---

## 알려진 버그

### [BUG-001] LLM-008 — 번개 쿼리 RAG 오검색 🔴

**심각도:** 높음 (안전 관련)
**상태:** 미해결

**현상:**
```
쿼리: "lightning coming no shelter in open field"
RAG 검색 결과:
  1위: redcross_altitude.md  ← 잘못된 문서
  2위: redcross_lightning.md
  3위: redcross_altitude.md

LLM 응답 (1단계): "Move downhill to a low rolling hill or tree."
                                                          ^^^^
                                          나무로 가라는 위험한 조언
```

**원인:**
`redcross_altitude.md`가 번개 쿼리에서 1위로 검색됩니다. 이유:
- 고산병 문서에 "open area", "shelter", "move downhill" 표현 포함
- FAISS 벡터 유사도가 `redcross_lightning.md`보다 높게 나옴
- 소형 모델이 1위 문서(고산병)를 따라 "나무로 이동" 조언 생성

**시도한 해결책:**
- Context Injection (prepend + append 동시): 효과 없음 — 0.8b 모델이 1위 RAG 문서를 우선 따름
- 주입 강도 강화 ("NEVER go near trees"): 효과 없음

**해결 방향:**
1. `redcross_altitude.md` 청킹 재검토 — "open area" 관련 문장을 별도 청크로 분리하거나 제거
2. `redcross_lightning.md`에 DOMAIN_TAGS 강화 (쿼리 키워드와 일치하도록)
3. 또는 Stage 1 키워드 필터에 "lightning", "thunder"를 명시적 도메인으로 추가

---

## 완료된 작업 목록

- [x] RAG 3-Stage 하이브리드 파이프라인 구현
- [x] 지식베이스 구축 (WHO BEC 17개 + Red Cross 10개 = 27개 문서, 281 청크)
- [x] Flask HTTP 서비스 래퍼 (`/health`, `/retrieve`, `/index`)
- [x] TypeScript 브릿지 (`oasis-rag-client.ts`, `OasisAdapter.ts`)
- [x] RAG 파이프라인 검증 109/109 (100%)
- [x] LLM 모델 비교 평가 → qwen3.5:0.8b 채택
- [x] Context Injection 22종 구현
- [x] LLM+RAG 통합 테스트 35개 작성 (엄격한 기준)
- [x] LLM+RAG 통합 테스트 34/35 (97.1%) 달성
- [x] `who_bec_skills_cpr.md` 신규 작성 (기존 지식베이스에 CPR 실제 단계 없었음)

---

## 남은 작업 (우선순위 순)

### 🔴 긴급 (출시 전 필수)

**1. BUG-001 수정 — 번개 RAG 오검색**
```
담당: RAG 팀
작업: redcross_altitude.md 재청킹 또는 lightning DOMAIN_TAGS 강화
검증: LLM-008 PASS 확인
```

**2. Pi5 실측 레이턴시 측정**
```
현재: PC 추정값(×12배 = ~64초) 사용
필요: 실제 Pi5에서 측정 후 목표값 재설정
참고: validation/test_latency.py 로 실측 가능
```

### 🟡 중요 (1차 출시 후)

**3. Node.js ChatFlow 통합 완성**
```
현재: OasisAdapter.ts 구현됨, ChatFlow.ts와 연결 필요
작업: STT → RAG → LLM → TTS 전체 루프 E2E 테스트
```

**4. Context Injection을 OasisAdapter.ts로 이전**
```
현재: chat_test.py와 test_llm_response.py에 중복 구현됨
문제: 두 파일 동기화 누락 위험
이상적: OasisAdapter.ts가 단일 진실 소스(single source of truth)
```

**5. 안전 폴백 강화**
```
현재: RAG 서비스 다운 시 "응급 서비스 전화" 메시지 출력
개선: 오프라인 최소 핵심 프로토콜 5개를 하드코딩으로 내장
      (심정지 CPR, 출혈, 아나필락시스, 기도폐쇄, 화상)
```

### 🟢 장기 개선

**6. 다국어 쿼리 지원**
```
현재: 영어 쿼리만 테스트됨
개선: 한국어, 스페인어 등 다국어 쿼리 → 번역 후 검색
```

**7. 음성 활성화 (Wake Word)**
```
개선: "OASIS" 또는 "Help" 호출어로 대기 → 응급 쿼리 감지
```

**8. 번개 및 날씨 응급 확장**
```
개선: redcross_lightning.md 콘텐츠 강화
      토네이도, 홍수, 지진 등 자연재해 대응 추가
```

---

## 모델 업그레이드 경로

현재 `qwen3.5:0.8b`는 Pi5 메모리 제약 내에서 최선이지만, 한계가 있습니다.

| 모델 | 크기 | 장점 | 단점 |
|------|------|------|------|
| `qwen3.5:0.8b` (현재) | 1.0 GB | CPR 100%, 안전성 100% | 64초 Pi5 추정, 주입 무시 |
| `gemma3:1b` | 815 MB | 빠름 (~62초) | CPR 0% (치명적 결함) |
| `gemma3:4b` | 2.5 GB | 더 정확 | Pi5 메모리 4.5GB 예산 위협 |
| `phi4` | ~2 GB | 의료 지식 강 | Pi5 테스트 필요 |

**권장:** Pi5 출시 전 `gemma3:4b` Pi5 실측 후 결정

---

## 개발 환경 메모

### 인덱스 재빌드가 필요한 경우
- `data/knowledge/` 에 파일 추가/수정
- `document_chunker.py` 청킹 로직 변경
- `medical_keywords.py` 키워드 추가

```bash
cd python/oasis-rag
python indexer.py
# "Index saved: 281 chunks" 확인
```

### 자주 겪는 문제

**RAG 서비스가 시작되지 않음:**
```bash
# 인덱스 파일이 없는 경우
python indexer.py  # 먼저 인덱스 빌드
python service.py
```

**Ollama 모델 없음:**
```bash
ollama pull qwen3.5:0.8b
```

**cp949 인코딩 오류 (Windows):**
```python
# 파일 상단에 이미 있음
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
```

**qwen3.5 응답이 비어있음:**
```python
# Qwen3 계열은 think 모드를 비활성화해야 함
payload["think"] = False  # 이미 적용됨
```

---

## 관련 문서

| 파일 | 내용 |
|------|------|
| `CLAUDE.md` | 프로젝트 전체 맥락 (Claude Code용) |
| `validation/results/model_evaluation_report.md` | LLM 모델 비교 평가 보고서 (한국어) |
| `validation/results/summary.json` | 최신 RAG 검증 결과 |
| `validation/results/llm_response_test.json` | 최신 LLM 통합 테스트 결과 |
