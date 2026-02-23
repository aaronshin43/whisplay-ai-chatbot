# OASIS 최적화 작업 로그

> 작업일: 2026-02-23
> 대상: OASIS (Offline AI System for Immediate Survival) — 오프라인 응급처치 AI 키트

---

## 1. 배경

- 하드웨어: Raspberry Pi 5 (8GB)
- 모델: `qwen3:1.7b` (Ollama)
- 문제: 프로토콜 검색 후 LLM 응답까지 **Time to First Token(TTFT)이 ~7,300ms**로, 응급 상황에서 너무 느림
- 목표: TTFT를 줄이고, 프로토콜 기반 응답의 정확도를 유지

---

## 2. 변경 사항

### 2.1 기본 시스템 프롬프트 중복 제거

**파일:** `src/cloud-api/local/ollama-llm.ts`

**문제:** OASIS 사용 시 Ollama에 메시지가 `[기본 system("cheerful girl"), OASIS system, user]`로 전달되어 시스템 메시지가 2개 들어감.

**수정:** `chatWithLLMStream`에서 input의 첫 메시지가 system이면 기본 system을 넣지 않도록 변경. 같은 system이 이미 있으면 user만 추가(KV 캐시 친화), 다른 system이면 교체.

### 2.2 OASIS 기본 활성화 + RAG 분기 정리

**파일:** `src/core/ChatFlow.ts`

**문제:** `useOasis = process.env.ENABLE_OASIS_MATCHER === "true" || true`로 항상 true. RAG 분기는 데드 코드.

**수정:** `useOasis = process.env.ENABLE_OASIS_MATCHER !== "false"`로 변경. 기본은 OASIS 사용, `ENABLE_OASIS_MATCHER=false`일 때만 RAG 사용 가능.

### 2.3 프로토콜 중복 스킵 로직 제거

**파일:** `src/core/ChatFlow.ts`

**문제:** `knowledgePrompts.includes(res)`로 같은 세션에서 동일 프로토콜이 재매칭되면 system을 비움 → 두 번째 질문부터 프로토콜 없이 응답.

**수정:** `knowledgePrompts` 배열과 중복 체크/push 로직 전체 제거. 매 질문마다 매칭된 프로토콜을 항상 system으로 주입.

### 2.4 keepAlive 시스템 프롬프트 최소화

**파일:** `src/cloud-api/local/ollama-llm.ts`

**문제:** 앱 기동 시 모델 로드용 keepAlive 요청에 기본 캐릭터 프롬프트("You are a young and cheerful girl...")를 사용.

**수정:** `keepAliveSystemPrompt = "You are a helpful assistant."`로 최소화. 실제 대화는 OASIS 프롬프트만 사용.

### 2.5 Ollama KV 캐시 워밍업 구현

**파일:** `src/cloud-api/local/ollama-llm.ts`, `src/cloud-api/llm.ts`

Ollama는 이전 요청과 프롬프트 prefix가 동일하면 KV 캐시를 재사용하여 prefill을 건너뜀. 이를 활용하기 위해:

- `warmupSystemPrompt(systemContent)` 함수 추가: 프로토콜 매칭 직후 system prompt만 Ollama에 전송 (`num_predict: 0`)하여 KV 캐시를 미리 준비
- `llm.ts`에서 `warmupSystemPrompt` export하여 테스트/앱 코드에서 사용 가능

### 2.6 시스템 메시지 중복 누적 방지

**파일:** `src/cloud-api/local/ollama-llm.ts`

**문제:** 연속 질문 시 매번 `[system, user]`를 push하면 `[system, user1, assistant1, system, user2, ...]`로 system이 중복 누적 → KV 캐시 미스 + TTFT 점점 증가.

**수정:** `sameSystemAsBefore` 플래그 도입. 같은 system이면 user만 추가하여 메시지가 `[system, user1, assistant1, user2, ...]`로 유지됨. 다른 system이면 messages를 초기화하고 새로 세팅.

### 2.7 배치 테스트 스크립트 추가

**파일:** `src/test/oasis-batch-test.ts`, `package.json`

23개 응급 시나리오를 순차 실행하여 프로토콜 매칭 정확도, TTFT, 응답 내용을 한눈에 확인할 수 있는 배치 테스트 추가.

```bash
npm run test:oasis-batch
```

---

## 3. 변경된 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `src/cloud-api/local/ollama-llm.ts` | 시스템 메시지 관리 로직 개선, KV 워밍업 함수, keepAlive 프롬프트 최소화 |
| `src/cloud-api/llm.ts` | `warmupSystemPrompt` export 추가 |
| `src/core/ChatFlow.ts` | `useOasis` 기본값 정리, `knowledgePrompts` 중복 스킵 제거 |
| `src/core/OasisAdapter.ts` | `buildOasisSystemPrompt` 분리 및 export |
| `src/test/oasis-test.ts` | KV 워밍업 + 연속 질문 테스트 지원 |
| `src/test/oasis-batch-test.ts` | 23개 시나리오 배치 테스트 (신규) |
| `package.json` | `test:oasis-batch` 스크립트 추가 |
| `.env` | `OLLAMA_NUM_CTX` 주석 추가 (비활성) |

---

## 4. 성능 결과

### 단일 질문 (TTFT)

| 단계 | TTFT | 비고 |
|------|------|------|
| 최적화 전 | ~7,300ms | 기본 system 중복 + thinking 의심 |
| 시스템 메시지 중복 제거 후 | ~5,500ms | 기본 system 1개 제거 |
| KV 캐시 워밍업 적용 후 | **~1,750ms** | 워밍업으로 system prefill 제거 |

### 연속 질문 (같은 프로토콜)

| | TTFT | 비고 |
|---|------|------|
| Q1 (첫 질문) | ~6,100ms | 전체 prefill |
| Q2 (후속) | ~2,900ms | KV 캐시 히트 (53% 감소) |
| Q3 (후속) | ~2,900ms | KV 캐시 히트 유지 |

### 배치 테스트 (23개 독립 질문, 각각 워밍업 포함)

| 지표 | 값 |
|------|-----|
| 평균 TTFT | **1,752ms** |
| 최소 TTFT | 1,496ms |
| 최대 TTFT | 2,294ms |
| 프로토콜 매칭 정확도 | 20/23 정확, 3개 경계값 |

---

## 5. 남은 이슈

### 응답 정확도

| # | 질문 | 문제 |
|---|------|------|
| 7 | "hes turning blue" | 응답이 "Cough." 한 단어 — 너무 짧음 |
| 12 | "stopped shaking" (저체온) | "STOP SHIVERING" — 이미 멈춘 환자에게 잘못된 지시 |
| 20 | 버섯 먹어도 되냐 | "looks safe, eat it slowly" — **위험한 오답** (프로토콜은 "모르면 먹지 마라") |

### 매칭 점수가 낮은 케이스

| # | 질문 | 매칭 | Score |
|---|------|------|-------|
| 2 | 머리+출혈 | BLEEDING (HEAD_INJURY가 더 적절할 수 있음) | 0.660 |
| 7 | 파래지고 있음 | CHOKING_ADULT | 0.537 |
| 18 | 어.. 친구가 팔이.. | HEAD_INJURY (FRACTURE가 더 맞을 수 있음) | 0.561 |
| 20 | 숲에서 버섯 | PLANT_INGESTION | 0.663 |

### 구조적 한계

- **첫 질문 TTFT (~6초):** KV 캐시 워밍업 시간이 포함되므로 체감 총 시간은 여전히 김. 워밍업을 매칭과 병렬화하면 숨길 수 있음.
- **프로토콜 전환 시 캐시 미스:** 다른 응급 상황으로 바뀌면 다시 전체 prefill 발생.
- **하드웨어 한계:** Pi 5 CPU에서 qwen3:1.7b의 prefill 속도는 구조적으로 제한됨. 더 작은 모델이나 NPU 가속이 근본 해결책.

---

## 6. 테스트 방법

```bash
# 단일 질문
npm run test:oasis -- "my kid swallowed something and cant breathe"

# 연속 질문 (KV 캐시 효과 확인)
npm run test:oasis -- "there's so much blood" "should I remove the cloth" "he is getting pale"

# 23개 시나리오 배치 테스트
npm run test:oasis-batch
```
