# OASIS Project Context

이 프로젝트는 오프라인 응급처치 AI 디바이스(Raspberry Pi 5)의 소프트웨어.
Whisplay 챗봇 포크 기반이며, Pocket RAG 논문의 3-Stage Hybrid RAG를 구현 중.

## 핵심 파일
- src/core/ChatFlow.ts: 메인 대화 루프 (STT → RAG → LLM → TTS)
- src/core/OasisAdapter.ts: RAG 결과를 시스템 프롬프트로 변환
- python/oasis-rag/: Hybrid RAG 파이프라인 (Python Flask, 새로 구현 예정)

## 현재 작업
Pocket RAG 논문 기반 3-Stage Hybrid RAG 파이프라인 구축 중.
Phase 순서: 정리 → 문서 준비 → RAG 구현 → 인덱싱 → TS 브릿지 → 테스트

## 규칙
- 의료 조언을 직접 생성하지 말 것 — RAG로 검증된 매뉴얼만 참조
- Pi5 메모리 제한: 전체 파이프라인 4.5GB 이내
- gemma3:1b 모델 사용 (Ollama)
- 임베딩: gte-small (sentence-transformers, 384차원)
- 벡터DB: FAISS (인메모리)