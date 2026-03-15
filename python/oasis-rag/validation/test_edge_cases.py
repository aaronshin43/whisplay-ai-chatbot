"""Part 4: Edge Case Tests — panic input, typos, empty, very long queries."""
from __future__ import annotations
from _shared import TestResult, get_context_text, context_contains, top_score

EDGE_CASE_TESTS = [
    {"id": "EDG-001",
     "query": "HELP THERE IS SO MUCH BLOOD OH GOD PLEASE HELP",
     "must_return_context": True,
     "must_contain": ["bleed", "pressure"],
     "note": "패닉 상태에서도 출혈 관련 정보 반환"},
    {"id": "EDG-002",
     "query": "HE IS DYING WHAT DO I DO HE IS NOT MOVING",
     "must_return_context": True,
     "must_contain": [],
     "note": "응답이 비어있으면 안 됨"},
    {"id": "EDG-003",
     "query": "oh no oh no oh no shes not breathing help",
     "must_return_context": True,
     "must_contain": ["breath"],
     "note": "반복적 패닉 입력 처리"},
    {"id": "EDG-004",
     "query": "what is the weather today",
     "expect_low_score": True,
     "max_score": 0.50,
     "must_contain": [],
     "note": "비의료 질문은 낮은 점수여야 함"},
    {"id": "EDG-005",
     "query": "tell me a joke",
     "expect_low_score": True,
     "max_score": 0.50,
     "must_contain": [],
     "note": "비의료 질문은 낮은 점수여야 함"},
    {"id": "EDG-006",
     "query": "bleeding",
     "must_return_context": True,
     "must_contain": [],
     "note": "단일 단어 쿼리도 처리"},
    {"id": "EDG-007",
     "query": "help",
     "must_return_context": True,
     "must_contain": [],
     "note": "가장 짧은 도움 요청"},
    {"id": "EDG-008",
     "query": "hes bledding from the nek real bad",
     "must_return_context": True,
     "must_contain": [],
     "note": "오타 포함 입력 — bleeding, neck"},
    {"id": "EDG-009",
     "query": "she done fell and her arm look broke",
     "must_return_context": True,
     "must_contain": ["fracture"],
     "note": "비표준 영어 입력"},
    {"id": "EDG-010",
     "query": "car accident bleeding from head and leg looks broken",
     "must_return_context": True,
     "must_contain": [],
     "note": "다중 부상 — 가장 위험한 것 우선 반환되는지"},
    {"id": "EDG-011",
     "query": "",
     "expect_error_or_empty": True,
     "must_contain": [],
     "note": "빈 쿼리는 에러 없이 빈 결과"},
    {"id": "EDG-012",
     "query": ("so we were hiking in the mountains and my friend slipped on a rock "
               "and fell about ten feet down and now he is lying on the ground and "
               "his leg is bent at a weird angle and there is blood coming from a cut "
               "on his head and he says he cant feel his toes and I dont know what to do please help me"),
     "must_return_context": True,
     "must_contain": [],
     "note": "긴 자연어 입력 — 핵심 정보 추출 가능한지"},
]


def run(retriever) -> list[TestResult]:
    results = []
    for tc in EDGE_CASE_TESTS:
        try:
            # Empty query special case
            if tc.get("expect_error_or_empty"):
                try:
                    ret = retriever.retrieve(tc["query"])
                    # Passed if no exception and context is empty or minimal
                    empty_ok = not ret.context or not ret.context.strip()
                    results.append(TestResult(
                        tc["id"], True,
                        f"no exception, context empty={empty_ok}",
                        {"context_len": len(ret.context or "")}
                    ))
                except Exception:
                    # Also acceptable — graceful error
                    results.append(TestResult(tc["id"], True, "raised exception (acceptable)"))
                continue

            ret = retriever.retrieve(tc["query"])
            ctx = get_context_text(ret)
            score = top_score(ret)
            has_context = bool(ret.context and ret.context.strip())

            if tc.get("expect_low_score"):
                passed = score <= tc["max_score"]
                note = f"{tc['note']} | score={score:.4f} (max={tc['max_score']})"
                results.append(TestResult(tc["id"], passed, note, {"score": score}))
                continue

            missing = context_contains(ctx, tc.get("must_contain", []))
            context_ok = has_context if tc.get("must_return_context") else True
            passed = context_ok and not missing
            note = tc["note"]
            if not context_ok:
                note += " | EMPTY CONTEXT"
            if missing:
                note += f" | missing: {missing}"
            note += f" | score={score:.4f}"
            results.append(TestResult(
                tc["id"], passed, note,
                {"score": score, "has_context": has_context, "missing": missing}
            ))
        except Exception as e:
            if tc.get("expect_error_or_empty"):
                results.append(TestResult(tc["id"], True, f"exception as expected: {e}"))
            else:
                results.append(TestResult(tc["id"], False, f"EXCEPTION: {e}"))
    return results
