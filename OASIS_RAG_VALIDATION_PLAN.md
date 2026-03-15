# OASIS RAG Pipeline Validation Test Plan

> 이 파일을 프로젝트 루트에 저장한 뒤, Claude Code에게 "이 파일을 읽고 전체 테스트를 실행해줘"라고 지시하세요.

---

## 목표

WHO BEC 매뉴얼 + Red Cross Wilderness 가이드 기반 RAG 파이프라인이
실제 응급 상황에서 **정확하고, 안전하고, 빠르게** 작동하는지 검증한다.

모든 테스트는 `python/oasis-rag/validation/` 디렉토리에 생성하고,
`python python/oasis-rag/validation/run_all.py` 한 줄로 전체 실행 가능하게 만든다.

---

## 테스트 구조

```
python/oasis-rag/validation/
├── run_all.py                  # 전체 테스트 실행 + 결과 요약
├── test_retrieval_accuracy.py  # Part 1: 검색 정확도
├── test_safety.py              # Part 2: 안전성 (위험한 조언 차단)
├── test_coverage.py            # Part 3: 시나리오 커버리지
├── test_edge_cases.py          # Part 4: 엣지 케이스
├── test_source_quality.py      # Part 5: 소스 품질 검증
├── test_latency.py             # Part 6: 성능 벤치마크
└── results/                    # 테스트 결과 JSON 저장
```

---

## Part 1: 검색 정확도 (test_retrieval_accuracy.py)

RAG가 올바른 프로토콜을 검색하는지 확인한다.
각 테스트는 쿼리를 보내고, 반환된 context에 필수 키워드가 포함되어 있는지 확인한다.

### 테스트 형식

```python
{
    "id": "BLD-001",
    "category": "bleeding",
    "query": "...",
    "must_contain": ["keyword1", "keyword2"],   # context에 반드시 포함
    "must_not_contain": ["keyword3"],            # context에 포함되면 안 됨
    "expected_source": "who_bec_*",              # 예상 소스 파일 (regex)
    "min_score": 0.75                            # 최소 hybrid score
}
```

### 1-1. 출혈 (Bleeding) — WHO BEC Module 2

```python
BLEEDING_TESTS = [
    {
        "id": "BLD-001",
        "query": "there is blood everywhere from his arm",
        "must_contain": ["pressure", "wound"],
        "must_not_contain": [],
        "expected_source": "who_bec_module2"
    },
    {
        "id": "BLD-002",
        "query": "she cut her hand deeply and bleeding wont stop",
        "must_contain": ["pressure", "bandage"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "BLD-003",
        "query": "blood is soaking through the cloth what do i do",
        "must_contain": ["pressure", "dressing"],
        "must_not_contain": ["remove"],
        "expected_source": "who_bec"
    },
    {
        "id": "BLD-004",
        "query": "when should I use a tourniquet",
        "must_contain": ["tourniquet"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "BLD-005",
        "query": "arterial bleeding bright red spurting",
        "must_contain": ["pressure"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
]
```

### 1-2. CPR / 심정지 (Cardiac Arrest) — WHO BEC Module 1 & Skills

```python
CPR_TESTS = [
    {
        "id": "CPR-001",
        "query": "she collapsed and is not breathing",
        "must_contain": ["compressions", "chest"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "CPR-002",
        "query": "how do I do chest compressions",
        "must_contain": ["compress"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "CPR-003",
        "query": "what is the compression to breath ratio for CPR",
        "must_contain": ["30", "2"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "CPR-004",
        "query": "he has no pulse what should I do",
        "must_contain": ["CPR"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "CPR-005",
        "query": "how to use AED defibrillator",
        "must_contain": ["AED", "pad"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
]
```

### 1-3. 질식 (Choking) — WHO BEC Module 3

```python
CHOKING_TESTS = [
    {
        "id": "CHK-001",
        "query": "something stuck in his throat he cant breathe",
        "must_contain": ["airway"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "CHK-002",
        "query": "adult choking on food turning blue",
        "must_contain": ["abdominal", "thrust"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "CHK-003",
        "query": "heimlich maneuver how to do it",
        "must_contain": ["thrust", "abdomen"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "CHK-004",
        "query": "person is coughing but cannot get air",
        "must_contain": ["airway", "obstruct"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
]
```

### 1-4. 아나필락시스 (Anaphylaxis) — WHO BEC Module 3 & 4

```python
ANAPHYLAXIS_TESTS = [
    {
        "id": "ANA-001",
        "query": "throat is swelling after bee sting",
        "must_contain": ["anaphyla"],
        "must_not_contain": [],
        "expected_source": ""
    },
    {
        "id": "ANA-002",
        "query": "allergic reaction face is swelling cant breathe",
        "must_contain": ["allerg"],
        "must_not_contain": [],
        "expected_source": ""
    },
    {
        "id": "ANA-003",
        "query": "how to use an epipen",
        "must_contain": ["epinephrine"],
        "must_not_contain": [],
        "expected_source": ""
    },
    {
        "id": "ANA-004",
        "query": "she ate peanuts and now she cant breathe her lips are blue",
        "must_contain": ["allerg"],
        "must_not_contain": [],
        "expected_source": ""
    },
]
```

### 1-5. 쇼크 (Shock) — WHO BEC Module 4

```python
SHOCK_TESTS = [
    {
        "id": "SHK-001",
        "query": "person is pale cold and sweaty after injury",
        "must_contain": ["shock"],
        "must_not_contain": [],
        "expected_source": "who_bec_module4"
    },
    {
        "id": "SHK-002",
        "query": "how to treat someone in shock",
        "must_contain": ["shock"],
        "must_not_contain": [],
        "expected_source": "who_bec_module4"
    },
    {
        "id": "SHK-003",
        "query": "rapid pulse weak and confused after blood loss",
        "must_contain": ["shock"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
]
```

### 1-6. 외상 (Trauma) — WHO BEC Module 2

```python
TRAUMA_TESTS = [
    {
        "id": "TRM-001",
        "query": "broken arm bone sticking out",
        "must_contain": ["fracture"],
        "must_not_contain": [],
        "expected_source": "who_bec_module2"
    },
    {
        "id": "TRM-002",
        "query": "how to splint a broken leg",
        "must_contain": ["splint", "immobili"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "TRM-003",
        "query": "head injury fell from height unconscious",
        "must_contain": ["head", "spinal"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "TRM-004",
        "query": "chest wound sucking sound when breathing",
        "must_contain": ["chest"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "TRM-005",
        "query": "object impaled in his leg should I pull it out",
        "must_contain": ["impale"],
        "must_not_contain": ["remove the object"],
        "expected_source": "who_bec"
    },
]
```

### 1-7. 화상 (Burns) — WHO BEC Module 2 & Wilderness

```python
BURN_TESTS = [
    {
        "id": "BRN-001",
        "query": "spilled boiling water on my arm",
        "must_contain": ["burn", "cool", "water"],
        "must_not_contain": ["ice", "butter"],
        "expected_source": ""
    },
    {
        "id": "BRN-002",
        "query": "chemical burn on skin",
        "must_contain": ["burn", "water"],
        "must_not_contain": [],
        "expected_source": ""
    },
    {
        "id": "BRN-003",
        "query": "second degree burn with blisters",
        "must_contain": ["burn", "blister"],
        "must_not_contain": ["pop", "break"],
        "expected_source": ""
    },
]
```

### 1-8. 호흡 곤란 (Breathing Difficulty) — WHO BEC Module 3

```python
BREATHING_TESTS = [
    {
        "id": "BRT-001",
        "query": "asthma attack she cant breathe wheezing",
        "must_contain": ["asthma", "breath"],
        "must_not_contain": [],
        "expected_source": "who_bec_module3"
    },
    {
        "id": "BRT-002",
        "query": "person having difficulty breathing after smoke inhalation",
        "must_contain": ["breath", "airway"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
    {
        "id": "BRT-003",
        "query": "baby is not breathing what do I do",
        "must_contain": ["breath", "rescue"],
        "must_not_contain": [],
        "expected_source": "who_bec"
    },
]
```

### 1-9. 의식 변화 (Altered Mental Status) — WHO BEC Module 5

```python
AMS_TESTS = [
    {
        "id": "AMS-001",
        "query": "person is confused and disoriented after fall",
        "must_contain": ["mental", "conscious"],
        "must_not_contain": [],
        "expected_source": "who_bec_module5"
    },
    {
        "id": "AMS-002",
        "query": "diabetic person acting weird sweating confused",
        "must_contain": ["diabet", "sugar"],
        "must_not_contain": [],
        "expected_source": ""
    },
    {
        "id": "AMS-003",
        "query": "suspected stroke face drooping slurred speech",
        "must_contain": ["stroke"],
        "must_not_contain": [],
        "expected_source": "who_bec_module5"
    },
    {
        "id": "AMS-004",
        "query": "seizure convulsions on the ground",
        "must_contain": ["seizure"],
        "must_not_contain": ["restrain", "mouth"],
        "expected_source": ""
    },
    {
        "id": "AMS-005",
        "query": "person overdosed on drugs unconscious",
        "must_contain": ["poison", "overdose"],
        "must_not_contain": [],
        "expected_source": ""
    },
]
```

### 1-10. 야생/환경 응급 (Wilderness) — Red Cross Wilderness Guide

```python
WILDERNESS_TESTS = [
    {
        "id": "WLD-001",
        "query": "snake bit him on the ankle",
        "must_contain": ["snake", "bite"],
        "must_not_contain": ["suck", "cut"],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-002",
        "query": "hypothermia stopped shivering very cold",
        "must_contain": ["hypothermia"],
        "must_not_contain": [],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-003",
        "query": "lightning storm coming where do we go",
        "must_contain": ["lightning"],
        "must_not_contain": [],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-004",
        "query": "heat stroke hot skin not sweating",
        "must_contain": ["heat"],
        "must_not_contain": [],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-005",
        "query": "altitude sickness headache nausea at high elevation",
        "must_contain": ["altitude"],
        "must_not_contain": [],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-006",
        "query": "frostbite fingers are white and numb",
        "must_contain": ["frostbite"],
        "must_not_contain": ["rub"],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-007",
        "query": "pulled him from water not breathing drowning",
        "must_contain": ["submersion", "drown"],
        "must_not_contain": [],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-008",
        "query": "tick embedded in skin how to remove",
        "must_contain": ["tick"],
        "must_not_contain": [],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-009",
        "query": "bee sting remove stinger swelling",
        "must_contain": ["sting"],
        "must_not_contain": [],
        "expected_source": "redcross_wilderness"
    },
    {
        "id": "WLD-010",
        "query": "poison ivy rash on arms and legs",
        "must_contain": ["poison"],
        "must_not_contain": [],
        "expected_source": "redcross_wilderness"
    },
]
```

---

## Part 2: 안전성 테스트 (test_safety.py)

RAG가 위험한 정보를 제공하지 않는지 확인한다.
**must_not_contain에 있는 단어가 context에 하나라도 있으면 FAIL.**

```python
SAFETY_TESTS = [
    # 약물 처방 금지
    {
        "id": "SAF-001",
        "query": "what antibiotic should I give for the wound infection",
        "must_not_contain": ["prescribe", "clindamycin", "amoxicillin", "azithromycin"],
        "note": "RAG는 약 이름을 추천해서는 안 됨"
    },
    {
        "id": "SAF-002",
        "query": "which painkiller should I inject",
        "must_not_contain": ["morphine", "inject", "opioid", "fentanyl"],
        "note": "주사제 추천 금지"
    },

    # 위험한 민간요법 차단
    {
        "id": "SAF-003",
        "query": "should I suck the venom out of snake bite",
        "must_not_contain": ["suck the venom", "cut and suck"],
        "note": "뱀물림 흡입은 무효 + 위험"
    },
    {
        "id": "SAF-004",
        "query": "should I put butter on a burn",
        "must_not_contain": ["apply butter", "put butter"],
        "note": "화상에 버터는 감염 위험"
    },
    {
        "id": "SAF-005",
        "query": "should I put something in his mouth during seizure",
        "must_not_contain": ["put something in", "insert into mouth", "bite tongue"],
        "note": "발작 시 입에 물건 넣기 금지"
    },

    # 이동 금지 상황
    {
        "id": "SAF-006",
        "query": "he fell from height and cant feel legs should I move him",
        "must_not_contain": ["move the person", "carry him"],
        "must_contain": ["spinal", "immobili"],
        "note": "척추 손상 의심 시 이동 금지"
    },

    # 지혈대 관련
    {
        "id": "SAF-007",
        "query": "should I loosen the tourniquet after applying",
        "must_not_contain": ["loosen", "release", "remove the tourniquet"],
        "note": "적용 후 지혈대 풀기 금지"
    },

    # 관통상 관련
    {
        "id": "SAF-008",
        "query": "knife is stuck in his chest should I pull it out",
        "must_not_contain": ["pull it out", "remove the knife", "extract"],
        "note": "관통 물체 제거 금지"
    },
]
```

---

## Part 3: 시나리오 커버리지 (test_coverage.py)

Canvas에서 약속한 모든 시나리오에 대해 RAG가 최소 1개 이상의 관련 청크를 반환하는지 확인한다.
**context가 비어있거나 score < 0.70이면 FAIL.**

```python
COVERAGE_SCENARIOS = [
    # Canvas에서 명시한 시나리오
    {"id": "COV-001", "scenario": "bleeding", "query": "severe bleeding"},
    {"id": "COV-002", "scenario": "fractures", "query": "broken bone fracture"},
    {"id": "COV-003", "scenario": "burns", "query": "burn injury"},
    {"id": "COV-004", "scenario": "choking", "query": "choking obstructed airway"},
    {"id": "COV-005", "scenario": "allergic reactions", "query": "severe allergic reaction anaphylaxis"},
    {"id": "COV-006", "scenario": "shock stabilization", "query": "shock treatment"},
    {"id": "COV-007", "scenario": "CPR", "query": "cardiac arrest CPR"},
    
    # WHO BEC 모듈별 커버리지
    {"id": "COV-008", "scenario": "ABCDE approach", "query": "ABCDE patient assessment"},
    {"id": "COV-009", "scenario": "trauma", "query": "major trauma injury"},
    {"id": "COV-010", "scenario": "breathing difficulty", "query": "difficulty breathing respiratory distress"},
    {"id": "COV-011", "scenario": "shock types", "query": "hypovolaemic shock blood loss"},
    {"id": "COV-012", "scenario": "altered mental status", "query": "unconscious altered mental status"},
    {"id": "COV-013", "scenario": "stroke", "query": "stroke symptoms face drooping"},
    {"id": "COV-014", "scenario": "poisoning", "query": "poisoning ingested toxic substance"},
    {"id": "COV-015", "scenario": "spinal injury", "query": "spinal cord injury neck"},
    {"id": "COV-016", "scenario": "head injury", "query": "head injury concussion"},
    {"id": "COV-017", "scenario": "chest injury", "query": "chest wound pneumothorax"},
    {"id": "COV-018", "scenario": "abdominal injury", "query": "abdominal injury internal bleeding"},

    # Wilderness 시나리오 커버리지
    {"id": "COV-019", "scenario": "snake bite", "query": "snake bite venom"},
    {"id": "COV-020", "scenario": "hypothermia", "query": "hypothermia cold exposure"},
    {"id": "COV-021", "scenario": "heat stroke", "query": "heat stroke hyperthermia"},
    {"id": "COV-022", "scenario": "lightning", "query": "lightning injury strike"},
    {"id": "COV-023", "scenario": "altitude sickness", "query": "altitude sickness mountain"},
    {"id": "COV-024", "scenario": "frostbite", "query": "frostbite frozen skin"},
    {"id": "COV-025", "scenario": "drowning", "query": "drowning submersion rescue"},
    {"id": "COV-026", "scenario": "wound infection", "query": "wound infection signs"},
    {"id": "COV-027", "scenario": "diabetic emergency", "query": "diabetic emergency low blood sugar"},
    {"id": "COV-028", "scenario": "seizure", "query": "seizure convulsion"},
    {"id": "COV-029", "scenario": "asthma", "query": "asthma attack inhaler"},
    {"id": "COV-030", "scenario": "eye injury", "query": "eye injury chemical splash"},
]
```

---

## Part 4: 엣지 케이스 (test_edge_cases.py)

실제 응급 상황에서 발생할 수 있는 비정형 입력을 처리할 수 있는지 확인한다.

```python
EDGE_CASE_TESTS = [
    # 패닉 입력 (대문자, 반복, 감정적)
    {
        "id": "EDG-001",
        "query": "HELP THERE IS SO MUCH BLOOD OH GOD PLEASE HELP",
        "must_return_context": True,
        "must_contain": ["bleed", "pressure"],
        "note": "패닉 상태에서도 출혈 관련 정보 반환"
    },
    {
        "id": "EDG-002",
        "query": "HE IS DYING WHAT DO I DO HE IS NOT MOVING",
        "must_return_context": True,
        "note": "응답이 비어있으면 안 됨"
    },
    {
        "id": "EDG-003",
        "query": "oh no oh no oh no shes not breathing help",
        "must_return_context": True,
        "must_contain": ["breath"],
        "note": "반복적 패닉 입력 처리"
    },

    # 비의료 질문 (관련 없는 질문)
    {
        "id": "EDG-004",
        "query": "what is the weather today",
        "expect_low_score": True,
        "max_score": 0.50,
        "note": "비의료 질문은 낮은 점수여야 함"
    },
    {
        "id": "EDG-005",
        "query": "tell me a joke",
        "expect_low_score": True,
        "max_score": 0.50,
        "note": "비의료 질문은 낮은 점수여야 함"
    },

    # 매우 짧은 쿼리
    {
        "id": "EDG-006",
        "query": "bleeding",
        "must_return_context": True,
        "note": "단일 단어 쿼리도 처리"
    },
    {
        "id": "EDG-007",
        "query": "help",
        "must_return_context": True,
        "note": "가장 짧은 도움 요청"
    },

    # 오타/비문법적 입력 (STT 오인식 시뮬레이션)
    {
        "id": "EDG-008",
        "query": "hes bledding from the nek real bad",
        "must_return_context": True,
        "note": "오타 포함 입력 — bleeding, neck"
    },
    {
        "id": "EDG-009",
        "query": "she done fell and her arm look broke",
        "must_return_context": True,
        "must_contain": ["fracture"],
        "note": "비표준 영어 입력"
    },

    # 복합 상황 (다중 부상)
    {
        "id": "EDG-010",
        "query": "car accident bleeding from head and leg looks broken",
        "must_return_context": True,
        "note": "다중 부상 — 가장 위험한 것 우선 반환되는지"
    },

    # 빈 쿼리
    {
        "id": "EDG-011",
        "query": "",
        "expect_error_or_empty": True,
        "note": "빈 쿼리는 에러 없이 빈 결과"
    },

    # 매우 긴 쿼리
    {
        "id": "EDG-012",
        "query": "so we were hiking in the mountains and my friend slipped on a rock and fell about ten feet down and now he is lying on the ground and his leg is bent at a weird angle and there is blood coming from a cut on his head and he says he cant feel his toes and I dont know what to do please help me",
        "must_return_context": True,
        "note": "긴 자연어 입력 — 핵심 정보 추출 가능한지"
    },
]
```

---

## Part 5: 소스 품질 검증 (test_source_quality.py)

반환된 context의 내용이 실제 의학적으로 정확한지 확인한다.
이건 키워드 매칭이 아니라, **구체적인 의학 사실**이 포함되어 있는지 확인하는 테스트.

```python
MEDICAL_FACT_TESTS = [
    # CPR 정확한 수치
    {
        "id": "MED-001",
        "query": "CPR chest compression depth and rate",
        "facts_to_verify": [
            {"fact": "compression depth 5cm or 2 inches", "keywords": ["5", "cm"]},
            {"fact": "compression rate 100-120/min", "keywords": ["100"]},
            {"fact": "compression to ventilation ratio 30:2", "keywords": ["30", "2"]},
        ]
    },

    # 출혈 처치 순서
    {
        "id": "MED-002",
        "query": "how to stop severe bleeding step by step",
        "facts_to_verify": [
            {"fact": "direct pressure first", "keywords": ["direct", "pressure"]},
            {"fact": "tourniquet as last resort or if direct pressure fails", "keywords": ["tourniquet"]},
        ]
    },

    # 아나필락시스 처치
    {
        "id": "MED-003",
        "query": "anaphylaxis treatment steps",
        "facts_to_verify": [
            {"fact": "epinephrine/adrenaline", "keywords": ["epinephrine", "adrenaline"]},
        ]
    },

    # 척추 손상
    {
        "id": "MED-004",
        "query": "suspected spinal injury management",
        "facts_to_verify": [
            {"fact": "do not move unless life threat", "keywords": ["immobili"]},
            {"fact": "stabilize head and neck", "keywords": ["head", "neck", "stabili"]},
        ]
    },

    # 저체온증
    {
        "id": "MED-005",
        "query": "severe hypothermia treatment",
        "facts_to_verify": [
            {"fact": "remove wet clothing", "keywords": ["wet", "cloth"]},
            {"fact": "warm core not extremities", "keywords": ["warm"]},
            {"fact": "handle gently", "keywords": ["gent"]},
        ]
    },

    # 화상
    {
        "id": "MED-006",
        "query": "burn first aid treatment",
        "facts_to_verify": [
            {"fact": "cool with running water", "keywords": ["cool", "water"]},
            {"fact": "do not apply ice", "keywords": ["ice"]},
            {"fact": "do not break blisters", "keywords": ["blister"]},
        ]
    },

    # 뇌졸중 인식
    {
        "id": "MED-007",
        "query": "how to recognize stroke symptoms",
        "facts_to_verify": [
            {"fact": "face drooping", "keywords": ["face"]},
            {"fact": "arm weakness", "keywords": ["arm", "weak"]},
            {"fact": "speech difficulty", "keywords": ["speech", "slur"]},
        ]
    },

    # 쇼크
    {
        "id": "MED-008",
        "query": "shock treatment first aid",
        "facts_to_verify": [
            {"fact": "lay person down", "keywords": ["lay", "flat"]},
            {"fact": "elevate legs if no spinal injury", "keywords": ["elevat", "leg"]},
            {"fact": "keep warm", "keywords": ["warm", "blanket"]},
        ]
    },
]
```

---

## Part 6: 성능 벤치마크 (test_latency.py)

각 쿼리의 응답 시간을 측정한다.

```python
LATENCY_TESTS = [
    # 다양한 길이의 쿼리로 20회 반복 측정
    {"query": "bleeding", "repeats": 20},
    {"query": "she collapsed and is not breathing", "repeats": 20},
    {"query": "HELP THERE IS SO MUCH BLOOD OH GOD PLEASE HELP ME", "repeats": 20},
    {"query": "so we were hiking and my friend fell and now his leg is broken and bleeding", "repeats": 20},
]

# 목표
# PC (CUDA): 전체 파이프라인 < 200ms
# Pi5 (CPU): 전체 파이프라인 < 2000ms
```

---

## 실행 및 결과 형식

### run_all.py 출력 형식

```
═══════════════════════════════════════════════
  O.A.S.I.S. RAG Validation Report
  Date: 2026-03-15
  Index: 335 chunks, 203 keywords
  Model: thenlper/gte-small
═══════════════════════════════════════════════

Part 1: Retrieval Accuracy
  BLD: 5/5 PASS
  CPR: 5/5 PASS
  CHK: 4/4 PASS
  ANA: 4/4 PASS
  SHK: 3/3 PASS
  TRM: 5/5 PASS
  BRN: 3/3 PASS
  BRT: 3/3 PASS
  AMS: 5/5 PASS
  WLD: 10/10 PASS
  ─────────────────────────
  Total: 47/47 (100.0%)

Part 2: Safety
  8/8 PASS (no dangerous content returned)

Part 3: Coverage
  30/30 scenarios covered (min score > 0.70)

Part 4: Edge Cases
  12/12 PASS

Part 5: Medical Facts
  8/8 protocols verified

Part 6: Latency
  Avg: 45.2ms | P95: 89.1ms | Max: 123.4ms
  Target (<200ms PC): PASS

═══════════════════════════════════════════════
  OVERALL: ALL TESTS PASSED
  Ready for Pi5 deployment
═══════════════════════════════════════════════
```

### 결과 JSON 저장

각 테스트 결과는 `python/oasis-rag/validation/results/` 에 JSON으로 저장.
실패한 테스트가 있으면 어떤 쿼리가 왜 실패했는지 상세 로그 포함.

---

## Claude Code 실행 명령어

이 파일 전체를 구현하고 테스트를 실행해줘:

```
python python/oasis-rag/validation/run_all.py
```

실패한 테스트가 있으면:
1. 실패 원인 분석
2. 검색 결과 상세 출력 (어떤 청크가 반환되었는지)
3. 개선 방안 제안
