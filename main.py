"""Mini NPU: 위치별 곱셈과 누적 덧셈으로 패턴을 판정한다."""

import json
import math
import re
from pathlib import Path
from time import perf_counter_ns

EPSILON = 1e-9
REPEATS = 100
SIZES = (5, 13, 25)
DATA_PATH = Path(__file__).with_name("data.json")
LABELS = {"+": "Cross", "cross": "Cross", "x": "X"}


def validate_matrix(matrix, size, name):
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError(f"{name} 행 수 불일치 (필요 {size}행)")
    for index, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{name} {index}행 열 수 불일치 (필요 {size}열)")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in row):
            raise ValueError(f"{name} {index}행에 유한한 숫자가 아닌 값이 있습니다.")


def read_matrix(name):
    matrix = []
    print(f"{name} 입력 (한 줄에 숫자 3개, 총 3줄)")
    while len(matrix) < 3:
        try:
            row = [float(value) for value in input(f"{len(matrix)+1}행 > ").split()]
            if len(row) != 3 or not all(map(math.isfinite, row)):
                raise ValueError
        except ValueError:
            print("입력 형식 오류. 각 줄에 유한한 숫자 3개를 공백으로 구분해 입력하세요.")
            continue
        matrix.append(row)
    return matrix


def mac(pattern, kernel):
    total = 0
    for pattern_row, kernel_row in zip(pattern, kernel):
        for value, weight in zip(pattern_row, kernel_row):
            total += value * weight
    return total


def classify(first, second, first_label, second_label):
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("MAC 점수가 유한한 숫자 범위를 벗어났습니다.")
    if abs(first - second) < EPSILON:
        return "UNDECIDED"
    return first_label if first > second else second_label


def normalize(label):
    if isinstance(label, str):
        standard = LABELS.get(label.strip().lower())
        if standard:
            return standard
    raise ValueError(f"알 수 없는 라벨 {label!r}")


def benchmark(samples):
    print(f"\n성능 분석 (크기별 MAC {REPEATS}회, 파일 읽기·출력 제외)")
    print("크기 | 평균 시간(ms) | MAC 연산 횟수(N²)")
    for size, pattern, kernel in samples:
        started = perf_counter_ns()
        for _ in range(REPEATS):
            mac(pattern, kernel)
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000
        print(f"{size}×{size} | {elapsed_ms / REPEATS:.6f} | {size ** 2}")


def manual_mode():
    first = read_matrix("필터 A")
    second = read_matrix("필터 B")
    print(f"필터 A, B를 각각 3×3으로 저장했습니다. A={first}, B={second}")
    pattern = read_matrix("입력 패턴")
    started = perf_counter_ns()
    score_a = mac(pattern, first)
    score_b = mac(pattern, second)
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    result = classify(score_a, score_b, "A", "B")
    print(f"A={score_a:.17g}, B={score_b:.17g}, MAC 2회 시간={elapsed_ms:.6f} ms")
    detail = f" (판정 불가, |A-B| < {EPSILON:g})" if result == "UNDECIDED" else ""
    print(f"판정 {result}{detail}")
    benchmark([(3, pattern, first)])


def require_object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name}은 JSON 객체여야 합니다.")
    return value


def case_result(key, case, filters):
    match = re.fullmatch(r"size_(5|13|25)_\d+", key)
    if match is None:
        raise ValueError("케이스 키는 size_{N}_{idx} 형식이어야 합니다.")
    size = int(match[1])
    case = require_object(case, key)
    group_key = f"size_{size}"
    group = require_object(filters.get(group_key), group_key)
    kernels = {}
    for label, matrix in group.items():
        standard = normalize(label)
        if standard in kernels:
            raise ValueError(f"{group_key}에 {standard} 필터가 중복됩니다.")
        validate_matrix(matrix, size, f"{group_key} {standard} 필터")
        kernels[standard] = matrix
    if set(kernels) != {"Cross", "X"}:
        raise ValueError(f"{group_key}에 Cross와 X 필터가 필요합니다.")
    pattern = case.get("input")
    validate_matrix(pattern, size, "입력 패턴")
    expected = normalize(case.get("expected"))
    cross_score = mac(pattern, kernels["Cross"])
    x_score = mac(pattern, kernels["X"])
    result = classify(cross_score, x_score, "Cross", "X")
    return cross_score, x_score, result, expected


def json_mode(path):
    data = require_object(json.loads(path.read_text(encoding="utf-8-sig")), "최상위 값")
    patterns = require_object(data.get("patterns"), "patterns")
    filters = data.get("filters")
    failures = []
    print(f"데이터 파일 {path.name}")
    for key, case in patterns.items():
        try:
            cross, x, result, expected = case_result(
                key, case, require_object(filters, "filters")
            )
            status = "PASS" if result == expected else "FAIL"
            if result == "UNDECIDED":
                status += " (동점 규칙)"
            print(f"{key} | Cross={cross:.17g} | X={x:.17g} | "
                  f"판정={result} | expected={expected} | {status}")
            if result != expected:
                reason = f"판정 {result}, 기대 {expected}"
                if result == "UNDECIDED":
                    reason += f" (동점 규칙, |Cross-X| < {EPSILON:g})"
                failures.append((key, reason))
        except (ValueError, OverflowError) as error:
            failures.append((key, str(error)))
            print(f"{key} | FAIL | {error}")

    # 고정 입력을 사용해 케이스가 잘못되어도 같은 조건으로 비교한다.
    samples = []
    for size in (3,) + SIZES:
        matrix = [[1.0] * size for _ in range(size)]
        samples.append((size, matrix, matrix))
    benchmark(samples)

    print(f"\n전체 {len(patterns)}건 | 통과 {len(patterns)-len(failures)}건 | 실패 {len(failures)}건")
    for key, reason in failures:
        print(f"- {key} — {reason}")


def main():
    print("1. 사용자 입력(3×3)\n2. data.json 분석")
    try:
        while True:
            mode = input("모드 선택 > ").strip()
            if mode == "1":
                manual_mode()
                return
            if mode == "2":
                json_mode(DATA_PATH)
                return
            print("1 또는 2를 입력하세요.")
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되었습니다.")
    except (OSError, ValueError, OverflowError) as error:
        print(f"실행 오류 — {error}")


if __name__ == "__main__":
    main()
