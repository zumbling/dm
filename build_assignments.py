"""번호별 담당자(assignments.json)를 만든다. 로컬에서 한 번 실행해 결과를 레포에 올린다.

규칙: 파손/정상을 각각 4명이 똑같이 나눈다. 지금까지 완료된 번호는 모두 LEAD 가 한 것이므로 LEAD 몫에 먼저 넣고,
LEAD 는 나머지 사람들과 총량이 같으니 (나머지를 먼저 올림) 남은 개수가 가장 적다. 나머지 사람은 번호 구간을 이어서 맡는다.
실행:  python build_assignments.py     (환경변수 LABELER_DATA: 완료 여부를 볼 로컬 data/img 폴더)
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LABELER_DATA", os.path.join(os.path.dirname(HERE), "2_dataset", "data", "img"))
PEOPLE = ["종빈", "수임", "윤진", "태희"]
LEAD = "종빈"
REQUIRED = ["paving_mask.png"]
ONE_OF = ["straight_mask.png", "stop_mask.png"]   # 둘 중 하나만 있으면 됨 (그 이미지에 없을 수도 있어서)


def main():
    done = {int(n) for n in os.listdir(DATA) if n.isdigit()
            and all(os.path.exists(os.path.join(DATA, n, "fix", f)) for f in REQUIRED)
            and any(os.path.exists(os.path.join(DATA, n, "fix", f)) for f in ONE_OF)}
    with open(os.path.join(HERE, "overview.json"), encoding="utf-8") as f:
        ov = json.load(f)
    assign = {}
    others = [p for p in PEOPLE if p != LEAD]
    for kind in ("파손", "정상"):
        nums = sorted(int(n) for n, v in ov.items() if v["kind"] == kind)
        base, rem = divmod(len(nums), len(PEOPLE))
        quota = {LEAD: base, **{p: base + (1 if i < rem else 0) for i, p in enumerate(others)}}  # 나머지 개수는 LEAD 가 아닌 쪽이 맡는다
        mine = [n for n in nums if n in done]
        rest = [n for n in nums if n not in done]
        take = max(quota[LEAD] - len(mine), 0)
        mine += rest[:take]
        rest = rest[take:]
        for n in mine:
            assign[str(n)] = LEAD
        for p in others:
            for n in rest[:quota[p]]:
                assign[str(n)] = p
            rest = rest[quota[p]:]
    with open(os.path.join(HERE, "assignments.json"), "w", encoding="utf-8") as f:
        json.dump(assign, f, ensure_ascii=False, separators=(",", ":"))
    print("완료(전부", LEAD, ")", len(done))
    for p in PEOPLE:
        c = Counter(ov[n]["kind"] for n, w in assign.items() if w == p)
        d = sum(1 for n, w in assign.items() if w == p and int(n) in done)
        print(f"{p}: 총 {sum(c.values())} (파손 {c['파손']}, 정상 {c['정상']}) · 완료 {d} · 남음 {sum(c.values()) - d}")


if __name__ == "__main__":
    main()
