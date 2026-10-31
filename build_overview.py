"""전체보기 탭용 메타데이터(overview.json)를 만든다. 로컬에서 한 번 실행해 결과를 레포에 올린다.

번호마다: AI Hub 원본 파일명, 원본이 속한 폴더(교체폐기 = 파손 데이터 / 정상), 시설물 상태(status), 파손 박스 수·종류.
실행:  python build_overview.py   (환경변수 LABELER_MAP / LABELER_JSON_ROOT 로 경로를 바꿀 수 있다)
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RENAME_MAP = os.environ.get("LABELER_MAP", os.path.join(
    ROOT, "extracted_download", "113.노후_시설물_이미지_데이터", "01.데이터", "1.Training", "원천데이터", "rename_map.csv"))
JSON_ROOT = os.environ.get("LABELER_JSON_ROOT", os.path.join(ROOT, "trash"))
KINDS = {"교체폐기": "파손", "정상": "정상"}   # 폴더 이름에 들어 있는 말 -> 구분


def main():
    where = {}   # 원본 파일명 stem -> (json 경로, 구분)
    for d in sorted(os.listdir(JSON_ROOT)):
        kind = next((v for k, v in KINDS.items() if k in d), None)
        full = os.path.join(JSON_ROOT, d)
        if kind and os.path.isdir(full):
            for f in os.listdir(full):
                if f.endswith(".json"):
                    where[f[:-5]] = (os.path.join(full, f), kind)
    out = {}
    with open(RENAME_MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            num, orig = os.path.splitext(r["new"])[0], os.path.splitext(r["original"])[0]
            hit = where.get(orig)
            if not hit:
                out[num] = {"orig": orig, "kind": "미확인"}
                continue
            with open(hit[0], encoding="utf-8") as jf:
                j = json.load(jf)
            ann = j.get("annotations", [])
            status = next((a["attributes"].get("status") for a in ann if a["category_id"] == 1), None)
            dmg = [a["attributes"].get("damageType", "?") for a in ann if a["category_id"] == 2]
            out[num] = {"orig": orig, "kind": hit[1], "status": status, "damage": dmg}
    with open(os.path.join(HERE, "overview.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    from collections import Counter
    print(len(out), "개", dict(Counter(v["kind"] for v in out.values())),
          "파손박스 있는 번호", sum(1 for v in out.values() if v.get("damage")))


if __name__ == "__main__":
    main()
