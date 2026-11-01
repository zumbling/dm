"""Drive 에서는 완료인데 로컬(2_dataset/data/img)에서는 미완료인 내 담당 번호의 라벨을 로컬로 받아 저장한다.

완료 기준: paving_mask.png + (straight_mask.png 또는 stop_mask.png)   — build_assignments.py 와 같다.
받는 파일: fix/ 안의 *_mask.png, points.json (original.jpeg 는 로컬 것을 그대로 둔다).

실행 (Drive 모드 환경변수 중 하나가 필요):
  LABELER_DRIVE_ID=<폴더ID> python pull_done.py --dry-run         # OAuth (token.json)
  LABELER_GAS_URL=<…/exec> LABELER_GAS_KEY=<키> python pull_done.py --dry-run
  --dry-run 을 빼면 실제로 받는다.  --who 수임 처럼 담당자 지정 (기본 종빈).
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.environ.get("LABELER_DATA", os.path.join(os.path.dirname(HERE), "2_dataset", "data", "img"))
REQUIRED = ["paving_mask.png"]
ONE_OF = ["straight_mask.png", "stop_mask.png"]


def local_done(n):
    d = os.path.join(LOCAL, str(n), "fix")
    return all(os.path.exists(os.path.join(d, f)) for f in REQUIRED) and any(os.path.exists(os.path.join(d, f)) for f in ONE_OF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", default="종빈")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import drive_sync as drv
    if not drv.enabled:
        sys.exit("LABELER_DRIVE_ID 또는 LABELER_GAS_URL(+LABELER_GAS_KEY) 환경변수를 넣고 실행하세요.")

    with open(os.path.join(HERE, "assignments.json"), encoding="utf-8") as f:
        mine = {int(k) for k, v in json.load(f).items() if v == a.who}
    _, drive_done = drv.index(REQUIRED, any_of=ONE_OF)
    targets = sorted(n for n in mine & set(drive_done) if not local_done(n))
    print(f"{a.who} 담당 {len(mine)}장 / Drive 완료 {len(mine & set(drive_done))}장 / 로컬 미완료라 받을 것 {len(targets)}장")
    print(targets)
    if a.dry_run or not targets:
        return

    for n in targets:
        cache = os.path.join(drv.CACHE, str(n), "fix")
        dst = os.path.join(LOCAL, str(n), "fix")
        os.makedirs(cache, exist_ok=True)
        orig = os.path.join(dst, "original.jpeg")
        if os.path.exists(orig) and not os.path.exists(os.path.join(cache, "original.jpeg")):
            shutil.copy2(orig, cache)   # 원본은 다시 내려받지 않게 캐시에 미리 둔다
        drv._pull(n)
        got = [f for f in os.listdir(cache) if f.endswith("_mask.png") or f == "points.json"]
        for f in os.listdir(dst):       # 로컬에 남은 옛 마스크는 Drive 상태로 맞춘다
            if f.endswith("_mask.png") and f not in got:
                os.remove(os.path.join(dst, f))
        for f in got:
            shutil.copy2(os.path.join(cache, f), os.path.join(dst, f))
        print(f"saved {n}: {sorted(got)}", "" if local_done(n) else "(⚠ 받은 뒤에도 미완료)")


if __name__ == "__main__":
    main()
