"""점자블록 데이터셋 라벨링 도구 (Gradio).

data/img/<번호>/fix/original.jpeg 를 한 장씩 열어, 점을 찍고 선으로 이어 5종 영역(정답·파손·배치·직진·정지)을 직접 그린다.
저장하면 data/img/<번호>/fix/ 에 원본·오버레이·마스크 PNG 와 points.json 이 만들어지고,
필요한 PNG 가 모두 있는 이미지는 다음부터 건너뛴다. 저장된 points.json 은 다시 열어 이어서 수정할 수 있다.

실행:  python labeler.py [--port 7860]
환경변수로 경로 바꾸기:  LABELER_DATA=이미지·라벨 폴더
                         LABELER_JSON=AI Hub 라벨링 JSON 폴더  LABELER_MAP=rename_map.csv (이미지 번호 <-> 원본 파일명)
화면: [라벨링] 5종 영역 그리기 / [파손 비교 (JSON)] 파손을 직접 그리고 AI Hub JSON 파손 박스와 겹쳐 비교
"""
import os

import argparse
import copy
import csv
import functools
import json
import math
import re

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import drive_sync as drv

HERE = os.path.dirname(os.path.abspath(__file__))
# LABELER_DRIVE_ID 가 있으면 Google Drive 공유 폴더를 직접 읽고 쓴다 (SAVE_ROOT 는 내려받은 파일을 두는 임시 캐시)
SAVE_ROOT = drv.CACHE if drv.enabled else os.environ.get("LABELER_DATA", os.path.join(HERE, "data", "img"))   # <번호>/fix/ 에 원본(original.jpeg)과 라벨이 함께 있다
ORIGINAL = "original.jpeg"
_ROOT = os.path.dirname(HERE)
JSON_DIR = os.environ.get("LABELER_JSON", os.path.join(_ROOT, "trash", "통행시설물_점자블록_교체폐기_라벨링_01"))
RENAME_MAP = os.environ.get("LABELER_MAP", os.path.join(
    _ROOT, "extracted_download", "113.노후_시설물_이미지_데이터", "01.데이터", "1.Training", "원천데이터", "rename_map.csv"))


def list_images(root):
    """root/<번호>/fix/original.jpeg 경로를 번호 순으로 반환."""
    nums = sorted(int(d) for d in os.listdir(root) if d.isdigit())
    return [p for p in (os.path.join(root, str(n), "fix", ORIGINAL) for n in nums) if os.path.exists(p)]


ALL_FILES = list_images(SAVE_ROOT) if os.path.isdir(SAVE_ROOT) else []
N = len(ALL_FILES)


@functools.lru_cache(maxsize=3)
def load_np(path):
    """이미지를 RGB numpy 로 읽어 최근 3장까지 캐시 (반환 배열은 수정하지 말 것)."""
    return np.array(Image.open(path).convert("RGB"))


def _need_images():
    if not ALL_FILES:
        raise gr.Error(f"이미지가 없습니다: {SAVE_ROOT}/<번호>/fix/{ORIGINAL}")


def _stem(i):
    return os.path.basename(os.path.dirname(os.path.dirname(ALL_FILES[i])))  # <번호>


def _fix_dir(i):
    return os.path.join(SAVE_ROOT, _stem(i), "fix")



# ====================================================================================
# 라벨링 로직 — 점 찍기 / 선 추가로 5종 영역을 그리고 data/img/<번호>/fix/ 에 저장
# ====================================================================================
EDIT_STEPS = ["① 정답 (점자 영역)", "② 파손", "③ 배치 (잘못된 배치)", "④ 직진 블록", "⑤ 정지 블록"]
EDIT_SHORT = ["정답", "파손", "배치", "직진", "정지"]
EDIT_FILES = ["paving", "damage", "misplaced", "straight", "stop"]
EDIT_COLORS = [(0, 200, 0), (255, 70, 70), (70, 120, 255), (255, 170, 0), (180, 80, 255)]
EDIT_TOOLS = ["점 찍기", "선 추가 (점 2개 선택)", "지우개 (점·선 클릭)"]
EDIT_VIEW_MAX = 1600  # 편집 화면에 표시하는 이미지의 긴 변 최대 픽셀 (클릭마다 전송되는 용량을 줄임)
OPTIONAL_FILES = {"damage", "misplaced"}  # 파손/배치는 없어도 됨 (그린 경우에만 마스크 저장)
# fix/ 에 이 PNG 가 전부 있으면 '완료'로 보고 순회 때 건너뛴다.
# 정답/직진/정지 마스크는 영역이 없어도 빈(검은) 마스크로 저장하므로 항상 존재한다.
REQUIRED_PNGS = [f"{n}_mask.png" for n in EDIT_FILES if n not in OPTIONAL_FILES]

DONE = set()  # Drive 모드: 완료된 번호 (시작 때 한 번 조회해 메모리에 두고 저장할 때마다 갱신)
if drv.enabled:
    _nums, DONE = drv.index(REQUIRED_PNGS)
    ALL_FILES = [os.path.join(SAVE_ROOT, str(n), "fix", ORIGINAL) for n in _nums]
    N = len(ALL_FILES)


def edit_scale(image):
    """원본 -> 편집 화면 표시 이미지의 배율 (1 이하). 점 좌표는 항상 원본 기준으로 저장한다."""
    return min(1.0, EDIT_VIEW_MAX / max(image.shape[:2]))


def _orient(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_dist(p, a, b):
    """점 p 와 선분 ab 사이의 거리."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = 0 if dx == dy == 0 else max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def seg_cross(p1, p2, p3, p4):
    """두 선분이 끝점이 아닌 곳에서 서로 가로지르면 True."""
    return _orient(p3, p4, p1) * _orient(p3, p4, p2) < 0 and _orient(p1, p2, p3) * _orient(p1, p2, p4) < 0


def find_faces(pts, lines):
    """선으로 둘러싸인 모든 닫힌 영역(면)을 찾는다. 점 하나에 선이 여러 개 붙어도(이미 닫은 영역의 점에서
    다른 점으로 다시 연결해도) 각각의 영역이 만들어진다. 선끼리 교차하지 않는다는 전제(평면 그래프).
    반환: (영역들[각 영역은 [x, y] 목록], 영역이 되지 못한 연결 덩어리 수)."""
    n = len(pts)
    adj = [[] for _ in range(n)]
    for a, b in lines:
        adj[a].append(b)
        adj[b].append(a)
    for v in range(n):  # 각 점의 이웃을 각도순으로 정렬
        adj[v].sort(key=lambda u: math.atan2(pts[u][1] - pts[v][1], pts[u][0] - pts[v][0]))
    used, faces, face_nodes = set(), [], set()
    for u in range(n):
        for v in adj[u]:
            if (u, v) in used:
                continue
            walk, a, b = [], u, v
            while (a, b) not in used:
                used.add((a, b))
                walk.append(a)
                nb = adj[b]
                c = nb[(nb.index(a) - 1) % len(nb)]  # 같은 면을 왼쪽에 두고 다음 변으로
                a, b = b, c
            area = 0.5 * sum(pts[walk[i]][0] * pts[walk[(i + 1) % len(walk)]][1]
                             - pts[walk[(i + 1) % len(walk)]][0] * pts[walk[i]][1] for i in range(len(walk)))
            if area > 1 and len(set(walk)) >= 3:  # 부호가 양수인 것이 안쪽 면, 음수는 바깥쪽 면
                faces.append([list(pts[i]) for i in walk])
                face_nodes |= set(walk)
    # 선이 있는데도 어떤 면에도 속하지 못한 연결 덩어리 (열린 선 등)
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in lines:
        parent[root(a)] = root(b)
    comps = {root(a) for a, _ in lines}
    covered = {root(i) for i in face_nodes}
    return faces, len(comps - covered)


@functools.lru_cache(maxsize=1)
def _orig_names():
    """이미지 번호(파일명 stem) -> AI Hub 원본 파일명 stem (rename_map.csv). 없으면 빈 dict."""
    try:
        with open(RENAME_MAP, encoding="utf-8") as f:
            return {os.path.splitext(r["new"])[0]: os.path.splitext(r["original"])[0] for r in csv.DictReader(f)}
    except OSError:
        return {}


@functools.lru_cache(maxsize=64)
def json_damage_boxes(i):
    """이미지 i 의 AI Hub JSON 파손 박스 [(종류, [x, y, w, h])] (원본 이미지 좌표). JSON 을 못 찾으면 None."""
    orig = _orig_names().get(_stem(i))
    path = os.path.join(JSON_DIR, f"{orig}.json") if orig else None
    if not path or not os.path.exists(path):  # 폴더에 없으면 data/img/<번호>/ 에 옮겨둔 JSON 을 쓴다
        path = os.path.join(SAVE_ROOT, _stem(i), f"{orig}.json") if orig else None
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    h, w_ = load_np(ALL_FILES[i]).shape[:2]
    sx, sy = w_ / j["images"][0]["width"], h / j["images"][0]["height"]
    return [(a["attributes"].get("damageType", "?"), [x * sx, y * sy, bw * sx, bh * sy])
            for a in j["annotations"] if a["category_id"] == 2 for x, y, bw, bh in [a["bbox"]]]


def cmp_text(st):
    """직접 그린 파손 영역과 JSON 파손 박스(합집합)의 겹침 수치."""
    boxes = json_damage_boxes(st["idx"])
    if boxes is None:
        return "JSON 을 찾지 못했습니다 (LABELER_JSON / LABELER_MAP 경로 확인)."
    h, w_ = load_np(ALL_FILES[st["idx"]]).shape[:2]
    k = max(1, max(h, w_) // 400)
    mine, box = Image.new("L", (w_ // k + 1, h // k + 1), 0), Image.new("L", (w_ // k + 1, h // k + 1), 0)
    for f in find_faces(st["data"][1]["pts"], st["data"][1]["lines"])[0]:
        ImageDraw.Draw(mine).polygon([(x / k, y / k) for x, y in f], fill=255)
    for _, (x, y, bw, bh) in boxes:
        ImageDraw.Draw(box).rectangle([x / k, y / k, (x + bw) / k, (y + bh) / k], fill=255)
    a, b = np.array(mine) > 0, np.array(box) > 0
    inter, union = (a & b).sum(), (a | b).sum()
    pct = lambda v: f"{v * 100:.1f}%"
    kinds = ", ".join(sorted({k_ for k_, _ in boxes})) or "-"
    return (f"[JSON 비교] 파손 박스 {len(boxes)}개 ({kinds}) · 내 파손 면적 {pct(a.mean())} · 박스 면적 {pct(b.mean())}\n"
            f"겹침 IoU {pct(inter / union) if union else '-'} · 내 파손 중 박스 안 비율 {pct(inter / a.sum()) if a.any() else '-'}")


def draw_edit(image, st):
    """5종 결과를 한 화면에 그림. 현재 단계는 진하게(점 번호 표시), 나머지 단계는 흐리게.
    표시용으로 줄인 이미지 위에 그리며, 점/선 좌표(원본 기준)에 배율을 곱해 위치를 맞춘다."""
    s_ = edit_scale(image)
    img = Image.fromarray(image).convert("RGBA")
    if s_ < 1:
        img = img.resize((round(img.width * s_), round(img.height * s_)), Image.BILINEAR)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lw = max(2, min(img.size) // 250)
    r = lw * 2.5
    font = ImageFont.load_default(size=max(14, min(img.size) // 45))
    if st.get("cmp"):
        for _, (x, y, bw, bh) in json_damage_boxes(st["idx"]) or []:
            d.rectangle([x * s_, y * s_, (x + bw) * s_, (y + bh) * s_], fill=(10, 132, 255, 60))
    for s in sorted(range(len(st["data"])), key=lambda k: k == st["step"]):  # 현재 단계를 마지막에 그려 다른 단계에 가려지지 않게
        data = st["data"][s]
        if st.get("cmp") and s != 1:
            continue  # 파손 비교 화면에서는 파손만 보여준다
        cur = s == st["step"]
        col = EDIT_COLORS[s]
        solid, faint = col + (255 if cur else 110,), col + (140 if cur else 40,)  # 현재 단계의 완료된 영역은 진하게
        P = [(x * s_, y * s_) for x, y in data["pts"]]
        for face in find_faces(data["pts"], data["lines"])[0]:
            d.polygon([(x * s_, y * s_) for x, y in face], fill=faint, outline=solid if cur else None, width=lw if cur else 1)
        for a_, b_ in data["lines"]:
            d.line([P[a_], P[b_]], fill=solid, width=lw)
        for i, (x, y) in enumerate(P):
            d.ellipse([x - r, y - r, x + r, y + r], fill=solid)
            if cur:
                d.text((x + r + 2, y - r - 2), str(i + 1), fill=(255, 255, 255, 255), font=font,
                       stroke_width=2, stroke_fill=(0, 0, 0, 255))
        if cur and st["sel"] is not None and st["sel"] < len(P):
            x, y = P[st["sel"]]
            d.ellipse([x - r * 2, y - r * 2, x + r * 2, y + r * 2], outline=(255, 215, 0, 255), width=lw)
    if st.get("cmp"):  # AI Hub JSON 파손 박스: 윤곽선과 종류만 (채움은 내 영역 아래에 먼저 그림)
        for kind, (x, y, bw, bh) in json_damage_boxes(st["idx"]) or []:
            d.rectangle([x * s_, y * s_, (x + bw) * s_, (y + bh) * s_], outline=(10, 132, 255, 255), width=lw)
            d.text((x * s_ + 4, y * s_ + 4), kind, fill=(255, 255, 255, 255), font=font, stroke_width=2, stroke_fill=(10, 80, 200, 255))
    return np.array(Image.alpha_composite(img, layer).convert("RGB"))


def edit_status(st):
    lines = []
    for s, data in enumerate(st["data"]):
        faces, op = find_faces(data["pts"], data["lines"])
        mark = "▶ " if s == st["step"] else "   "
        lines.append(f"{mark}{EDIT_SHORT[s]}: 점 {len(data['pts'])}개, 선 {len(data['lines'])}개, 영역 {len(faces)}개"
                     + (f", 영역이 안 된 선 묶음 {op}개" if op else "") + (" 🔒완료(저장됨)" if s in st["locked"] else ""))
    return "\n".join(lines)


def is_complete(i):
    if drv.enabled:
        return int(_stem(i)) in DONE
    d = _fix_dir(i)
    return all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_PNGS)


def load_fix(idx):
    """저장된 fix/points.json 이 있으면 불러와 이어서 수정할 수 있게 한다."""
    data = [{"pts": [], "lines": []} for _ in EDIT_FILES]
    path = os.path.join(_fix_dir(idx), "points.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                steps = json.load(f).get("steps", {})
            for s, name in enumerate(EDIT_FILES):
                if name in steps:
                    data[s] = {"pts": [list(q) for q in steps[name].get("pts", [])],
                               "lines": [list(q) for q in steps[name].get("lines", [])]}
        except (OSError, ValueError):
            pass
    return data


def ds_new(idx, cmp=False):
    """cmp=True: 파손 비교 화면용 (파손 단계만 편집, JSON 박스를 겹쳐 보여줌).
    touched: 이번에 수정한 단계 — 저장할 때 나머지 단계는 디스크의 최신 내용을 써서 다른 화면의 저장을 덮어쓰지 않는다."""
    if drv.enabled:
        drv.pull(int(_stem(idx)))  # 이 번호의 fix/ 를 Drive 에서 내려받는다
    data = load_fix(idx)
    locked = [s for s, d in enumerate(data) if find_faces(d["pts"], d["lines"])[0]]  # 이미 그려 저장된 단계 (표시용)
    return {"idx": idx, "step": 1 if cmp else 0, "tool": 0, "sel": None, "dirty": False, "touched": [],
            "locked": locked, "cmp": cmp, "data": data}


def _unlock(st):
    """현재 단계가 '완료(🔒)' 표시 상태면 표시를 푼다 (수정을 막지는 않는다)."""
    if st["step"] in st["locked"]:
        st["locked"].remove(st["step"])


def _touch(st):
    st["dirty"] = True
    if st["step"] not in st["touched"]:
        st["touched"].append(st["step"])


def ds_head(st):
    if st.get("cmp"):
        return (f"### 이미지 {st['idx'] + 1} / {N} · `{_stem(st['idx'])}번`\n"
                f"파란 박스 = AI Hub JSON 파손 · 저장: `data/img/{_stem(st['idx'])}/fix/damage_mask.png`")
    done = sum(is_complete(i) for i in range(N))
    cur = "✅ 완료" if is_complete(st["idx"]) else "⬜ 미완료"
    return (f"### 이미지 {st['idx'] + 1} / {N} · `{_stem(st['idx'])}번` · {cur}\n"
            f"완료 {done}장 · 남음 {N - done}장 · 저장: `data/img/{_stem(st['idx'])}/fix/`")


def _ds_out(st, extra="", head=False, radios=False):
    """편집 화면 갱신값 7개: 캔버스, 상태, 상태 문구, 머리말, 단계 라디오, 도구 라디오, 저장 안 한 변경 여부."""
    msg = ((extra + "\n" if extra else "") + edit_status(st) + ("\n" + cmp_text(st) if st.get("cmp") else "")
           + ("\n● 저장하지 않은 변경이 있습니다" if st["dirty"] else ""))
    return (draw_edit(load_np(ALL_FILES[st["idx"]]), st), st, msg,
            ds_head(st) if head else gr.skip(),
            gr.update(value=EDIT_STEPS[st["step"]]) if radios else gr.skip(),
            gr.update(value=EDIT_TOOLS[st["tool"]]) if radios else gr.skip(),
            bool(st["dirty"]))


def _need_state(st):
    _need_images()
    if st is None:
        raise gr.Error("페이지를 새로고침한 뒤 다시 시도하세요.")


def ds_click(st, evt: gr.SelectData):
    _need_state(st)
    image = load_np(ALL_FILES[st["idx"]])
    x, y = evt.index
    if x != x or y != y:
        return (gr.skip(),) * 7
    s_ = edit_scale(image)
    x, y = x / s_, y / s_  # 표시 이미지 좌표 -> 원본 좌표
    st = copy.deepcopy(st)
    _unlock(st)  # 이미 그려진 단계도 바로 이어서 그리거나 지울 수 있다
    data = st["data"][st["step"]]
    big = max(image.shape[:2])
    near = [(math.hypot(x - px, y - py), i) for i, (px, py) in enumerate(data["pts"])]
    near = min(near) if near else (math.inf, None)
    if st["tool"] == 2:  # 지우개: 점(붙은 선 포함) 우선, 없으면 가장 가까운 선
        reach = max(15, 0.03 * big)
        if near[1] is not None and near[0] <= reach:
            k = near[1]
            data["pts"].pop(k)
            data["lines"] = [[a - (a > k), b - (b > k)] for a, b in data["lines"] if k not in (a, b)]
            _touch(st)
            return _ds_out(st, f"점 {k + 1} 과 붙은 선을 지웠습니다.")
        segs = [(_seg_dist((x, y), data["pts"][a], data["pts"][b]), i) for i, (a, b) in enumerate(data["lines"])]
        if segs and min(segs)[0] <= reach:
            a, b = data["lines"].pop(min(segs)[1])
            _touch(st)
            return _ds_out(st, f"선 {a + 1} - {b + 1} 을 지웠습니다.")
        gr.Info("점이나 선 가까이를 클릭하세요.")
        return _ds_out(st)
    if st["tool"] == 0:  # 점 찍기
        if near[0] <= max(6, 0.006 * big):
            gr.Info("이미 점이 있는 위치입니다.")
            return _ds_out(st)
        data["pts"].append([int(x), int(y)])
        _touch(st)
        return _ds_out(st)
    # 선 추가: 점 2개를 차례로 선택하면 선이 그려짐 (이미 닫힌 영역의 점이어도 다른 점과 연결 가능)
    if near[1] is None or near[0] > max(15, 0.03 * big):
        gr.Info("점 가까이를 클릭하세요.")
        return _ds_out(st)
    i = near[1]
    if st["sel"] is None:
        st["sel"] = i
        return _ds_out(st, f"점 {i + 1} 선택됨 — 이어줄 다른 점을 클릭하세요.")
    if st["sel"] == i:
        st["sel"] = None
        return _ds_out(st, "선택을 취소했습니다.")
    a, b = sorted([st["sel"], i])
    st["sel"] = None
    if [a, b] in data["lines"]:
        return _ds_out(st, "이미 이어진 선입니다.")
    pa, pb = data["pts"][a], data["pts"][b]
    for c, d_ in data["lines"]:
        if len({a, b, c, d_}) == 4 and seg_cross(pa, pb, data["pts"][c], data["pts"][d_]):
            gr.Info("다른 선과 교차합니다. 교차 지점에 점을 먼저 찍고 나눠서 이어주세요.")
            return _ds_out(st, "교차하는 선은 추가할 수 없습니다.")
    data["lines"].append([a, b])
    _touch(st)
    return _ds_out(st, f"점 {a + 1} - {b + 1} 선을 추가했습니다.")


def ds_set_step(st, label):
    _need_state(st)
    st = copy.deepcopy(st)
    st["step"], st["tool"], st["sel"] = EDIT_STEPS.index(label), 0, None  # 단계를 바꾸면 도구는 '점 찍기'로
    return _ds_out(st, radios=True)


def ds_set_tool(st, label):
    _need_state(st)
    st = copy.deepcopy(st)
    st["tool"], st["sel"] = EDIT_TOOLS.index(label), None
    return _ds_out(st)


def ds_reset_step(st):
    """현재 단계를 마지막으로 저장한 상태로 되돌린다 (저장한 적 없으면 빈 단계)."""
    _need_state(st)
    st = copy.deepcopy(st)
    saved = load_fix(st["idx"])[st["step"]]
    st["data"][st["step"]] = saved
    st["sel"] = None
    if st["step"] in st["touched"]:
        st["touched"].remove(st["step"])
    st["dirty"] = bool(st["touched"])
    if find_faces(saved["pts"], saved["lines"])[0] and st["step"] not in st["locked"]:
        st["locked"].append(st["step"])  # 저장된 영역이 있는 단계는 다시 완료 표시
    return _ds_out(st, f"{EDIT_SHORT[st['step']]} 단계를 마지막 저장본으로 되돌렸습니다.")


def ds_undo_point(st):
    _need_state(st)
    st = copy.deepcopy(st)
    _unlock(st)
    data = st["data"][st["step"]]
    if data["pts"]:
        k = len(data["pts"]) - 1
        data["pts"].pop()
        data["lines"] = [ln for ln in data["lines"] if k not in ln]  # 지운 점에 붙은 선도 함께 삭제
        _touch(st)
    st["sel"] = None
    return _ds_out(st)


def ds_undo_line(st):
    _need_state(st)
    st = copy.deepcopy(st)
    _unlock(st)
    data = st["data"][st["step"]]
    if data["lines"]:
        data["lines"].pop()
        _touch(st)
    st["sel"] = None
    return _ds_out(st)


def ds_clear_step(st):
    """현재 단계를 빈 상태로 초기화."""
    _need_state(st)
    st = copy.deepcopy(st)
    _unlock(st)
    st["data"][st["step"]] = {"pts": [], "lines": []}
    st["sel"] = None
    _touch(st)
    return _ds_out(st)


def _save_if(st, save):
    """화면에서 '저장하시겠습니까' 에 확인을 눌렀고 바뀐 내용이 있으면 저장하고 안내 문구를 돌려준다."""
    return _save_fix(st) + "\n" if save and st["dirty"] else ""


def ds_next_step(st, dirty=False, save=False):
    """다음 단계로 (건너뛰기). 현재 단계의 그림은 지우지 않는다. 도구는 '점 찍기'로 초기화."""
    _need_state(st)
    st = copy.deepcopy(st)
    if st.get("cmp"):
        return _ds_out(st, "파손 비교 화면에는 파손 단계만 있습니다.")
    msg = _save_if(st, save)
    st["step"], st["tool"], st["sel"] = min(st["step"] + 1, len(EDIT_STEPS) - 1), 0, None
    return _ds_out(st, msg, radios=True)


def _save_fix(st):
    """data/img/<번호>/fix/ 에 5종 마스크 PNG + points.json 저장.
    정답/직진/정지는 영역이 없어도 빈(검은) 마스크로 저장하고, 파손/배치는 그린 경우에만 저장한다.
    fix/ 에 REQUIRED_PNGS 가 모두 있으면 '완료'로 판단해 순회 때 건너뛴다."""
    img = load_np(ALL_FILES[st["idx"]])
    h, w_ = img.shape[:2]
    fresh = load_fix(st["idx"])  # 수정하지 않은 단계는 디스크의 최신 내용 유지 (다른 화면에서 저장한 것 보호)
    st["data"] = [d if s in st["touched"] else fresh[s] for s, d in enumerate(st["data"])]
    fix = _fix_dir(st["idx"])
    os.makedirs(fix, exist_ok=True)
    faces = [find_faces(d["pts"], d["lines"])[0] for d in st["data"]]
    removed = []
    for s, fs in enumerate(faces):
        m = Image.new("L", (w_, h), 0)
        dr = ImageDraw.Draw(m)
        for f in fs:
            dr.polygon([tuple(p) for p in f], fill=255)
        mpath = os.path.join(fix, f"{EDIT_FILES[s]}_mask.png")
        if fs or EDIT_FILES[s] not in OPTIONAL_FILES:
            m.save(mpath)
        elif os.path.exists(mpath):
            os.remove(mpath)  # 파손/배치를 지웠다면 예전 마스크 파일도 정리
            removed.append(os.path.basename(mpath))
    with open(os.path.join(fix, "points.json"), "w", encoding="utf-8") as f:
        json.dump({"image_size": [w_, h],
                   "steps": {EDIT_FILES[s]: {**st["data"][s], "polygons": faces[s]} for s in range(len(EDIT_FILES))}},
                  f, ensure_ascii=False)
    st["dirty"], st["touched"] = False, []
    if drv.enabled:
        num = int(_stem(st["idx"]))
        DONE.add(num)
        drv.push_async(num, [n for n in os.listdir(fix) if n.endswith("_mask.png") or n == "points.json"], removed)
        if drv.errors:
            gr.Warning("Drive 업로드 실패:\n" + "\n".join(drv.errors[-3:]))
            drv.errors.clear()
    summary = ", ".join(f"{EDIT_SHORT[s]} {len(faces[s])}개" for s in range(len(faces)) if faces[s])
    if not summary:
        gr.Warning("영역이 하나도 없습니다 — 블록 없는 이미지로 저장했습니다.")
    return f"저장됨: {fix}\n영역: {summary or '없음'} (마스크 / points.json — 파손·배치는 그린 경우만)"


def _find_incomplete(start, direction):
    i = start
    while 0 <= i < N:
        if not is_complete(i):
            return i
        i += direction
    return None


def _goto(st, idx, extra="", warn=True, cmp=False):
    if warn and st and st.get("dirty"):
        gr.Warning("저장하지 않은 변경이 있어 버려졌습니다.")
    return _ds_out(ds_new(idx, cmp), extra, head=True, radios=True)


def _find(st, start, direction):
    """다음/이전 이미지 번호 (완료 여부와 상관없이 전체를 순서대로). 범위를 벗어나면 None."""
    return start if 0 <= start < N else None


def ds_init(cmp=False):
    """시작하자마자: 처음 이미지부터 (라벨링 화면은 이미 완료된 이미지를 건너뛰고 첫 미완료 이미지부터)."""
    _need_images()
    j = _find_incomplete(0, 1)  # 두 탭 모두 같은 이미지(첫 미완료)에서 시작
    return _goto(None, 0 if j is None else j, "" if j is not None else "모든 이미지가 완료되었습니다.", cmp=cmp)


def ds_prev(st, dirty=False, save=False):
    _need_state(st)
    st = copy.deepcopy(st)
    j = _find(st, st["idx"] - 1, -1)
    if j is None:
        return _ds_out(st, "첫 이미지입니다.")
    msg = _save_if(st, save)
    return _goto(st, j, msg, cmp=st.get("cmp", False))


def ds_next(st, dirty=False, save=False):
    _need_state(st)
    st = copy.deepcopy(st)
    j = _find(st, st["idx"] + 1, 1)
    if j is None:
        return _ds_out(st, "마지막 이미지입니다.")
    msg = _save_if(st, save)
    return _goto(st, j, msg, cmp=st.get("cmp", False))


def ds_jump(st, n, dirty=False, save=False):
    """번호로 직접 이동 (완료된 이미지도 열 수 있음 — 수정 후 저장하면 덮어씀)."""
    _need_images()
    _need_state(st)
    st = copy.deepcopy(st)
    idx = min(max(int(n or 1), 1), N) - 1
    msg = _save_if(st, save)
    if not st.get("cmp") and is_complete(idx):
        msg += "이미 완료된 이미지입니다 (저장하면 덮어씁니다)."
    return _goto(st, idx, msg, cmp=st.get("cmp", False))


def ds_save(st):
    _need_state(st)
    st = copy.deepcopy(st)
    msg = _save_fix(st)
    return st, ds_head(st), msg, False


def ds_save_next(st):
    """저장하고 다음 이미지로."""
    _need_state(st)
    st = copy.deepcopy(st)
    msg = _save_fix(st)
    j = _find(st, st["idx"] + 1, 1)
    if j is None:
        return _ds_out(st, "마지막 이미지입니다.", head=True) + (msg,)
    return _goto(st, j, warn=False, cmp=st.get("cmp", False)) + (msg,)



# ====================================================================================
# 화면
# ====================================================================================
# 이미지 위에 겹쳐 나오는 기본 전체보기/다운로드/공유 아이콘은 끄고, 이미지 밖(제목 아래) 버튼으로 대체
IMG_NO_OVERLAY = dict(show_download_button=False, show_fullscreen_button=False, show_share_button=False)
_JS_FULLSCREEN = ("() => { const i = document.querySelector('#%s img'); "
                  "if (i) { (i.requestFullscreen || i.webkitRequestFullscreen).call(i); } }")
_JS_DOWNLOAD = ("async () => { const i = document.querySelector('#%s img'); if (!i) return; "
                "const blob = await (await fetch(i.src)).blob(); const a = document.createElement('a'); "
                "a.href = URL.createObjectURL(blob); a.download = '%s'; document.body.appendChild(a); a.click(); "
                "a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }")
# 업로드용 입력 이미지 안쪽에 남는 전체화면 아이콘 숨김 (업로드/삭제 버튼은 그대로)
PAGE_CSS = """#img_input button[aria-label="View in full screen"] { display: none !important; }
.gradio-container { max-width: 1500px !important; }
.head h3 { margin: 0 0 2px; }
.head p { margin: 0; color: var(--body-text-color-subdued); }
.mono textarea { font-family: ui-monospace, Menlo, monospace; font-size: 12px; }"""


def img_tools(elem_id, filename):
    """이미지(elem_id) 바깥에 전체보기 / 다운로드 버튼을 둔다. 브라우저에서 바로 동작(서버 왕복 없음)."""
    with gr.Row():
        full = gr.Button("⛶ 전체보기", size="sm", scale=0, min_width=110)
        down = gr.Button("⬇ 다운로드", size="sm", scale=0, min_width=110)
    full.click(None, None, None, js=_JS_FULLSCREEN % elem_id)
    down.click(None, None, None, js=_JS_DOWNLOAD % (elem_id, filename))


# 저장하지 않은 변경이 있을 때만 '저장하시겠습니까' 팝업 (확인 = 저장 후 진행, 취소 = 저장 없이 진행)
_ASK = "d ? confirm('현재까지의 상태를 저장하시겠습니까?\\n\\n[확인] 저장하고 %s\\n[취소] 저장하지 않고 %s') : false"
JS_ASK_NAV = "(s, d, f) => [s, d, " + _ASK % ("이동", "이동") + "]"
JS_ASK_JUMP = "(s, n, d, f) => [s, n, d, " + _ASK % ("이동", "이동") + "]"
JS_ASK_STEP = "(s, d, f) => [s, d, " + _ASK % ("다음 단계로", "다음 단계로") + "]"


def build_editor(cmp=False):
    """편집 화면 하나를 만든다. cmp=True 면 파손만 그리고 JSON 파손 박스와 겹쳐 비교하는 화면."""
    sfx = "_cmp" if cmp else ""
    ds_state = gr.State(None)
    ds_dirty = gr.Checkbox(visible=False)  # 저장하지 않은 변경 여부 (브라우저 쪽 팝업 판단용)
    ds_save_flag = gr.Checkbox(visible=False)  # 팝업에서 '저장' 을 골랐는지
    ds_head_md = gr.Markdown("불러오는 중...", elem_classes="head")
    with gr.Accordion("사용법", open=False):
        if cmp:
            gr.Markdown("**파손**을 직접 그려 AI Hub JSON 의 파손 박스(파랑)와 비교합니다. `점 찍기` 로 모서리마다 점을 찍고 "
                        "`선 추가` 로 점 2개를 차례로 클릭해 영역을 닫으세요. 오른쪽 `상태` 에 겹침 수치가 바로 나옵니다. "
                        "저장하면 라벨링 화면과 같은 `fix/damage_mask.png` 에 기록됩니다.")
        else:
            gr.Markdown("**단계**를 고르고 `점 찍기`로 모서리마다 점을 찍은 뒤, `선 추가`로 바꿔 **점 2개를 차례로 클릭하면 선이 이어집니다.** "
                        "선이 한 바퀴 닫히면 안쪽이 영역이 되고, 이미 닫은 영역의 점에서도 다시 이어서 새 영역을 만들 수 있습니다. "
                        "저장하면 `fix/` 에 PNG 가 만들어지며(파손·배치는 안 그려도 됨), "
                        "이미 그려진 단계는 상태창에 🔒완료로 표시되며, 그 위에 바로 점·선을 더 그릴 수 있고 `지우개` 로 점이나 선을 클릭해 지울 수도 있습니다. "
                        "변경한 채 `다음`/`이전`/`이동`/`다음 단계` 를 누르면 저장할지 묻습니다.")
    with gr.Row(equal_height=True):
        ds_prev_btn = gr.Button("◀ 이전", scale=3)
        ds_next_btn = gr.Button("다음 ▶", scale=3)
        ds_jump_n = gr.Number(value=1, precision=0, minimum=1, maximum=max(N, 1), show_label=False, container=False,
                              min_width=80, scale=1)
        ds_jump_btn = gr.Button("번호로 이동", scale=2)
    with gr.Row():
        ds_step = gr.Radio([EDIT_STEPS[1]] if cmp else EDIT_STEPS, value=EDIT_STEPS[1 if cmp else 0], label="단계", scale=5)
        ds_tool = gr.Radio(EDIT_TOOLS, value=EDIT_TOOLS[0], label="도구", scale=3)
    with gr.Row():
        with gr.Column(scale=4):
            ds_canvas = gr.Image(label="편집 화면", show_label=False, interactive=False, elem_id="img_edit" + sfx, **IMG_NO_OVERLAY)
            img_tools("img_edit" + sfx, "edit.png")
        with gr.Column(scale=1, min_width=260):
            ds_msg = gr.Textbox(label="상태", lines=9 if cmp else 7, interactive=False, elem_classes="mono")
            with gr.Row():
                ds_save_btn = gr.Button("💾 저장", variant="secondary")
                ds_save_next_btn = gr.Button("💾 저장 후 다음 ▶", variant="primary")
            ds_save_msg = gr.Textbox(label="저장 결과", lines=3, interactive=False, elem_classes="mono")
            gr.Markdown("**편집**")
            with gr.Row():
                ds_undo_pt_btn = gr.Button("↩ 점 취소", size="sm")
                ds_undo_ln_btn = gr.Button("↩ 선 취소", size="sm")
            with gr.Row():
                ds_clear_btn = gr.Button("이 단계 초기화", size="sm")
                ds_reset_btn = gr.Button("↺ 저장본으로 되돌리기", size="sm")
            if not cmp:
                ds_nextstep_btn = gr.Button("다음 단계 ▶ (건너뛰기)", size="sm")

    DS_OUT = [ds_canvas, ds_state, ds_msg, ds_head_md, ds_step, ds_tool, ds_dirty]
    quiet = dict(show_progress="hidden", trigger_mode="multiple")  # 클릭마다 로딩 표시 없음, 빠른 연속 클릭도 모두 처리
    demo.load(lambda: ds_init(cmp), None, DS_OUT)
    ds_canvas.select(ds_click, [ds_state], DS_OUT, **quiet)
    ds_step.input(ds_set_step, [ds_state, ds_step], DS_OUT, **quiet)
    ds_tool.input(ds_set_tool, [ds_state, ds_tool], DS_OUT, **quiet)
    ds_undo_pt_btn.click(ds_undo_point, [ds_state], DS_OUT, **quiet)
    ds_undo_ln_btn.click(ds_undo_line, [ds_state], DS_OUT, **quiet)
    ds_clear_btn.click(ds_clear_step, [ds_state], DS_OUT, **quiet)
    ds_reset_btn.click(ds_reset_step, [ds_state], DS_OUT, **quiet)
    if not cmp:
        ds_nextstep_btn.click(ds_next_step, [ds_state, ds_dirty, ds_save_flag], DS_OUT, js=JS_ASK_STEP, **quiet)
    ds_prev_btn.click(ds_prev, [ds_state, ds_dirty, ds_save_flag], DS_OUT, js=JS_ASK_NAV)
    ds_next_btn.click(ds_next, [ds_state, ds_dirty, ds_save_flag], DS_OUT, js=JS_ASK_NAV)
    ds_jump_btn.click(ds_jump, [ds_state, ds_jump_n, ds_dirty, ds_save_flag], DS_OUT, js=JS_ASK_JUMP)
    ds_save_btn.click(ds_save, [ds_state], [ds_state, ds_head_md, ds_save_msg, ds_dirty])
    ds_save_next_btn.click(ds_save_next, [ds_state], DS_OUT + [ds_save_msg])
    return ds_state, DS_OUT


def ds_sync(src, dst, cmp):
    """탭을 열면 다른 탭이 보던 이미지로 맞춘다 (이 탭에 저장 안 한 변경이 있으면 그대로 둔다)."""
    if src is None or (dst is not None and dst["dirty"]):
        return (gr.skip(),) * 7
    return _goto(None, src["idx"], cmp=cmp)

with gr.Blocks(title="점자블록 데이터셋 라벨링", css=PAGE_CSS) as demo:
    gr.Markdown("## 점자블록 데이터셋 라벨링")
    gr.Markdown(f"이미지·라벨 `{SAVE_ROOT}/<번호>/fix/` ({N}장)" if N else
                f"⚠️ 이미지를 찾을 수 없습니다: `{SAVE_ROOT}/<번호>/fix/{ORIGINAL}`")
    with gr.Tab("라벨링") as tab_label:
        st_label, out_label = build_editor(False)
    with gr.Tab("파손 비교 (JSON)") as tab_cmp:
        st_cmp, out_cmp = build_editor(True)
    # 탭을 바꿀 때 보던 이미지와 디스크의 최신 저장 내용으로 맞춤
    tab_label.select(lambda a, b: ds_sync(a, b, False), [st_cmp, st_label], out_label)
    tab_cmp.select(lambda a, b: ds_sync(a, b, True), [st_label, st_cmp], out_cmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="점자블록 데이터셋 라벨링 도구")
    ap.add_argument("--port", type=int, default=7860)
    demo.launch(server_name="127.0.0.1", server_port=ap.parse_args().port)
