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
import base64
import functools
import html
import io
import json
import math
import re
import threading
import time

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
EDIT_VIEW_MAX = int(os.environ.get("LABELER_VIEW_MAX", 1600))  # 편집 화면에 표시하는 이미지의 긴 변 최대 픽셀 (클릭마다 전송되는 용량을 줄임)
OPTIONAL_FILES = {"damage", "misplaced"}  # 파손/배치는 없어도 됨 (그린 경우에만 마스크 저장)
# fix/ 에 이 PNG 가 전부 있으면 '완료'로 보고 순회 때 건너뛴다.
# 정답/직진/정지 마스크는 영역이 없어도 빈(검은) 마스크로 저장하므로 항상 존재한다.
# 완료 기준: 정답 마스크 + (직진 또는 정지 마스크 중 하나). 직진·정지는 그 이미지에 없을 수도 있어서 둘 중 하나만 있으면 된다.
# 마스크는 그 단계를 거치거나 그렸을 때만 저장되므로(빈 마스크 포함), 파일이 있다는 건 그 단계를 처리했다는 뜻이다.
REQUIRED_PNGS = ["paving_mask.png"]
ONE_OF_PNGS = ["straight_mask.png", "stop_mask.png"]


def masks_complete(names):
    """fix/ 에 있는 파일 이름들로 완료 여부를 판단한다."""
    return all(n in names for n in REQUIRED_PNGS) and any(n in names for n in ONE_OF_PNGS)

DONE = set()  # Drive 모드: 완료된 번호 (시작 때 한 번 조회해 메모리에 두고 저장할 때마다 갱신)
if drv.enabled:
    _nums, DONE = drv.index(REQUIRED_PNGS, extra=[f"{n}_mask.png" for n in EDIT_FILES] + ["points.json"], any_of=ONE_OF_PNGS)
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


_FACES = {}


def find_faces(pts, lines):
    """_find_faces 의 캐시판. 클릭할 때마다 5단계를 전부 다시 계산하지 않고, 바뀐 단계만 계산한다.
    반환값은 공유되므로 호출하는 쪽에서 수정하면 안 된다."""
    key = (tuple(map(tuple, pts)), tuple(map(tuple, lines)))
    if key not in _FACES:
        if len(_FACES) > 512:
            _FACES.clear()
        _FACES[key] = _find_faces(pts, lines)
    return _FACES[key]


def _find_faces(pts, lines):
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


try:
    with open(os.path.join(HERE, "damage_boxes.json"), encoding="utf-8") as _f:
        DAMAGE_BOXES = json.load(_f)   # 번호 -> {size: [원본 폭, 높이], boxes: [[종류, x, y, w, h]]}
except (OSError, ValueError):
    DAMAGE_BOXES = {}


def _saved_boxes(i):
    """damage_boxes.json 에서 읽은 파손 박스 (원본 JSON 과 같은 형식). 번호가 overview.json 에 있는데 박스가 없으면 파손 박스 없음([])."""
    num = _stem(i)
    e = DAMAGE_BOXES.get(num)
    if e is None:
        return [] if DAMAGE_BOXES and num in OVERVIEW else None
    h, w_ = load_np(ALL_FILES[i]).shape[:2]
    sx, sy = w_ / e["size"][0], h / e["size"][1]
    return [(k, [x * sx, y * sy, bw * sx, bh * sy]) for k, x, y, bw, bh in e["boxes"]]


@functools.lru_cache(maxsize=64)
def json_damage_boxes(i):
    """이미지 i 의 AI Hub JSON 파손 박스 [(종류, [x, y, w, h])] (원본 이미지 좌표). JSON 을 못 찾으면 None."""
    orig = _orig_names().get(_stem(i))
    path = os.path.join(JSON_DIR, f"{orig}.json") if orig else None
    if not path or not os.path.exists(path):  # 폴더에 없으면 data/img/<번호>/ 에 옮겨둔 JSON 을 쓴다
        path = os.path.join(SAVE_ROOT, _stem(i), f"{orig}.json") if orig else None
    if not path or not os.path.exists(path):
        return _saved_boxes(i)  # 원본 JSON 이 없는 서버(Drive·Render)는 build_overview.py 가 만든 좌표 파일을 쓴다
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


_BASE = {}


def _base_img(image):
    """표시용으로 줄인 RGBA 이미지를 이미지마다 한 번만 만든다 (image 자체를 값에 함께 보관해 id 재사용을 막는다)."""
    hit = _BASE.get(id(image))
    if hit and hit[0] is image:
        return hit[1]
    s_ = edit_scale(image)
    img = Image.fromarray(image).convert("RGBA")
    if s_ < 1:
        img = img.resize((round(img.width * s_), round(img.height * s_)), Image.BILINEAR)
    if len(_BASE) >= 3:
        _BASE.clear()
    _BASE[id(image)] = (image, img)
    return img


def draw_edit(image, st):
    """5종 결과를 한 화면에 그림. 현재 단계는 진하게(점 번호 표시), 나머지 단계는 흐리게.
    표시용으로 줄인 이미지 위에 그리며, 점/선 좌표(원본 기준)에 배율을 곱해 위치를 맞춘다."""
    s_ = edit_scale(image)
    img = _base_img(image)  # 합성은 새 이미지를 돌려주므로 캐시된 원본은 바뀌지 않는다
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
                     + (f", 영역이 안 된 선 묶음 {op}개" if op else "") + (" 완료(저장됨)" if s in st["locked"] else ""))
    return "\n".join(lines)


def is_complete(i):
    if drv.enabled:
        return int(_stem(i)) in DONE
    d = _fix_dir(i)
    return masks_complete(set(os.listdir(d)) if os.path.isdir(d) else set())


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


def ds_new(idx, cmp=False, sid=None):
    """cmp=True: 파손 비교 화면용 (파손 단계만 편집, JSON 박스를 겹쳐 보여줌). sid: 브라우저 세션 (작업 중 표시용).
    touched: 이번에 수정한 단계 — 저장할 때 나머지 단계는 디스크의 최신 내용을 써서 다른 화면의 저장을 덮어쓰지 않는다."""
    if drv.enabled:
        drv.pull(int(_stem(idx)), max_age=120)  # 이 번호의 fix/ 를 Drive 에서 내려받는다 (미리 받아 둔 건 건너뜀)
        drv.prefetch(*(int(_stem(j)) for j in (idx + 1, idx + 2) if j < N))  # 다음 두 장을 미리 받아 둔다
    data = load_fix(idx)
    locked = [s for s, d in enumerate(data) if find_faces(d["pts"], d["lines"])[0]]  # 이미 그려 저장된 단계 (표시용)
    first = 1 if cmp else 0
    # visited: 거쳐 간 단계. 이미 마스크가 저장된 단계는 거친 것으로 본다 (기존 데이터는 그대로 완료 유지).
    visited = sorted({first} | {s for s, n in enumerate(EDIT_FILES)
                                if os.path.exists(os.path.join(_fix_dir(idx), f"{n}_mask.png"))})
    st = {"idx": idx, "step": first, "tool": 0, "sel": None, "dirty": False, "touched": [],
          "locked": locked, "cmp": cmp, "data": data, "visited": visited, "sid": sid}
    _claim(st)
    return st


# 작업 중 표시: 번호 -> {(세션, cmp): 마지막 조작 시각}. 서버 프로세스 하나 안에서만 공유된다.
# 같은 브라우저 페이지(라벨링·파손 비교 탭)는 같은 세션이라 서로를 '다른 사람'으로 보지 않는다.
BUSY_TTL = 600  # 이 시간(초) 동안 아무 조작이 없으면 자리를 비운 것으로 본다 (창을 닫으면 바로 풀림)
_busy = {}
_busy_lock = threading.Lock()


def _claim(st):
    """이 화면이 지금 보는 번호를 '작업 중' 으로 표시한다 (이전에 보던 번호의 표시는 푼다)."""
    if not st or not st.get("sid"):
        return
    key, num = (st["sid"], bool(st.get("cmp"))), _stem(st["idx"])
    with _busy_lock:
        for n in [n for n, h in _busy.items() if key in h and n != num]:
            del _busy[n][key]
            if not _busy[n]:
                del _busy[n]
        _busy.setdefault(num, {})[key] = time.time()


def pick_who(who, request: gr.Request):
    """캐릭터를 골랐을 때: 그 사람의 첫 미완료 이미지로."""
    sid = request.session_hash
    if who:
        _WHO[sid] = who
    return go_mine(request)


def go_mine(request: gr.Request):
    sid = request.session_hash
    j = _first_mine(sid)
    if j is None:
        gr.Info("담당 번호를 모두 완료했습니다." if _WHO.get(sid) else "먼저 캐릭터를 골라 주세요.")
        j = _find_incomplete(0, 1, sid) or 0
    return _goto(None, j, cmp=False, sid=sid)


def _release(sid):
    """브라우저 창을 닫으면 그 세션의 표시를 모두 푼다."""
    _WHO.pop(sid, None)
    with _busy_lock:
        for n in list(_busy):
            _busy[n] = {k: t for k, t in _busy[n].items() if k[0] != sid}
            if not _busy[n]:
                del _busy[n]


def busy_others(i, sid):
    """i번째 이미지를 지금 다른 세션 몇 곳이 열어 두고 있는지."""
    now = time.time()
    with _busy_lock:
        return len({k[0] for k, t in _busy.get(_stem(i), {}).items() if k[0] != sid and now - t < BUSY_TTL})


def _unlock(st):
    """현재 단계가 '완료(🔒)' 표시 상태면 표시를 푼다 (수정을 막지는 않는다)."""
    if st["step"] in st["locked"]:
        st["locked"].remove(st["step"])


def _touch(st):
    st["dirty"] = True
    if st["step"] not in st["touched"]:
        st["touched"].append(st["step"])


try:
    with open(os.path.join(HERE, "assignments.json"), encoding="utf-8") as _f:
        ASSIGN = json.load(_f)   # 번호 -> 담당자 이름 (build_assignments.py 가 만든다)
except (OSError, ValueError):
    ASSIGN = {}
PEOPLE = ["수임", "윤진", "태희", "종빈"]  # 전체보기 카드·담당 필터 표시 순서
_WHO = {}   # 브라우저 세션 -> 이 세션이 고른 사람 (처음 접속 때 캐릭터를 눌러 정한다. 로그인이 아니라 자기 선택)


def _first_mine(sid):
    """고른 사람의 담당 번호 중 완료되지 않았고 다른 사람이 작업 중이지 않은 첫 이미지. 고른 사람이 없거나 다 끝났으면 None."""
    who = _WHO.get(sid)
    if not who:
        return None
    for i in range(N):
        if ASSIGN.get(_stem(i)) == who and not is_complete(i) and not busy_others(i, sid):
            return i
    return None


@functools.lru_cache(maxsize=None)
def person_img(name):
    """담당자 그림을 data URI 로 (파일 서버 설정 없이 로컬·배포 모두 같은 방식으로 보이게)."""
    try:
        with open(os.path.join(HERE, "assets", "people", f"{name}.jpg"), "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return ""


def _person_tag(who):
    img = person_img(who)
    return '<img src="%s" alt="%s">' % (img, who) if img else ""


def owner_badge(num):
    who = ASSIGN.get(str(num))
    if not who:
        return ""
    return f'<div class="owner">{_person_tag(who)}<span>담당 {who}</span></div>'


def ds_head(st):
    num = _stem(st["idx"])
    if st.get("cmp"):
        title = f"이미지 {st['idx'] + 1} / {N} · {num}번"
        sub = f"파란 박스 = AI Hub JSON 파손 · 저장: <code>data/img/{num}/fix/damage_mask.png</code>"
    else:
        done = sum(is_complete(i) for i in range(N))
        cur = "완료" if is_complete(st["idx"]) else "미완료"
        others = busy_others(st["idx"], st.get("sid"))
        cur += f" · 다른 사람이 작업 중{f' ({others}명)' if others > 1 else ''}" if others else ""
        title = f"이미지 {st['idx'] + 1} / {N} · {num}번 · {cur}"
        sub = f"완료 {done}장 · 남음 {N - done}장 · 저장: <code>data/img/{num}/fix/</code>"
    owner, who = ASSIGN.get(str(num)), _WHO.get(st.get("sid"))
    alien = f'<span class="alien" data-owner="{owner}" data-num="{num}" hidden></span>' if who and owner and owner != who else ""
    return f'<div class="head-row"><div class="head-main"><h3>{title}</h3><p>{sub}</p></div>{owner_badge(num)}{alien}</div>'


def _ds_out(st, extra="", head=False, radios=False):
    """편집 화면 갱신값 7개: 캔버스, 상태, 상태 문구, 머리말, 단계 라디오, 도구 라디오, 저장 안 한 변경 여부."""
    _claim(st)  # 조작할 때마다 '작업 중' 시각을 갱신
    msg = ((extra + "\n" if extra else "") + edit_status(st) + ("\n" + cmp_text(st) if st.get("cmp") else "")
           + ("\n[저장하지 않은 변경이 있습니다]" if st["dirty"] else ""))
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
    if st["step"] not in st["visited"]:
        st["visited"].append(st["step"])
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
    if st["step"] not in st["visited"]:
        st["visited"].append(st["step"])
    return _ds_out(st, msg, radios=True)


def _save_fix(st):
    """같은 번호를 두 화면이 동시에 저장하거나, 저장 도중 Drive 에서 받은 파일이 끼어들지 않게 번호별로 잠그고 저장한다."""
    with drv.lock(int(_stem(st["idx"]))):
        return _save_fix_locked(st)


def _save_fix_locked(st):
    """data/img/<번호>/fix/ 에 5종 마스크 PNG + points.json 저장.
    정답/직진/정지는 영역이 없어도 빈(검은) 마스크로 저장하고, 파손/배치는 그린 경우에만 저장한다.
    fix/ 에 정답 마스크와 (직진 또는 정지) 마스크가 있으면 '완료'로 판단해 순회 때 건너뛴다."""
    _claim(st)
    if busy_others(st["idx"], st.get("sid")):
        gr.Warning("다른 사람도 이 이미지를 작업 중입니다 — 같은 단계를 서로 고쳤다면 나중에 저장한 쪽이 남습니다.")
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
        required = EDIT_FILES[s] not in OPTIONAL_FILES
        # 정답·직진·정지는 영역이 없어도 그 단계를 거쳤으면 빈 마스크로 저장한다. 거치지 않은 단계는 저장하지 않아 '미완료'로 남는다.
        if fs or (required and s in st["visited"]):
            buf = io.BytesIO()
            m.save(buf, format="PNG")
            drv.write_file(mpath, buf.getvalue())
        elif not required and os.path.exists(mpath):
            os.remove(mpath)  # 파손/배치를 지웠다면 예전 마스크 파일도 정리
            removed.append(os.path.basename(mpath))
    drv.write_file(os.path.join(fix, "points.json"), json.dumps(
        {"image_size": [w_, h],
         "steps": {EDIT_FILES[s]: {**st["data"][s], "polygons": faces[s]} for s in range(len(EDIT_FILES))}},
        ensure_ascii=False).encode("utf-8"))  # 쓰는 도중에 다른 화면이 반쯤 쓰인 파일을 읽지 않게
    st["dirty"], st["touched"] = False, []
    if drv.enabled:
        num = int(_stem(st["idx"]))
        if masks_complete(set(os.listdir(fix))):
            DONE.add(num)
        else:
            DONE.discard(num)
        drv.push_async(num, [n for n in os.listdir(fix) if n.endswith("_mask.png") or n == "points.json"], removed)
        if drv.errors:
            gr.Warning("Drive 업로드 실패:\n" + "\n".join(drv.errors[-3:]))
            drv.errors.clear()
    summary = ", ".join(f"{EDIT_SHORT[s]} {len(faces[s])}개" for s in range(len(faces)) if faces[s])
    seen = {EDIT_FILES[s] for s in st["visited"]}
    todo = [] if st.get("cmp") else (["정답"] if "paving" not in seen else []) + (
        ["직진 또는 정지"] if not {"straight", "stop"} & seen else [])
    if todo:
        gr.Warning("아직 안 거친 단계: " + ", ".join(todo) + " — 정답과 직진·정지 중 하나를 거쳐야 완료됩니다.")
    elif not summary:
        gr.Warning("영역이 하나도 없습니다 — 블록 없는 이미지로 저장했습니다.")
    return (f"저장됨: {fix}\n영역: {summary or '없음'} (마스크 / points.json — 파손·배치는 그린 경우만)"
            + (f"\n미완료 — 남은 단계: {', '.join(todo)}" if todo else ""))


def _find_incomplete(start, direction, sid=None):
    """완료되지 않았고 다른 사람이 작업 중이지 않은 첫 이미지."""
    i = start
    while 0 <= i < N:
        if not is_complete(i) and not busy_others(i, sid):
            return i
        i += direction
    return None


def _goto(st, idx, extra="", warn=True, cmp=False, sid=None):
    if warn and st and st.get("dirty"):
        gr.Warning("저장하지 않은 변경이 있어 버려졌습니다.")
    sid = sid or (st or {}).get("sid")
    if busy_others(idx, sid):
        gr.Warning(f"{_stem(idx)}번은 다른 사람이 작업 중입니다.")
    return _ds_out(ds_new(idx, cmp, sid), extra, head=True, radios=True)


def _find(st, start, direction):
    """다음/이전 이미지 번호 (완료 여부와 상관없이 순서대로). 캐릭터를 고른 사람은 자기 담당 번호만 순회한다 (남의 번호는 번호 이동·전체보기로).
    범위를 벗어나면 None."""
    who = None if st.get("cmp") else _WHO.get(st.get("sid"))
    while who and 0 <= start < N and ASSIGN.get(_stem(start), who) != who:
        start += direction
    return start if 0 <= start < N else None


def ds_init(cmp=False, sid=None):
    """시작하자마자: 처음 이미지부터 (완료됐거나 다른 사람이 작업 중인 이미지를 건너뛰고 첫 미완료 이미지부터)."""
    _need_images()
    j = _first_mine(sid)  # 캐릭터를 고른 사람은 자기 담당의 첫 미완료부터
    if j is None:
        j = _find_incomplete(0, 1, sid)  # 두 탭 모두 같은 이미지(첫 미완료)에서 시작
    return _goto(None, 0 if j is None else j, "" if j is not None else "남은 이미지가 없습니다 (모두 완료됐거나 다른 사람이 작업 중).",
                 cmp=cmp, sid=sid)


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
IMG_NO_OVERLAY = dict(show_download_button=False, show_fullscreen_button=False, show_share_button=False,
                      format="jpeg")  # 기본 webp 는 인코딩이 느려 클릭마다 지연됨
_JS_FULLSCREEN = ("() => { const i = document.querySelector('#%s img'); "
                  "if (i) { (i.requestFullscreen || i.webkitRequestFullscreen).call(i); } }")
_JS_DOWNLOAD = ("async () => { const i = document.querySelector('#%s img'); if (!i) return; "
                "const blob = await (await fetch(i.src)).blob(); const a = document.createElement('a'); "
                "a.href = URL.createObjectURL(blob); a.download = '%s'; document.body.appendChild(a); a.click(); "
                "a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }")
# 업로드용 입력 이미지 안쪽에 남는 전체화면 아이콘 숨김 (업로드/삭제 버튼은 그대로)
BLUSH = gr.themes.Color(name="blush", c50="#fff7f9", c100="#fdecf0", c200="#fbd9e2", c300="#f7bfcd", c400="#f0a0b5",
                        c500="#e47f9a", c600="#cc6283", c700="#aa4d6b", c800="#873e55", c900="#6a3345", c950="#4a2230")
THEME = gr.themes.Base(
    primary_hue=BLUSH, secondary_hue=BLUSH, neutral_hue=gr.themes.colors.slate,
    font=["Pretendard", "Apple SD Gothic Neo", "Noto Sans KR", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "Menlo", "monospace"],
    radius_size=gr.themes.sizes.radius_sm, text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#faf8f9", body_text_color="#2b2428", body_text_color_subdued="#7d7479",
    block_background_fill="#ffffff", block_border_width="1px", block_border_color="#ece6e9", block_shadow="none",
    block_label_background_fill="transparent", block_label_text_color="#7d7479", block_label_text_weight="600",
    block_title_text_weight="600", block_title_text_color="#4a3f45",
    input_background_fill="#ffffff", input_border_color="#e4dce0", input_border_color_focus="#e47f9a", input_shadow_focus="0 0 0 3px #fdecf0",
    button_primary_background_fill="#f3b3c4", button_primary_background_fill_hover="#ee9db3", button_primary_text_color="#4a2230",
    button_primary_border_color="transparent",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_hover="#fff7f9",
    button_secondary_text_color="#3d3338", button_secondary_border_color="#e4dce0",
    checkbox_label_background_fill_selected="#fdecf0", checkbox_label_border_color_selected="#e47f9a",
    checkbox_label_text_color_selected="#6a3345", checkbox_background_color_selected="#e47f9a", checkbox_border_color_selected="#e47f9a",
    loader_color="#e47f9a", slider_color="#e47f9a",
)
PAGE_CSS = """@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css");
#img_input button[aria-label="View in full screen"] { display: none !important; }
.gradio-container { max-width: 1440px !important; font-family: Pretendard, "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif !important; }
footer { display: none !important; }
/* 상단 바 */
.app-header { display: flex; align-items: baseline; gap: 14px; padding: 6px 2px 14px; margin-bottom: 6px; border-bottom: 1px solid #ece6e9; }
.app-header .brand { font-size: 20px; font-weight: 700; letter-spacing: -0.02em; color: #2b2428; }
.app-header .brand b { color: #cc6283; font-weight: 700; }
.app-header .meta { font-size: 13px; color: #7d7479; }
.notice { padding: 10px 14px; border: 1px solid #f7bfcd; background: #fff7f9; border-radius: 6px; color: #6a3345; font-size: 14px; }
fieldset .wrap { gap: 6px !important; }
/* 탭 */
button[role="tab"] { font-weight: 600 !important; color: #7d7479 !important; padding: 10px 16px !important; }
button[role="tab"][aria-selected="true"] { color: #aa4d6b !important; border-bottom: 2px solid #e47f9a !important; }
.click-ack { position: fixed; width: 18px; height: 18px; border-radius: 50%; border: 2px solid #e47f9a; background: rgba(228,127,154,.35); pointer-events: none; z-index: 9999; }
/* 점을 찍을 때마다 머리말(번호·담당자 그림)이 흐려졌다 돌아오는 깜빡임 방지: 로딩 중 흐리게 하는 기본 효과를 끈다 */
.head .html-container, .head .html-container.pending { opacity: 1 !important; transition: none !important; }
#who_radio, #back_btn { position: absolute !important; left: -9999px !important; width: 1px !important; height: 1px !important; overflow: hidden !important; }
.change-who { margin-left: auto; padding: 5px 12px; font-size: 12.5px; color: #6a3345; background: #fff; border: 1px solid #e4dce0; border-radius: 6px; cursor: pointer; }
.change-who:hover { border-color: #e47f9a; background: #fff7f9; }
.who-overlay, .alien-overlay { position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center; justify-content: center; background: rgba(43, 36, 40, .55); }
.who-box, .alien-box { background: #fff; border-radius: 14px; padding: 26px 30px; text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,.25); }
.who-box h2, .alien-box h2 { margin: 0 0 6px; font-size: 22px; font-weight: 700; letter-spacing: -0.02em; color: #2b2428; }
.who-box p { margin: 0 0 16px; color: #7d7479; font-size: 14px; }
.who-grid { display: grid; grid-template-columns: repeat(4, 150px); gap: 12px; }
.who-card { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 8px 8px 10px; background: #fff; border: 1px solid #ece6e9; border-radius: 10px; cursor: pointer; font-weight: 600; color: #4a3f45; }
.who-card:hover { border-color: #e47f9a; background: #fff7f9; }
.who-card img { width: 100%; border-radius: 7px; }
.alien-box { max-width: 460px; border-top: 5px solid #cc6283; }
.alien-box img { width: 100%; border-radius: 8px; margin-bottom: 14px; }
.alien-box h3 { margin: 0 0 8px; font-size: 26px; font-weight: 800; color: #cc6283; }
.alien-box p { margin: 0 0 16px; color: #4a3f45; font-size: 14.5px; }
.alien-btns { display: flex; gap: 10px; justify-content: center; }
.alien-btns button { padding: 9px 16px; font-weight: 600; border-radius: 8px; cursor: pointer; border: 1px solid #e4dce0; background: #fff; color: #3d3338; }
.alien-btns button[data-act="back"] { background: #f3b3c4; border-color: transparent; color: #4a2230; }
/* 담당자 */
.head-row { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
.owner { display: flex; align-items: center; gap: 10px; padding: 3px 14px 3px 3px; background: #fff; border: 1px solid #ece6e9; border-radius: 10px; }
.owner img { height: 56px; border-radius: 7px; display: block; }
.owner span { font-size: 12.5px; font-weight: 600; color: #6a3345; white-space: nowrap; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat { display: flex; align-items: center; gap: 12px; padding: 8px 14px 8px 8px; background: #fff; border: 1px solid #ece6e9; border-radius: 10px; }
.stat img { height: 66px; border-radius: 7px; display: block; }
.stat .name { font-size: 12.5px; font-weight: 600; color: #7d7479; }
.stat .pct { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; color: #aa4d6b; line-height: 1.2; }
.stat .sub { font-size: 12px; color: #7d7479; }
.stat .bar { height: 5px; margin-top: 6px; background: #fdecf0; border-radius: 3px; overflow: hidden; }
.stat .bar i { display: block; height: 100%; background: #e47f9a; border-radius: 3px; }
.stat .body { flex: 1; min-width: 0; }
/* 머리말·섹션 */
.head h3 { margin: 0 0 2px; font-size: 17px; font-weight: 700; letter-spacing: -0.01em; color: #2b2428; }
.head p { margin: 0; color: #7d7479; font-size: 13px; }
.section-label p { margin: 6px 0 0; font-size: 12px; font-weight: 700; letter-spacing: 0.04em; color: #7d7479; }
.mono textarea { font-family: ui-monospace, Menlo, monospace; font-size: 12px; line-height: 1.55; }
/* 버튼은 작고 단정하게 */
button { font-weight: 600 !important; letter-spacing: -0.01em; }
button.primary:hover { filter: none; }
button.sm { font-size: 12.5px !important; }
button.secondary { border: 1px solid #e4dce0 !important; }
button.secondary:hover { border-color: #e47f9a !important; color: #6a3345 !important; }
/* 표: 셀 글꼴을 본문과 같게 */
.table-wrap, table, table td, table td *, table th, table th * { font-family: Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif !important; font-size: 13px; }
table td, table td * { font-weight: 500 !important; }
.pager { align-items: center !important; }
.pager p { margin: 0; color: #7d7479; font-size: 13px; }
table thead th { background: #fff7f9 !important; color: #4a3f45 !important; font-weight: 700 !important; }
table tbody tr:hover td { background: #fff7f9 !important; }
/* 아코디언 */
.gradio-container .label-wrap { font-weight: 600; color: #4a3f45; }"""


# 클릭한 자리에 바로 표시를 띄워 서버 응답을 기다리는 동안에도 클릭이 접수됐음을 보여준다 (서버 응답으로 화면이 바뀌면 사라진다)
JS_CLICK_ACK = """() => {
  if (window.__clickAck) return; window.__clickAck = true;
  const clear = () => document.querySelectorAll('.click-ack').forEach(n => n.remove());
  document.addEventListener('click', (e) => {
    const img = e.target.closest('#img_edit img, #img_edit_cmp img');
    if (!img) return;
    const host = img.closest('#img_edit, #img_edit_cmp');
    host.__ackSrc = img.src;  // 이 이미지가 새 이미지로 바뀌면(서버 응답 도착) 표시를 지운다
    const dot = document.createElement('div');
    dot.className = 'click-ack';
    dot.style.left = (e.clientX - 9) + 'px'; dot.style.top = (e.clientY - 9) + 'px';
    document.body.appendChild(dot);
    if (!host.__ackObs) {
      host.__ackObs = new MutationObserver(() => {
        const cur = host.querySelector('img');
        if (!cur || cur.src !== host.__ackSrc) clear();
      });
      host.__ackObs.observe(host, {subtree: true, childList: true, attributes: true, attributeFilter: ['src']});
    }
    setTimeout(clear, 8000);
  }, true);
}"""


def _data_uri(path):
    try:
        with open(path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return ""


# 처음 접속하면 '누구세요?' 에서 캐릭터를 고르고(브라우저에 기억), 남의 담당 번호를 열면 경고창을 띄운다.
# 고른 사람은 숨긴 라디오(#who_radio)를 눌러 서버에 알리고, 경고창의 '돌아가기'는 숨긴 버튼(#back_btn)을 누른다.
_JS_WHO = """() => {
  const PEOPLE = __PEOPLE__, ALERT = __ALERT__, KEY = 'labeler_who';
  const esc = (t) => String(t).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const mine = () => localStorage.getItem(KEY);
  const pick = (n) => { const i = document.querySelector('#who_radio input[value="' + n + '"]'); if (i) i.click(); };
  const picker = () => {
    if (document.querySelector('.who-overlay')) return;
    const o = document.createElement('div'); o.className = 'who-overlay';
    o.innerHTML = '<div class="who-box"><h2>누구세요?</h2><p>캐릭터를 누르면 내 담당 번호의 첫 미완료 이미지부터 시작합니다.</p><div class="who-grid">'
      + Object.keys(PEOPLE).map(n => '<button class="who-card" data-who="' + esc(n) + '"><img src="' + PEOPLE[n] + '"><span>' + esc(n) + '</span></button>').join('')
      + '</div></div>';
    document.body.appendChild(o);
  };
  const alien = (owner, num) => {
    document.querySelectorAll('.alien-overlay').forEach(n => n.remove());
    const o = document.createElement('div'); o.className = 'alien-overlay';
    o.innerHTML = '<div class="alien-box"><img src="' + ALERT + '"><h2>감히 내 것을 넘보느냐?</h2><h3>네 이놈!</h3>'
      + '<p>' + esc(num) + '번은 <b>' + esc(owner) + '</b> 담당입니다.</p>'
      + '<div class="alien-btns"><button data-act="back">내 담당으로 돌아가기</button><button data-act="stay">그래도 계속</button></div></div>';
    document.body.appendChild(o);
  };
  document.addEventListener('click', (e) => {
    const c = e.target.closest('.who-card');
    if (c) { localStorage.setItem(KEY, c.dataset.who); document.querySelector('.who-overlay').remove(); pick(c.dataset.who); return; }
    if (e.target.closest('.change-who')) { picker(); return; }
    const b = e.target.closest('.alien-btns button');
    if (b) { document.querySelector('.alien-overlay').remove(); if (b.dataset.act === 'back') document.querySelector('#back_btn').click(); }
  });
  let last = '', timer = null;
  const check = () => {
    const w = document.querySelector('.change-who'); if (w) w.textContent = mine() ? '내 캐릭터: ' + mine() + ' · 변경' : '캐릭터 선택';
    const head = [...document.querySelectorAll('.head')].find(h => h.offsetParent !== null);
    const m = head && head.querySelector('.alien');
    if (!m) { last = ''; return; }
    const key = m.dataset.num + '|' + m.dataset.owner;
    if (key !== last) { last = key; alien(m.dataset.owner, m.dataset.num); }
  };
  new MutationObserver(() => { clearTimeout(timer); timer = setTimeout(check, 80); }).observe(document.body, {subtree: true, childList: true});
  if (!mine()) picker();
}"""
JS_WHO = (_JS_WHO.replace("__PEOPLE__", json.dumps({n: person_img(n) for n in PEOPLE}, ensure_ascii=False))
          .replace("__ALERT__", json.dumps(_data_uri(os.path.join(HERE, "assets", "alert.jpg")))))
JS_PAGE = "() => { (%s)(); (%s)(); }" % (JS_CLICK_ACK, JS_WHO)


def img_tools(elem_id, filename):
    """이미지(elem_id) 바깥에 전체보기 / 다운로드 버튼을 둔다. 브라우저에서 바로 동작(서버 왕복 없음)."""
    with gr.Row():
        full = gr.Button("전체 화면", size="sm", scale=0, min_width=110)
        down = gr.Button("다운로드", size="sm", scale=0, min_width=110)
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
    ds_head_md = gr.HTML("불러오는 중...", elem_classes="head")
    with gr.Accordion("사용법", open=False):
        if cmp:
            gr.Markdown("**파손**을 직접 그려 AI Hub JSON 의 파손 박스(파랑)와 비교합니다. `점 찍기` 로 모서리마다 점을 찍고 "
                        "`선 추가` 로 점 2개를 차례로 클릭해 영역을 닫으세요. 오른쪽 `상태` 에 겹침 수치가 바로 나옵니다. "
                        "저장하면 라벨링 화면과 같은 `fix/damage_mask.png` 에 기록됩니다.")
        else:
            gr.Markdown("**단계**를 고르고 `점 찍기`로 모서리마다 점을 찍은 뒤, `선 추가`로 바꿔 **점 2개를 차례로 클릭하면 선이 이어집니다.** "
                        "선이 한 바퀴 닫히면 안쪽이 영역이 되고, 이미 닫은 영역의 점에서도 다시 이어서 새 영역을 만들 수 있습니다. "
                        "저장하면 `fix/` 에 PNG 가 만들어지며(파손·배치는 안 그려도 됨), "
                        "이미 그려진 단계는 상태창에 완료로 표시되며, 그 위에 바로 점·선을 더 그릴 수 있고 `지우개` 로 점이나 선을 클릭해 지울 수도 있습니다. "
                        "변경한 채 `다음`/`이전`/`이동`/`다음 단계` 를 누르면 저장할지 묻습니다.")
    with gr.Row(equal_height=True):
        ds_prev_btn = gr.Button("이전", scale=3)
        ds_next_btn = gr.Button("다음", scale=3)
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
                ds_save_btn = gr.Button("저장", variant="secondary")
                ds_save_next_btn = gr.Button("저장 후 다음", variant="primary")
            ds_save_msg = gr.Textbox(label="저장 결과", lines=3, interactive=False, elem_classes="mono")
            gr.Markdown("**편집**", elem_classes="section-label")
            with gr.Row():
                ds_undo_pt_btn = gr.Button("점 취소", size="sm")
                ds_undo_ln_btn = gr.Button("선 취소", size="sm")
            with gr.Row():
                ds_clear_btn = gr.Button("이 단계 초기화", size="sm")
                ds_reset_btn = gr.Button("저장본으로 되돌리기", size="sm")
            if not cmp:
                ds_nextstep_btn = gr.Button("다음 단계 (건너뛰기)", size="sm")

    DS_OUT = [ds_canvas, ds_state, ds_msg, ds_head_md, ds_step, ds_tool, ds_dirty]
    quiet = dict(show_progress="hidden", trigger_mode="multiple")  # 클릭마다 로딩 표시 없음, 빠른 연속 클릭도 모두 처리
    def init(who, request: gr.Request):
        if who:
            _WHO[request.session_hash] = who  # 브라우저에 기억해 둔 캐릭터 (localStorage)
        return ds_init(cmp, request.session_hash)
    demo.load(init, [who_radio], DS_OUT, js="(w) => [localStorage.getItem('labeler_who') || w]")
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
    return _goto(None, src["idx"], cmp=cmp, sid=src.get("sid"))

# ====================================================================================
# 전체보기 — 번호마다 원래 파손 데이터인지 정상 데이터인지, 어떤 이미지·라벨 파일이 있는지
# (구분·상태·파손 박스는 build_overview.py 가 만든 overview.json 에서 읽는다)
# ====================================================================================
try:
    with open(os.path.join(HERE, "overview.json"), encoding="utf-8") as _f:
        OVERVIEW = json.load(_f)
except (OSError, ValueError):
    OVERVIEW = {}
OV_PAGE = 50
OV_COLS = ["번호", "담당", "구분", "상태", "파손 박스", "완료", "있는 라벨"]
OV_KINDS = ["전체", "파손", "정상", "미확인"]
OV_STATES = ["전체", "완료", "미완료"]
OV_WHO = ["전체"] + PEOPLE
OV_CAPTION = {ORIGINAL: "원본", "points.json": "points.json", **{f"{n}_mask.png": f"{EDIT_SHORT[i]} 마스크" for i, n in enumerate(EDIT_FILES)}}


def _ov_files(num):
    """번호 num 의 fix/ 에 있는 파일 이름들 (Drive 모드는 시작 때 조회한 목록 + 이 서버가 저장한 것)."""
    d = os.path.join(SAVE_ROOT, str(num), "fix")
    names = set(os.listdir(d)) if os.path.isdir(d) else set()
    return names | drv.present.get(num, set()) if drv.enabled else names


def _ov_meta(num):
    return OVERVIEW.get(str(num), {"kind": "미확인"})


def _ov_done(num, idx):
    return int(num) in DONE if drv.enabled else is_complete(idx)


def ov_stats():
    """담당자별 완료율 카드 (담당 번호 중 완료된 비율)."""
    nums = {int(_stem(i)): i for i in range(N)}
    cards = []
    for who in PEOPLE:
        mine = [int(n) for n, w in ASSIGN.items() if w == who]
        done = sum(1 for n in mine if n in nums and _ov_done(n, nums[n]))
        pct = 100 * done / len(mine) if mine else 0
        cards.append(f'<div class="stat">{_person_tag(who)}'
                     f'<div class="body"><div class="name">{who}</div><div class="pct">{pct:.1f}%</div>'
                     f'<div class="sub">{done:,} / {len(mine):,}</div><div class="bar"><i style="width:{pct:.1f}%"></i></div></div></div>')
    return '<div class="stats">' + "".join(cards) + "</div>"


def ov_go(kind, state, who, page, delta=0, reset=False):
    """필터를 적용해 page(+delta) 쪽을 표로 만든다. 반환: 표, 안내 문구, 현재 쪽, 담당자별 완료율."""
    nums = {int(_stem(i)): i for i in range(N)}
    rows_n = [n for n in sorted(nums)
              if (kind == "전체" or _ov_meta(n)["kind"] == kind)
              and (who == "전체" or ASSIGN.get(str(n)) == who)
              and (state == "전체" or (_ov_done(n, nums[n]) == (state == "완료")))]
    pages = max(1, math.ceil(len(rows_n) / OV_PAGE))
    page = 1 if reset else min(max(int(page or 1) + delta, 1), pages)
    rows = []
    for n in rows_n[(page - 1) * OV_PAGE: page * OV_PAGE]:
        m, files = _ov_meta(n), _ov_files(n)
        dmg = m.get("damage") or []
        rows.append([n, ASSIGN.get(str(n), "-"), m["kind"], m.get("status") or "-",
                     f"{len(dmg)}개 ({', '.join(sorted(set(dmg)))})" if dmg else "-",
                     "완료" if _ov_done(n, nums[n]) else "미완료",
                     " ".join(EDIT_SHORT[i] for i, e in enumerate(EDIT_FILES) if f"{e}_mask.png" in files) or "-"])
    if not OVERVIEW:
        gr.Warning("overview.json 이 없어 구분·파손 박스를 알 수 없습니다 (build_overview.py 로 만들어 두세요).")
    return rows, f"조건에 맞는 번호 {len(rows_n)}개 · {page} / {pages} 쪽", page, ov_stats()


def ov_pick(evt: gr.SelectData):
    """표에서 번호를 누르면 그 번호의 설명과 들어 있는 이미지(원본·마스크)를 보여준다."""
    num = int(evt.row_value[0])
    if drv.enabled:
        drv.pull(num, max_age=120)
    m, d = _ov_meta(num), os.path.join(SAVE_ROOT, str(num), "fix")
    names = [ORIGINAL] + [f"{n}_mask.png" for n in EDIT_FILES]
    items = [(os.path.join(d, n), OV_CAPTION[n]) for n in names if os.path.exists(os.path.join(d, n))]
    dmg = m.get("damage") or []
    info = (f"### {num}번 · {m['kind']} 데이터 · 담당 {ASSIGN.get(str(num), '-')}\n"
            f"- 원본 파일: `{m.get('orig', '?')}` · AI Hub 상태: `{m.get('status') or '-'}`\n"
            f"- JSON 파손 박스: {', '.join(dmg) if dmg else '없음'}\n"
            f"- 들어 있는 파일: {', '.join(OV_CAPTION[n] for n in names + ['points.json'] if os.path.exists(os.path.join(d, n))) or '없음'}")
    return info, items


with gr.Blocks(title="점자블록 라벨링", theme=THEME, css=PAGE_CSS, js=JS_PAGE) as demo:
    _where = "Google Drive 연동" if drv.enabled else html.escape(str(SAVE_ROOT))
    who_radio = gr.Radio(PEOPLE, value=None, elem_id="who_radio", show_label=False)  # 숨김: 브라우저가 고른 캐릭터를 서버에 알림
    back_btn = gr.Button("내 담당으로", elem_id="back_btn")  # 숨김: 경고창의 '내 담당으로 돌아가기'
    gr.HTML('<div class="app-header"><span class="brand">점자블록 <b>라벨링</b></span><button class="change-who">캐릭터 선택</button>'
            + (f'<span class="meta">{N}장 · {_where}</span></div>' if N else '</div>')
            + ('' if N else f'<div class="notice">이미지를 찾을 수 없습니다: <code>{html.escape(str(SAVE_ROOT))}/&lt;번호&gt;/fix/{ORIGINAL}</code></div>'))
    with gr.Tab("라벨링") as tab_label:
        st_label, out_label = build_editor(False)
    with gr.Tab("파손 비교 (JSON)") as tab_cmp:
        st_cmp, out_cmp = build_editor(True)
    with gr.Tab("전체보기") as tab_ov:
        ov_stats_html = gr.HTML(ov_stats() if N else "")
        with gr.Row():
            ov_kind = gr.Radio(OV_KINDS, value="전체", label="구분 (원래 AI Hub 폴더)", scale=4)
            ov_who = gr.Radio(OV_WHO, value="전체", label="담당", scale=5)
            ov_state = gr.Radio(OV_STATES, value="전체", label="라벨링 상태", scale=3)
        with gr.Row(elem_classes="pager", equal_height=True):
            ov_prev = gr.Button("이전 쪽", scale=0, min_width=110)
            ov_next = gr.Button("다음 쪽", scale=0, min_width=110)
            ov_page = gr.Number(value=1, precision=0, show_label=False, container=False, scale=0, min_width=70)
            ov_info = gr.Markdown()
        ov_table = gr.Dataframe(headers=OV_COLS, value=[], interactive=False, wrap=True, label="번호를 누르면 아래에 이미지가 나옵니다")
        ov_detail = gr.Markdown("표에서 번호를 누르세요.")
        ov_gallery = gr.Gallery(label="들어 있는 이미지", columns=3, height="auto", object_fit="contain")
    OV_IN, OV_OUT = [ov_kind, ov_state, ov_who, ov_page], [ov_table, ov_info, ov_page, ov_stats_html]
    tab_ov.select(lambda k, s, w, p: ov_go(k, s, w, p), OV_IN, OV_OUT)
    for _c in (ov_kind, ov_state, ov_who):
        _c.change(lambda k, s, w, p: ov_go(k, s, w, p, reset=True), OV_IN, OV_OUT)
    ov_prev.click(lambda k, s, w, p: ov_go(k, s, w, p, -1), OV_IN, OV_OUT)
    ov_next.click(lambda k, s, w, p: ov_go(k, s, w, p, 1), OV_IN, OV_OUT)
    ov_page.submit(lambda k, s, w, p: ov_go(k, s, w, p), OV_IN, OV_OUT)
    ov_table.select(ov_pick, None, [ov_detail, ov_gallery])
    who_radio.input(pick_who, [who_radio], out_label)
    back_btn.click(go_mine, None, out_label)
    # 탭을 바꿀 때 보던 이미지와 디스크의 최신 저장 내용으로 맞춤
    tab_label.select(lambda a, b: ds_sync(a, b, False), [st_cmp, st_label], out_label)
    tab_cmp.select(lambda a, b: ds_sync(a, b, True), [st_label, st_cmp], out_cmp)

    def _on_close(request: gr.Request):
        _release(request.session_hash)
    demo.unload(_on_close)  # 창을 닫으면 '작업 중' 표시를 바로 푼다


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="점자블록 데이터셋 라벨링 도구")
    ap.add_argument("--port", type=int, default=7860)
    demo.launch(server_name="127.0.0.1", server_port=ap.parse_args().port)
