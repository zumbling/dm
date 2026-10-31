"""labeler.py 가 Google Drive 의 공유 폴더를 직접 읽고 쓰게 하는 모듈 (Drive for Desktop 불필요).

공유 폴더 구조:  <번호>/fix/original.jpeg, *_mask.png, points.json
환경변수:
  LABELER_DRIVE_ID   공유 폴더 ID (폴더 URL 끝의 문자열). 이게 있어야 Drive 모드가 켜진다.
  LABELER_CREDS      OAuth 클라이언트 파일 (기본 credentials.json)
  LABELER_CACHE      내려받은 파일을 두는 임시 폴더 (기본 .drive_cache)
  LABELER_RANGE      맡은 번호 구간, 예: 1-1500 (기본: 전체)
처음 실행하면 브라우저에서 구글 로그인 창이 한 번 뜬다 (token.json 에 저장).
배포(서버)에서는 로그인 창을 띄울 수 없으므로 token.json 의 내용을 환경변수 LABELER_TOKEN_JSON 에 넣는다.
  -> 서버가 폴더 소유자 계정으로 읽고 쓴다 (파일 소유자가 소유자 본인이라 개인 드라이브 용량 문제도 없다).

로그인 없이 쓰기 (구글 로그인·credentials·token 불필요):
  폴더 소유자가 apps_script/Code.gs 를 웹앱으로 한 번 배포하고, 서버에는 아래 두 값만 넣는다.
  LABELER_GAS_URL   배포한 웹앱 URL (…/exec)
  LABELER_GAS_KEY   Code.gs 의 KEY 스크립트 속성과 같은 비밀 문자열
  -> 이 모드에서는 LABELER_DRIVE_ID 도 필요 없다 (폴더 ID 는 웹앱 쪽에 있다).
"""
import base64
import io
import json
import os
import urllib.request
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_ID = os.environ.get("LABELER_DRIVE_ID")
CREDS = os.environ.get("LABELER_CREDS", os.path.join(HERE, "credentials.json"))
TOKEN = os.path.join(HERE, "token.json")
CACHE = os.environ.get("LABELER_CACHE", os.path.join(HERE, ".drive_cache"))
RANGE = os.environ.get("LABELER_RANGE", "")
GAS_URL = os.environ.get("LABELER_GAS_URL")
GAS_KEY = os.environ.get("LABELER_GAS_KEY", "")
enabled = bool(ROOT_ID or GAS_URL)

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER = "application/vnd.google-apps.folder"
_local = threading.local()
_creds = None
_pool = ThreadPoolExecutor(max_workers=1)   # 저장 업로드를 화면과 따로 처리
errors = []                                 # 업로드 실패 메시지
fix_ids = {}                                # 번호 -> fix 폴더 ID
_num_locks = {}                             # 번호 -> 잠금: 저장과 내려받기가 같은 번호 파일을 동시에 쓰지 않게
_guard = threading.Lock()
_pending = {}                               # 번호 -> 아직 Drive 에 올리지 못한 저장 수
_saved = {}                                 # 번호 -> 마지막으로 로컬에 저장한 시각


def lock(num):
    """번호별 잠금. 저장(labeler._save_fix)과 내려받기(_pull)가 함께 쓴다."""
    with _guard:
        return _num_locks.setdefault(num, threading.Lock())


def write_file(p, data):
    """반쯤 쓰인 파일이 읽히지 않게 임시 파일에 다 쓴 뒤 바꿔치기한다."""
    with open(p + ".part", "wb") as fh:
        fh.write(data)
    os.replace(p + ".part", p)


def _credentials():
    global _creds
    if _creds is None:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        env = os.environ.get("LABELER_TOKEN_JSON")
        if env:  # 서버 배포: 환경변수의 토큰을 쓰고 만료되면 갱신만 한다 (파일·브라우저 불필요)
            c = Credentials.from_authorized_user_info(json.loads(env), SCOPES)
            if not c.valid:
                c.refresh(Request())
            _creds = c
            return _creds
        c = Credentials.from_authorized_user_file(TOKEN, SCOPES) if os.path.exists(TOKEN) else None
        if not c or not c.valid:
            if c and c.expired and c.refresh_token:
                c.refresh(Request())
            else:
                c = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES).run_local_server(port=0)
            with open(TOKEN, "w") as f:
                f.write(c.to_json())
        _creds = c
    return _creds


def _svc():
    """googleapiclient 는 스레드마다 따로 만들어야 안전하다."""
    if not hasattr(_local, "svc"):
        from googleapiclient.discovery import build
        _local.svc = build("drive", "v3", credentials=_credentials(), cache_discovery=False)
    return _local.svc


def _list(q, fields):
    out, tok = [], None
    while True:
        r = _svc().files().list(q=q, fields=f"nextPageToken, files({fields})", pageSize=1000, pageToken=tok,
                                supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        out += r["files"]
        tok = r.get("nextPageToken")
        if not tok:
            return out


def _gas(action, **kw):
    """Apps Script 웹앱 호출 (POST 는 302 로 결과 주소에 넘어가고 urllib 이 따라간다)."""
    req = urllib.request.Request(GAS_URL, data=json.dumps({"key": GAS_KEY, "action": action, **kw}).encode(),
                                 headers={"Content-Type": "text/plain"})
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.loads(r.read().decode())
    if res.get("error"):
        raise RuntimeError(res["error"])
    return res


def _cleanup(dst, remote):
    """다른 곳에서 지워진 마스크가 캐시에 남아 있지 않게 정리"""
    for n in os.listdir(dst):
        if n.endswith(("_mask.png", ".json")) and n not in remote:
            os.remove(os.path.join(dst, n))


def _in_range(n):
    if not RANGE:
        return True
    a, _, b = RANGE.partition("-")
    return int(a or 0) <= n <= int(b or 10 ** 9)


present = {}                                 # 번호 -> fix/ 에 있는 파일 이름들 (index 때 조회한 이름만, 전체보기 탭용)


def index(required, extra=(), any_of=()):
    """Drive 를 몇 번만 조회해 (이미지가 있는 번호 목록, 완료된 번호 집합) 을 만든다.
    required: 완료로 보려면 fix/ 에 모두 있어야 하는 파일 이름들. any_of: 그중 하나라도 있어야 하는 파일 이름들. extra: 전체보기 탭에 보여줄 추가 파일 이름들 (present 에 기록)."""
    if GAS_URL:
        res = _gas("index", required=list(required), anyOf=list(any_of))
        return sorted(n for n in res["nums"] if _in_range(n)), set(res["done"])
    num_of = {f["id"]: int(f["name"]) for f in
              _list(f"'{ROOT_ID}' in parents and mimeType='{FOLDER}' and trashed=false", "id,name")
              if f["name"].isdigit()}
    fix_ids.clear()
    for f in _list(f"name='fix' and mimeType='{FOLDER}' and trashed=false", "id,parents"):
        n = num_of.get((f.get("parents") or [None])[0])
        if n is not None:
            fix_ids.setdefault(n, f["id"])
    num_of_fix = {fid: n for n, fid in fix_ids.items()}

    have = {num_of_fix[p] for f in _list("name='original.jpeg' and trashed=false", "parents")
            for p in f.get("parents", []) if p in num_of_fix}
    cnt = {}
    names = " or ".join(f"name='{n}'" for n in dict.fromkeys([*required, *any_of, *extra]))
    for f in _list(f"({names}) and trashed=false", "name,parents"):
        for p in f.get("parents", []):
            if p in num_of_fix:
                cnt.setdefault(num_of_fix[p], set()).add(f["name"])
    done = {n for n, s in cnt.items() if set(required) <= s and (not any_of or any(a in s for a in any_of))}
    present.clear()
    for n in have:
        present[n] = {"original.jpeg"} | cnt.get(n, set())
    return sorted(n for n in have if _in_range(n)), done


_pull_lock = threading.Lock()
_pulled = {}                                   # 번호 -> 마지막으로 받은 시각
_prefetch_pool = ThreadPoolExecutor(max_workers=1)


def pull(num, max_age=0):
    """<번호>/fix/ 를 받는다. 한 번에 하나씩만 받고(두 탭이 같은 임시 파일을 덮어쓰지 않게),
    max_age 초 안에 이미 받았으면(미리 받아 둔 경우) 건너뛴다 — 방금 저장한 마스크를 되돌려 지우는 것도 막는다."""
    with _pull_lock:
        if max_age and time.time() - _pulled.get(num, 0) < max_age:
            return
        _pull(num)
        _pulled[num] = time.time()


def prefetch(*nums):
    """보고 있는 동안 다음 이미지들을 백그라운드로 미리 받아 둔다 (실패해도 조용히 넘어간다)."""
    def run(n):
        try:
            pull(n, max_age=120)
        except Exception:
            pass
    for n in nums:
        _prefetch_pool.submit(run, n)


def _pull(num):
    """<번호>/fix/ 를 CACHE 로 내려받는다. 원본 이미지는 이미 받았으면 건너뛰고, 마스크·json 은 항상 최신으로.
    받는 동안 이 번호를 저장했거나 아직 업로드 중이면 마스크·json 은 쓰지 않는다 (Drive 의 옛 파일이 방금 저장한 걸 덮지 않게)."""
    dst = os.path.join(CACHE, str(num), "fix")
    os.makedirs(dst, exist_ok=True)
    started = time.time()
    if GAS_URL:
        res = _gas("pull", num=num, orig=not os.path.exists(os.path.join(dst, "original.jpeg")))
        got, remote = [(f["name"], base64.b64decode(f["b64"])) for f in res["files"]], set(res["names"])
    else:
        from googleapiclient.http import MediaIoBaseDownload
        fid = fix_ids.get(num)
        if not fid:
            return
        got, remote = [], set()
        for f in _list(f"'{fid}' in parents and trashed=false", "id,name,size,mimeType"):
            if f["mimeType"] == FOLDER:
                continue
            remote.add(f["name"])
            p = os.path.join(dst, f["name"])
            if f["name"] == "original.jpeg" and os.path.exists(p) and os.path.getsize(p) == int(f.get("size", -1)):
                continue
            buf = io.BytesIO()
            dl, ok = MediaIoBaseDownload(buf, _svc().files().get_media(fileId=f["id"], supportsAllDrives=True)), False
            while not ok:
                _, ok = dl.next_chunk()
            got.append((f["name"], buf.getvalue()))
    with lock(num):
        local_newer = _pending.get(num) or _saved.get(num, 0) >= started
        for name, data in got:
            if name == "original.jpeg" or not local_newer:
                write_file(os.path.join(dst, name), data)
        if not local_newer:
            _cleanup(dst, remote)  # 다른 곳에서 지워진 마스크가 캐시에 남아 있지 않게 정리


def _push(num, names, removed):
    try:
        if GAS_URL:
            files = [{"name": n, "b64": base64.b64encode(open(os.path.join(CACHE, str(num), "fix", n), "rb").read()).decode()}
                     for n in names]
            _gas("push", num=num, files=files, removed=removed)
            return
        from googleapiclient.http import MediaFileUpload
        fid = fix_ids.get(num)
        if not fid:
            parent = next((f["id"] for f in _list(f"'{ROOT_ID}' in parents and name='{num}' and mimeType='{FOLDER}' and trashed=false", "id")), None)
            parent = parent or _svc().files().create(body={"name": str(num), "parents": [ROOT_ID], "mimeType": FOLDER},
                                                     fields="id", supportsAllDrives=True).execute()["id"]
            fid = _svc().files().create(body={"name": "fix", "parents": [parent], "mimeType": FOLDER},
                                        fields="id", supportsAllDrives=True).execute()["id"]
            fix_ids[num] = fid
        exist = {f["name"]: f["id"] for f in _list(f"'{fid}' in parents and trashed=false", "id,name")}
        for n in names:
            media = MediaFileUpload(os.path.join(CACHE, str(num), "fix", n), resumable=False)
            if n in exist:
                _svc().files().update(fileId=exist[n], media_body=media, supportsAllDrives=True).execute()
            else:
                _svc().files().create(body={"name": n, "parents": [fid]}, media_body=media,
                                      supportsAllDrives=True).execute()
        for n in removed:
            if n in exist:
                _svc().files().delete(fileId=exist[n], supportsAllDrives=True).execute()
    except Exception as e:  # 백그라운드라서 화면에 직접 못 띄우므로 모아 둔다
        errors.append(f"{num}번 업로드 실패: {e}")
    finally:
        with _guard:
            _pending[num] -= 1


def push_async(num, names, removed=()):
    """저장한 파일들을 백그라운드로 Drive 에 올린다 (names: 올릴 파일, removed: Drive 에서 지울 파일).
    올리기를 마칠 때까지 이 번호는 Drive 에서 다시 받지 않는다."""
    with _guard:
        _pending[num] = _pending.get(num, 0) + 1
        _saved[num] = time.time()
    _pool.submit(_push, num, list(names), list(removed))
