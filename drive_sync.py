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
"""
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_ID = os.environ.get("LABELER_DRIVE_ID")
CREDS = os.environ.get("LABELER_CREDS", os.path.join(HERE, "credentials.json"))
TOKEN = os.path.join(HERE, "token.json")
CACHE = os.environ.get("LABELER_CACHE", os.path.join(HERE, ".drive_cache"))
RANGE = os.environ.get("LABELER_RANGE", "")
enabled = bool(ROOT_ID)

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER = "application/vnd.google-apps.folder"
_local = threading.local()
_creds = None
_pool = ThreadPoolExecutor(max_workers=1)   # 저장 업로드를 화면과 따로 처리
errors = []                                 # 업로드 실패 메시지
fix_ids = {}                                # 번호 -> fix 폴더 ID


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


def _in_range(n):
    if not RANGE:
        return True
    a, _, b = RANGE.partition("-")
    return int(a or 0) <= n <= int(b or 10 ** 9)


def index(required):
    """Drive 를 몇 번만 조회해 (이미지가 있는 번호 목록, 완료된 번호 집합) 을 만든다.
    required: 완료로 보려면 fix/ 에 있어야 하는 파일 이름들."""
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
    names = " or ".join(f"name='{n}'" for n in required)
    for f in _list(f"({names}) and trashed=false", "name,parents"):
        for p in f.get("parents", []):
            if p in num_of_fix:
                cnt.setdefault(num_of_fix[p], set()).add(f["name"])
    done = {n for n, s in cnt.items() if len(s) == len(required)}
    return sorted(n for n in have if _in_range(n)), done


def pull(num):
    """<번호>/fix/ 를 CACHE 로 내려받는다. 원본 이미지는 이미 받았으면 건너뛰고, 마스크·json 은 항상 최신으로."""
    from googleapiclient.http import MediaIoBaseDownload
    dst = os.path.join(CACHE, str(num), "fix")
    os.makedirs(dst, exist_ok=True)
    fid = fix_ids.get(num)
    if not fid:
        return
    for f in _list(f"'{fid}' in parents and trashed=false", "id,name,size,mimeType"):
        if f["mimeType"] == FOLDER:
            continue
        p = os.path.join(dst, f["name"])
        if f["name"] == "original.jpeg" and os.path.exists(p) and os.path.getsize(p) == int(f.get("size", -1)):
            continue
        req = _svc().files().get_media(fileId=f["id"], supportsAllDrives=True)
        with io.FileIO(p + ".part", "wb") as fh:
            dl, ok = MediaIoBaseDownload(fh, req), False
            while not ok:
                _, ok = dl.next_chunk()
        os.replace(p + ".part", p)
    # 다른 곳에서 지워진 마스크가 캐시에 남아 있지 않게 정리
    remote = {f["name"] for f in _list(f"'{fid}' in parents and trashed=false", "name")}
    for n in os.listdir(dst):
        if n.endswith(("_mask.png", ".json")) and n not in remote:
            os.remove(os.path.join(dst, n))


def _push(num, names, removed):
    from googleapiclient.http import MediaFileUpload
    try:
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


def push_async(num, names, removed=()):
    """저장한 파일들을 백그라운드로 Drive 에 올린다 (names: 올릴 파일, removed: Drive 에서 지울 파일)."""
    _pool.submit(_push, num, list(names), list(removed))
