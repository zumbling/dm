# labeler_drive — Google Drive 연동 라벨링 앱

Drive 공유 폴더(`<번호>/fix/…`)를 직접 읽고 쓰는 `2_dataset/labeler.py` 변형.

## 로컬 실행
    pip install -r requirements.txt
    export LABELER_DRIVE_ID=<폴더 ID>      # 폴더 URL 끝의 문자열
    export LABELER_RANGE=1-1500            # 맡은 번호 구간 (선택)
    python labeler.py                      # 처음엔 브라우저에서 구글 로그인 (credentials.json 필요)

## 로그인 없이 배포 (Apps Script 웹앱 방식, 추천)
구글 클라우드 콘솔·credentials·token 이 필요 없다. 폴더 소유자가 `apps_script/Code.gs` 를 웹앱으로 한 번 배포한다 (절차는 파일 맨 위 주석).
서버(Render 등)에는 환경변수 두 개만 넣는다:
- `LABELER_GAS_URL` = 배포한 웹앱 URL (…/exec)
- `LABELER_GAS_KEY` = 스크립트 속성 `KEY` 와 같은 비밀 문자열
(권장) `LABELER_AUTH` = `아이디:비밀번호` — 없으면 URL 을 아는 누구나 저장할 수 있다.

## 배포 (OAuth 토큰 방식: Hugging Face Spaces 등)
1. 로컬에서 위처럼 한 번 실행해 `token.json` 을 만든다 (폴더 소유자 계정으로 로그인).
2. 이 폴더 내용을 Gradio Space 에 올린다 (`app.py` 가 진입점). `credentials.json`·`token.json`·`.env` 는 올리지 않는다.
3. Space → Settings → Secrets 에 추가:
   - `LABELER_DRIVE_ID` = 폴더 ID
   - `LABELER_TOKEN_JSON` = `token.json` 파일 내용 전체
   - (권장) `LABELER_AUTH` = `아이디:비밀번호`
4. OAuth 동의 화면을 **프로덕션으로 게시**해야 한다. '테스트' 상태면 토큰이 7일 뒤 만료된다.

## 주의
- 로그인(`LABELER_AUTH`) 없이 공개하면 URL 을 아는 누구나 소유자 Drive 에 쓸 수 있다.
- 구간 분할(`LABELER_RANGE`)은 서버 전체에 하나다. 여러 명이 동시에 쓰면 같은 번호를 열 수 있으니 번호 이동으로 각자 다른 구간에서 작업한다.
