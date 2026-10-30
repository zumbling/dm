"""배포용 진입점 (Hugging Face Spaces 등). 로컬은 `python labeler.py` 를 쓴다.

필요한 환경변수(Secrets):
  LABELER_DRIVE_ID    공유 폴더 ID
  LABELER_TOKEN_JSON  로컬에서 한 번 로그인해 만든 token.json 의 내용
선택:
  LABELER_AUTH        "아이디:비밀번호" — 있으면 접속에 로그인 요구
"""
import os

from labeler import demo

auth = tuple(os.environ["LABELER_AUTH"].split(":", 1)) if os.environ.get("LABELER_AUTH") else None
demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), auth=auth)
