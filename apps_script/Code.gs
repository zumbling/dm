/**
 * labeler 가 로그인 없이 Drive 폴더를 읽고 쓰게 하는 웹앱 (폴더 소유자 권한으로 실행된다).
 *
 * 설정:
 *  1) script.google.com 에서 새 프로젝트를 만들고 이 코드를 붙여넣는다.
 *  2) 프로젝트 설정 → 스크립트 속성에 추가:  ROOT_ID = 공유 폴더 ID,  KEY = 긴 임의 문자열
 *  3) appsscript.json 에 "oauthScopes": ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/script.external_request"] 추가
 *     (프로젝트 설정 → 'appsscript.json 매니페스트 파일 표시' 체크)
 *  4) 배포 → 새 배포 → 유형 '웹 앱' → 실행 사용자 '나', 액세스 권한 '모든 사용자' → 배포
 *  5) 나온 URL(…/exec)을 서버의 LABELER_GAS_URL, KEY 값을 LABELER_GAS_KEY 로 넣는다.
 * 코드를 고치면 '배포 관리'에서 새 버전으로 다시 배포해야 반영된다.
 */
var P = PropertiesService.getScriptProperties();
var FOLDER = "application/vnd.google-apps.folder";

function doPost(e) {
  try {
    var r = JSON.parse(e.postData.contents);
    if (!P.getProperty("KEY") || r.key !== P.getProperty("KEY")) return out_({ error: "unauthorized" });
    return out_(ACTIONS[r.action](r));
  } catch (err) {
    return out_({ error: String(err) });
  }
}

function out_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}

// Drive REST 조회 (DriveApp 으로 4천 개 폴더를 도는 것보다 훨씬 빠르다)
function list_(q, fields) {
  var out = [], tok = "";
  do {
    var url = "https://www.googleapis.com/drive/v3/files?pageSize=1000&supportsAllDrives=true&includeItemsFromAllDrives=true" +
      "&q=" + encodeURIComponent(q) + "&fields=" + encodeURIComponent("nextPageToken,files(" + fields + ")") +
      (tok ? "&pageToken=" + tok : "");
    var res = JSON.parse(UrlFetchApp.fetch(url, { headers: { Authorization: "Bearer " + ScriptApp.getOAuthToken() } }).getContentText());
    out = out.concat(res.files);
    tok = res.nextPageToken;
  } while (tok);
  return out;
}

function fixFolder_(num, create) {
  var root = DriveApp.getFolderById(P.getProperty("ROOT_ID"));
  var it = root.getFoldersByName(String(num));
  var d = it.hasNext() ? it.next() : (create ? root.createFolder(String(num)) : null);
  if (!d) return null;
  var f = d.getFoldersByName("fix");
  return f.hasNext() ? f.next() : (create ? d.createFolder("fix") : null);
}

var ACTIONS = {
  // 이미지가 있는 번호와 완료된 번호 (fix/ 에 required 파일이 모두 있는 번호)
  index: function (r) {
    var numOf = {};
    list_("'" + P.getProperty("ROOT_ID") + "' in parents and mimeType='" + FOLDER + "' and trashed=false", "id,name")
      .forEach(function (f) { if (/^\d+$/.test(f.name)) numOf[f.id] = parseInt(f.name, 10); });
    var fixNum = {};
    list_("name='fix' and mimeType='" + FOLDER + "' and trashed=false", "id,parents").forEach(function (f) {
      var n = numOf[(f.parents || [])[0]];
      if (n !== undefined && fixNum[f.id] === undefined) fixNum[f.id] = n;
    });
    var have = {}, cnt = {};
    list_("name='original.jpeg' and trashed=false", "parents").forEach(function (f) {
      (f.parents || []).forEach(function (p) { if (fixNum[p] !== undefined) have[fixNum[p]] = true; });
    });
    var q = "(" + r.required.map(function (n) { return "name='" + n + "'"; }).join(" or ") + ") and trashed=false";
    list_(q, "name,parents").forEach(function (f) {
      (f.parents || []).forEach(function (p) {
        if (fixNum[p] !== undefined) { cnt[fixNum[p]] = cnt[fixNum[p]] || {}; cnt[fixNum[p]][f.name] = 1; }
      });
    });
    var done = Object.keys(cnt).filter(function (n) { return Object.keys(cnt[n]).length === r.required.length; }).map(Number);
    return { nums: Object.keys(have).map(Number), done: done };
  },

  // <번호>/fix/ 의 파일들 (r.orig 가 false 면 원본 이미지는 이름만 알리고 내용은 생략)
  pull: function (r) {
    var fix = fixFolder_(r.num, false), files = [], names = [];
    if (fix) {
      var it = fix.getFiles();
      while (it.hasNext()) {
        var f = it.next(), n = f.getName();
        names.push(n);
        if (n === "original.jpeg" && !r.orig) continue;
        files.push({ name: n, b64: Utilities.base64Encode(f.getBlob().getBytes()) });
      }
    }
    return { files: files, names: names };
  },

  // 저장한 파일 올리기 (같은 이름은 휴지통으로 보내고 새로 만든다), removed 는 삭제
  push: function (r) {
    var fix = fixFolder_(r.num, true);
    var lock = LockService.getScriptLock();
    lock.waitLock(30000);
    try {
      r.files.forEach(function (f) {
        var old = fix.getFilesByName(f.name);
        while (old.hasNext()) old.next().setTrashed(true);
        var mime = f.name.indexOf(".png") > 0 ? "image/png" : "application/json";
        fix.createFile(Utilities.newBlob(Utilities.base64Decode(f.b64), mime, f.name));
      });
      (r.removed || []).forEach(function (n) {
        var old = fix.getFilesByName(n);
        while (old.hasNext()) old.next().setTrashed(true);
      });
    } finally {
      lock.releaseLock();
    }
    return { ok: true };
  },
};
