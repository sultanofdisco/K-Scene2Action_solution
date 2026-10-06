"""Build a local, self-contained review of held-out F1-T errors."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

from f1.training.data import digest, load_manifest, write_json


def read_predictions(path, rows):
    data = json.loads(path.read_text(encoding="utf-8"))
    indexed = {r["sample_id"]: r for r in data}
    if len(indexed) != len(data) or set(indexed) != {r["sample_id"] for r in rows}:
        raise ValueError("Prediction IDs must match the entire manifest exactly")
    for row in rows:
        pred = indexed[row["sample_id"]]
        if pred["label"] != row["label"] or pred["group_id"] != row["group_id"]:
            raise ValueError("Prediction label/group mismatch")
        score = pred["uncalibrated_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid prediction score")
    return indexed


def build(manifest, predictions, text_predictions, output):
    rows = load_manifest(manifest)
    report = json.loads((predictions.parent / "report.json").read_text(encoding="utf-8"))
    text_report = json.loads((text_predictions.parent / "report.json").read_text(encoding="utf-8"))
    for result, kind in ((report, "both"), (text_report, "text")):
        if result["manifest_sha256"] != digest(manifest) or result["mode"] != "cv" or result["features"] != kind:
            raise ValueError("Expected CV reports from this manifest")
    both = read_predictions(predictions, rows)
    text = read_predictions(text_predictions, rows)
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
    errors = []
    for row in rows:
        score = both[row["sample_id"]]["uncalibrated_score"]
        if int(score >= 0.5) == row["label"]:
            continue
        other = next(r for r in pairs[row["pair_id"]] if r["sample_id"] != row["sample_id"])
        errors.append({
            **row, "error_type": "FN" if row["label"] else "FP", "score": score,
            "text_only_score": text[row["sample_id"]]["uncalibrated_score"],
            "pair_command": other["command"], "pair_label": other["label"],
            "pair_sample_id": other["sample_id"],
            "pair_score": both[other["sample_id"]]["uncalibrated_score"],
            "review": {"label_verdict": "unreviewed", "primary_cause": "unreviewed", "note": ""},
        })
    errors.sort(key=lambda r: (r["error_type"] != "FN", -abs(r["score"] - 0.5), r["sample_id"]))
    counts = Counter(r["error_type"] for r in errors)
    sources = sorted({r["source"] for r in rows})
    source_counts = [{"source": s, "samples": sum(r["source"] == s for r in rows),
                      "FP": sum(r["source"] == s and r["error_type"] == "FP" for r in errors),
                      "FN": sum(r["source"] == s and r["error_type"] == "FN" for r in errors)} for s in sources]
    meta = {"format_version": 1, "threshold": 0.5, "sample_count": len(rows), "error_count": len(errors),
            "error_counts": dict(counts), "source_counts": source_counts,
            "evaluation": "leave-one-group-out diagnostic; not independent test",
            "manifest_sha256": digest(manifest), "predictions_sha256": digest(predictions),
            "text_predictions_sha256": digest(text_predictions), "encoder_lock": report["encoder_lock"]}
    meta["review_id"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "errors.json", {"metadata": meta, "errors": errors})
    cards = []
    esc = html.escape
    for n, r in enumerate(errors, 1):
        raw = (manifest.parent / r["image_path"]).read_bytes()
        mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
        uri = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        kind = "유해 지시 누락" if r["label"] else "정상 지시 오차단"
        text_wrong = int(r["text_only_score"] >= .5) != r["label"]
        cards.append(f'''<article id="{esc(r['sample_id'])}" data-id="{esc(r['sample_id'])}" data-type="{r['error_type']}" data-source="{esc(r['source'])}">
<div class="cardhead"><b>{n:02d} <span class="badge {r['error_type']}">{kind}</span></b><code>{esc(r['sample_id'])}</code></div>
<p class="muted">{esc(r['source'])} / 검증 그룹 {esc(r['group_id'])}</p>
<img loading="lazy" src="{uri}" alt="{esc(r['sample_id'])} 원본 장면" title="클릭하면 원본 크기로 보기">
<div class="command">{esc(r['command'])}</div>
<div class="scores"><span>데이터 라벨 <b>{'유해 (1)' if r['label'] else '정상 (0)'}</b></span><span>결합 점수 <b>{r['score']:.4f}</b></span><span>텍스트 단독 <b>{r['text_only_score']:.4f}</b> ({'동일 오답' if text_wrong else '정답'})</span></div>
<details><summary>같은 이미지의 반대 라벨 지시와 기존 선별 근거</summary><p>{esc(r['pair_command'])}</p><p>라벨: {'유해' if r['pair_label'] else '정상'} / 결합 점수: {r['pair_score']:.4f} / {esc(r['pair_sample_id'])}</p><p>기존 선별 근거: {esc(r['review_reason'])}</p><p>기존 라벨은 에이전트 선별 결과이며 사람의 확정 검토 결과가 아니다.</p></details>
<div class="review"><label>라벨 검토<select data-field="label_verdict"><option value="unreviewed">미검토</option><option value="keep">기존 라벨 유지</option><option value="change_to_0">정상으로 수정 제안</option><option value="change_to_1">유해로 수정 제안</option><option value="uncertain">판단 보류</option></select></label>
<label>주요 원인 가설<select data-field="primary_cause"><option value="unreviewed">미검토</option><option value="label_issue">라벨 재검토 필요</option><option value="text_interpretation">지시 의미 판단</option><option value="visual_grounding">물체 또는 장면 연결</option><option value="image_quality">이미지 해상도 또는 시야</option><option value="threshold">임계값 영향 의심</option><option value="insufficient_context">맥락 부족</option><option value="unknown">아직 원인 불명</option><option value="other">기타</option></select></label></div>
<label>검토 메모<textarea data-field="note" placeholder="눈으로 확인한 근거, 라벨 수정 이유, 필요한 추가 사례를 적어주세요."></textarea></label></article>''')
    payload = json.dumps({"metadata": meta, "errors": errors}, ensure_ascii=False).replace("<", "\\u003c")
    page = TEMPLATE.replace("__TOTAL__", str(len(errors))).replace("__FP__", str(counts['FP'])).replace("__FN__", str(counts['FN']))
    page = page.replace("__SAMPLES__", str(len(rows)))
    page = page.replace("__CARDS__", "\n".join(cards)).replace("__PAYLOAD__", payload)
    page = page.replace("__SOURCES__", "".join(f'<option value="{esc(s)}">{esc(s.split("/")[-1])}</option>' for s in sources))
    (output / "error-review.html").write_text(page, encoding="utf-8")
    lines = ["# F1-T 오답 검토표", "", "[이미지와 검토 입력란이 있는 검토표 열기](error-review.html)", "",
             f"전체 {len(rows)}개 중 오답 {len(errors)}개: 정상 오차단 {counts['FP']}개, 유해 누락 {counts['FN']}개.", "",
             "4개 그룹을 하나씩 제외해 평가한 예측을 사용했다. 임계값은 임시 기준 0.5이며 점수는 보정된 확률이 아니다. 독립 시험 결과나 운영 판정으로 해석하지 않는다.", "",
             "## 검토 방법", "", "1. 유해 누락부터 확인하고, 다음으로 정상 오차단을 확인한다.",
             "2. 원본 이미지와 지시를 보고 데이터 라벨이 타당한지 검토한다. 이미지를 클릭하면 확대할 수 있다.",
             "3. 반대 라벨 지시와 텍스트 단독 점수를 참고하고, 원인 가설과 관찰한 근거를 적는다. 점수 비교만으로 원인을 확정하지 않는다.",
             "4. 화면 상단의 검토 결과 JSON 내보내기로 의견을 저장한다. 브라우저 임시 저장은 보조 기능이며 창을 닫기 전에 파일로 내보낸다.",
             "5. 라벨 수정은 제안만 기록한다. 원본 데이터와 모델에는 자동 반영되지 않는다.", "",
             "원인 가설과 검토 메모는 비워두었다. 데이터를 검토한 뒤 재학습에 반영하면 이 검토표는 개발용 자료가 되며, 최종 성능은 새로운 독립 시험 데이터에서 확인해야 한다.", "",
             "## 출처별 집계", "", "| 출처 | 전체 | 정상 오차단 | 유해 누락 |", "|---|---:|---:|---:|"]
    lines += [f"| {r['source'].split('/')[-1]} | {r['samples']} | {r['FP']} | {r['FN']} |" for r in source_counts]
    lines += ["", "## 전체 오답 목록", "", "| ID | 유형 | 데이터 라벨 | 결합 점수 | 텍스트 점수 | 지시 |", "|---|---|---|---:|---:|---|"]
    for r in errors:
        cmd = r['command'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| [{r['sample_id']}](error-review.html#{r['sample_id']}) | {r['error_type']} | {r['label']} | {r['score']:.4f} | {r['text_only_score']:.4f} | {cmd} |")
    lines += ["", "FP는 정상 오차단, FN은 유해 누락이다. HTML은 원본 이미지를 포함하며 별도 서버 없이 브라우저에서 열 수 있다.", "",
              "`errors.json`에는 원문, 이미지 경로와 해시, 검증 점수, 비어 있는 검토 필드 및 입력 파일 해시가 있다. 화면에서 작성한 의견은 내보낸 JSON에 저장된다."]
    (output / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return meta


TEMPLATE = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1-T 오답 검토표</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f9;color:#182334;font:15px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px}h1{font-size:32px;margin:4px 0}h2{font-size:18px}.muted{color:#58667a}.intro{background:#102c45;color:white;padding:28px;border-radius:16px}.intro p{max-width:900px}.stats{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}.stats span{background:#fff;padding:12px 20px;border:1px solid #d9e0e9;border-radius:10px}.stats b{font-size:24px;margin-right:8px}.toolbar{position:sticky;top:0;z-index:2;background:#f2f5f9f5;padding:14px 0;border-bottom:1px solid #cad5e0;display:flex;gap:8px;flex-wrap:wrap}select,input,button,textarea{font:inherit;border:1px solid #bdcad9;border-radius:7px;padding:8px;background:white;color:#182334}button{cursor:pointer}button.primary{background:#174d77;color:white}#search{flex:1;min-width:200px}#status{min-height:25px;color:#385573}article{background:white;border:1px solid #dce4ed;border-radius:14px;padding:24px;margin:20px 0;scroll-margin-top:150px}article[hidden]{display:none}.cardhead,.scores{display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap}.badge{border-radius:6px;padding:4px 10px;margin-left:8px;font-size:14px}.FN{background:#ffe1df;color:#9f2720}.FP{background:#fff0ca;color:#775105}img{display:block;max-width:100%;max-height:460px;object-fit:contain;margin:18px auto;cursor:zoom-in;background:#edf1f6}.command{font:500 18px/1.65 system-ui;background:#f3f6fa;padding:18px;border-left:4px solid #397daf;white-space:pre-wrap;overflow-wrap:anywhere}.scores{margin:16px 0}.review{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}label{display:block;font-weight:600}label select{display:block;width:100%;font-weight:400;margin:5px 0 12px}textarea{display:block;width:100%;min-height:80px;resize:vertical;font-weight:400;margin-top:5px}details{border-top:1px solid #e0e6ed;padding-top:12px}summary{cursor:pointer;color:#235f8a}details p{overflow-wrap:anywhere}dialog{max-width:96vw;max-height:96vh;border:0;border-radius:10px;padding:16px}dialog img{max-width:90vw;max-height:82vh}dialog::backdrop{background:#000b}@media(max-width:650px){main{padding:16px 10px}article{padding:16px}.review{grid-template-columns:1fr}.cardhead code{font-size:12px}}</style></head><body><main>
<header class="intro"><div>F1-T / CPU 초기 실험 / RoboJailBench</div><h1>오답 검토표</h1><p>원본 장면과 지시를 함께 보며 라벨과 실패 원인을 검토합니다. 4개 그룹을 하나씩 제외한 검증 결과이며, 실제 운영 판정이 아닙니다. 기존 라벨도 검토 대상입니다.</p></header>
<div class="stats"><span><b>__SAMPLES__</b>전체 샘플</span><span><b>__TOTAL__</b>오답</span><span><b>__FN__</b>유해 누락</span><span><b>__FP__</b>정상 오차단</span></div>
<p class="muted">임시 기준: 점수 ≥ 0.5면 유해로 분류. 점수는 보정된 확률이 아닙니다. 유해 누락을 먼저, 각 유형 안에서는 기준에서 멀리 벗어난 오답부터 표시합니다. 아래 원인 항목은 검토자의 가설이며 모델의 설명이 아닙니다.</p>
<div class="toolbar"><select id="kind" aria-label="오답 유형"><option value="">모든 오답</option value="FN">유해 누락</option><option value="FP">정상 오차단</option></select><select id="source" aria-label="데이터 출처"><option value="">모든 출처</option>__SOURCES__</select><select id="reviewed" aria-label="검토 상태"><option value="">모든 검토 상태</option><option value="no">라벨 미검토</option><option value="yes">라벨 검토 기록 있음</option></select><input id="search" placeholder="ID 또는 지시 검색" aria-label="ID 또는 지시 검색"><button class="primary" id="export">검토 결과 JSON 내보내기</button><button id="import">검토 JSON 불러오기</button><input type="file" accept="application/json,.json" id="file" hidden></div>
<p id="status" role="status"></p><p class="muted">의견은 브라우저에 임시 저장합니다. 검토를 마치면 JSON으로 내보내 주세요. 라벨 수정 제안은 원본 데이터나 모델에 자동 반영되지 않습니다. 저장된 JSON을 불러오면 현재 작성한 의견을 대체합니다.</p>
<section id="cards">__CARDS__</section><dialog id="zoom"><button id="closezoom">닫기</button><img alt="원본 장면 확대"></dialog>
</main><script type="application/json" id="payload">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent), key='f1t-review-'+data.metadata.review_id;
const cards=[...document.querySelectorAll('article')], rows=new Map(data.errors.map(r=>[r.sample_id,r]));
const status=document.getElementById('status');let storageOK=true;
function applyReviews(items){if(!Array.isArray(items)||items.length!==rows.size||new Set(items.map(r=>r.sample_id)).size!==rows.size)throw Error('검토 ID 목록이 일치하지 않습니다.');
const pending=[];for(const r of items){if(!rows.has(r.sample_id)||!r.review)throw Error('알 수 없는 검토 항목입니다.');const card=document.getElementById(r.sample_id),v=r.review;
for(const f of ['label_verdict','primary_cause']){if(![...card.querySelector('[data-field="'+f+'"]').options].some(o=>o.value===v[f]))throw Error('검토 선택값이 올바르지 않습니다.');}if(typeof v.note!=='string')throw Error('메모 형식이 올바르지 않습니다.');pending.push([r.sample_id,{label_verdict:v.label_verdict,primary_cause:v.primary_cause,note:v.note}]);}
for(const [id,v] of pending){rows.get(id).review=v;for(const el of document.getElementById(id).querySelectorAll('[data-field]'))el.value=v[el.dataset.field];}}
function persist(){try{localStorage.setItem(key,JSON.stringify(data.errors.map(r=>({sample_id:r.sample_id,review:r.review}))));}catch{storageOK=false;}}
function filter(){const kind=document.getElementById('kind').value,source=document.getElementById('source').value,reviewed=document.getElementById('reviewed').value,q=document.getElementById('search').value.toLowerCase();let shown=0,done=0;
for(const card of cards){const r=rows.get(card.dataset.id),checked=r.review.label_verdict!=='unreviewed';if(checked)done++;const show=(!kind||r.error_type===kind)&&(!source||r.source===source)&&(!reviewed||(reviewed==='yes')===checked)&&(!q||(r.sample_id+' '+r.command).toLowerCase().includes(q));card.hidden=!show;if(show)shown++;}status.textContent=`표시 ${shown} / ${cards.length}개 | 라벨 검토 기록 ${done}개`+(storageOK?' | 창을 닫기 전에 JSON으로 내보내 주세요.':' | 임시 저장 불가: 반드시 JSON으로 내보내 주세요.');}
try{const saved=localStorage.getItem(key);if(saved)applyReviews(JSON.parse(saved));}catch{storageOK=false;}
for(const card of cards){for(const el of card.querySelectorAll('[data-field]'))el.addEventListener('input',()=>{rows.get(card.dataset.id).review[el.dataset.field]=el.value;persist();filter();});card.querySelector('img').addEventListener('click',()=>{document.querySelector('#zoom img').src=card.querySelector('img').src;document.getElementById('zoom').showModal();});}
for(const id of ['kind','source','reviewed','search'])document.getElementById(id).addEventListener('input',filter);
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({...data,exported_at:new Date().toISOString()},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='f1t-error-review-notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.getElementById('import').onclick=()=>document.getElementById('file').click();document.getElementById('file').onchange=async e=>{try{if(!e.target.files.length)return;const v=JSON.parse(await e.target.files[0].text());if(v.metadata?.review_id!==data.metadata.review_id)throw Error('다른 실험의 검토 파일입니다.');applyReviews(v.errors);persist();filter();}catch(err){status.textContent='불러오기 실패: '+err.message;}finally{e.target.value='';}};
document.getElementById('closezoom').onclick=()=>document.getElementById('zoom').close();filter();
</script></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/robojailbench_f1t_english_v1/selected.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/f1t/cv-both-cpu-v1.2/predictions.json"))
    parser.add_argument("--text-predictions", type=Path, default=Path("artifacts/f1t/cv-text-cpu-v1.2/predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/f1t/error-review-cpu-v1.2"))
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.predictions, args.text_predictions, args.output), ensure_ascii=False, indent=2))
