"""Build the one-file review page for the two human assessors from the blinded sheet.

    python -m evaluation.build_review_page --sheet human_review_sheet.csv --out review_tool.html

The page shows each reply with its ticket and the source article, and records, per reply: the number of unsupported
claims, which claims, whether the citations are correct, and optional notes. It shows no labels, no validator verdicts
and no group information, so the assessment stays blind. Answers are kept in the assessor's browser as they type and
leave as CSV text to copy (the viewer blocks file downloads).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

TEMPLATE = r"""<title>Reply Review Sheet</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: working tool. A fixed list of replies on the left, one reply and its evidence on the right, answers pinned at the end. */
:root{
  --bg:#f6f7f8; --panel:#ffffff; --ink:#1c2429; --muted:#5d6a72; --line:#d9dee2;
  --accent:#0b6e76; --accent-ink:#ffffff; --accent-soft:#dff0f1;
  --ok:#2e7d4f; --ok-soft:#e1f1e7; --warn:#a55b00; --warn-soft:#fbecd5; --bad:#b3372f;
  --cite:#e8eef7; --cite-ink:#264a80;
  --sans:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif; --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#13191d; --panel:#1b2329; --ink:#e4e9ec; --muted:#9aa8b1; --line:#2f3b43;
  --accent:#4cc3cb; --accent-ink:#0c1a1c; --accent-soft:#17383b;
  --ok:#6cc58e; --ok-soft:#173326; --warn:#e9a652; --warn-soft:#3a2a10; --bad:#ef7f77;
  --cite:#1d2c45; --cite-ink:#9bbcf0; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#13191d; --panel:#1b2329; --ink:#e4e9ec; --muted:#9aa8b1; --line:#2f3b43;
  --accent:#4cc3cb; --accent-ink:#0c1a1c; --accent-soft:#17383b;
  --ok:#6cc58e; --ok-soft:#173326; --warn:#e9a652; --warn-soft:#3a2a10; --bad:#ef7f77;
  --cite:#1d2c45; --cite-ink:#9bbcf0; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans);margin:0;
  padding-block:16px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font-size:1.25rem;margin:0 0 4px;font-weight:600;text-wrap:balance}
h2{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;font-weight:600}
.wrap{max-width:1180px;margin:0 auto}
.top{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:flex-end;justify-content:space-between;margin-bottom:12px}
.top p{margin:0;color:var(--muted);max-width:62ch}
.who{display:flex;gap:8px;align-items:center}
label{font-size:.85rem;color:var(--muted)}
input[type=text],input[type=number],textarea{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:7px 9px;width:100%}
input[type=text]#who{width:7rem}
input[type=number]{width:6rem}
textarea{min-height:5.5rem;resize:vertical}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
details.how{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin-bottom:14px}
details.how summary{cursor:pointer;font-weight:600}
details.how ul{margin:8px 0 0;padding-left:1.1rem;max-width:80ch}
.layout{display:grid;grid-template-columns:200px minmax(0,1fr);gap:16px;align-items:start}
.list{position:sticky;top:8px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;max-height:calc(100vh - 24px);overflow:auto}
.progress{font-family:var(--mono);font-size:.82rem;margin-bottom:8px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:4px 0 10px}
.bar i{display:block;height:100%;background:var(--accent);width:0}
.items{display:grid;grid-template-columns:repeat(auto-fill,minmax(44px,1fr));gap:6px}
.items button{font:500 .78rem var(--mono);border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:5px;padding:5px 0;cursor:pointer}
.items button.done{background:var(--ok-soft);border-color:var(--ok);color:var(--ok)}
.items button.part{background:var(--warn-soft);border-color:var(--warn);color:var(--warn)}
.items button.cur{outline:2px solid var(--accent);outline-offset:1px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin-bottom:14px;min-width:0}
.rid{font:500 .9rem var(--mono);color:var(--accent)}
.ticket{white-space:pre-wrap;overflow-wrap:anywhere}
.reply{white-space:pre-wrap;overflow-wrap:anywhere;font-size:1.02rem}
.cite{font:500 .8em var(--mono);background:var(--cite);color:var(--cite-ink);border-radius:4px;padding:1px 5px;white-space:nowrap}
.src{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.9rem;max-height:20rem;overflow:auto;border-left:3px solid var(--line);padding-left:12px;margin:0 0 10px}
.srchead{font:500 .82rem var(--mono);color:var(--cite-ink);margin-bottom:4px}
.row{display:flex;flex-wrap:wrap;gap:14px 24px;align-items:flex-start}
.field{display:flex;flex-direction:column;gap:4px;min-width:0}
.field.grow{flex:1 1 280px}
.seg{display:flex;gap:8px}
.seg label{display:flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:6px;padding:7px 12px;cursor:pointer;color:var(--ink);background:var(--bg)}
.seg input{accent-color:var(--accent)}
.seg label.on{border-color:var(--accent);background:var(--accent-soft)}
.nav{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
button.btn{font:600 .9rem var(--sans);border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:8px 14px;cursor:pointer}
button.btn.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.note{font-size:.85rem;color:var(--muted)}
.export textarea{font:.8rem/1.4 var(--mono);min-height:8rem}
.hint{color:var(--warn);font-size:.85rem;min-height:1.2em}
@media (max-width:760px){.layout{grid-template-columns:1fr}.list{position:static;max-height:none}}
@media (prefers-reduced-motion:no-preference){.bar i{transition:width .2s}}
</style>

<div class="wrap">
  <div class="top">
    <div>
      <h1>Reply review sheet</h1>
      <p>__N__ replies the system would send. For each one, check every factual statement against the source article and the citations against the sentence they support.</p>
    </div>
    <div class="who"><label for="who">Your initials</label><input type="text" id="who" maxlength="6" autocomplete="off"></div>
  </div>

  <details class="how" id="how" open>
    <summary>How to assess (read once)</summary>
    <ul>
      <li><b>Claim</b> means a factual statement about CloudServe's product, limits, steps, timings or settings. Greetings, thanks and offers to help are not claims.</li>
      <li>A claim is <b>supported</b> if the source article states it or plainly implies it. If a number, limit, step, setting or timing is not in the article, count it as <b>unsupported</b>, even if it sounds right.</li>
      <li>Enter how many unsupported claims the reply contains (0 if none) and paste or type each one in the box.</li>
      <li><b>Citations correct</b> means each [DOC-...] tag sits on a sentence that the cited article really supports, and no needed sentence is missing a tag.</li>
      <li>Work alone. Do not compare with the second assessor until you have both copied your results. The page does not show how the system or any checker scored a reply.</li>
      <li>Your answers are kept in this browser as you type. Use the export box at the bottom when you finish.</li>
    </ul>
  </details>

  <div class="layout">
    <aside class="list" aria-label="Replies">
      <div class="progress" id="progress">0 of __N__ done</div>
      <div class="bar"><i id="bar"></i></div>
      <div class="items" id="items"></div>
      <p class="note" style="margin-top:10px">Green: finished. Amber: started.</p>
    </aside>

    <main>
      <section class="card">
        <h2>Reply <span class="rid" id="rid"></span></h2>
        <div class="reply" id="reply"></div>
      </section>
      <section class="card">
        <h2>Customer ticket</h2>
        <div class="ticket" id="ticket"></div>
      </section>
      <section class="card">
        <h2>Source article(s) the reply cites</h2>
        <div id="sources"></div>
      </section>
      <section class="card">
        <h2>Your assessment</h2>
        <div class="row">
          <div class="field"><label for="count">Unsupported claims (count)</label>
            <input type="number" id="count" min="0" max="20" inputmode="numeric"></div>
          <div class="field grow"><label for="claims">Which claims were unsupported</label>
            <textarea id="claims" placeholder="One per line. Leave empty if the count is 0."></textarea></div>
        </div>
        <div class="row" style="margin-top:12px">
          <div class="field"><span class="note" id="citeq">Citations correct?</span>
            <div class="seg" role="radiogroup" aria-labelledby="citeq">
              <label id="lyes"><input type="radio" name="cites" value="yes" id="cyes">Yes</label>
              <label id="lno"><input type="radio" name="cites" value="no" id="cno">No</label>
            </div></div>
          <div class="field grow"><label for="notes">Notes (optional)</label>
            <textarea id="notes" style="min-height:3.2rem"></textarea></div>
        </div>
        <div class="hint" id="hint"></div>
        <div class="nav">
          <button class="btn" id="prev">Previous</button>
          <button class="btn primary" id="next">Save and next</button>
          <button class="btn" id="nextopen">Next unfinished</button>
        </div>
      </section>
      <section class="card export">
        <h2>Export your results</h2>
        <p class="note">Copy this text into a file called <b>review_&lt;your initials&gt;.csv</b> and send it back. It holds only your answers and the reply numbers.</p>
        <textarea id="csv" readonly rows="6" aria-label="Results as CSV"></textarea>
        <div class="nav"><button class="btn primary" id="copy">Copy results</button><span class="note" id="copied"></span></div>
      </section>
    </main>
  </div>
</div>

<script>
const DATA = __DATA__;
const SRC = __SOURCES__;
const KEY = 'reply-review-v1';
let state = {who: '', answers: {}, cur: 0};
try { const s = JSON.parse(localStorage.getItem(KEY) || 'null'); if (s && s.answers) state = Object.assign(state, s); } catch (e) {}
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function ans(i){ return state.answers[DATA[i].id] || {}; }
function status(i){ const a = ans(i); const full = a.count !== undefined && a.count !== '' && a.cites; const any = a.count || a.cites || a.claims || a.notes; return full ? 'done' : any ? 'part' : ''; }
function csvCell(v){ v = String(v == null ? '' : v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function buildCsv(){
  const head = ['review_id','unsupported_claims_found(count)','claim_text(s)','citations_correct(yes/no)','notes','assessor_initials'];
  const rows = DATA.map(d => { const a = state.answers[d.id] || {}; return [d.id, a.count ?? '', (a.claims || '').replace(/\n/g, ' | '), a.cites || '', (a.notes || '').replace(/\n/g, ' '), state.who]; });
  return [head, ...rows].map(r => r.map(csvCell).join(',')).join('\n');
}
function renderList(){
  const box = $('items'); box.innerHTML = '';
  let done = 0;
  DATA.forEach((d, i) => {
    const st = status(i); if (st === 'done') done++;
    const b = document.createElement('button'); b.textContent = d.id; b.className = st + (i === state.cur ? ' cur' : '');
    b.setAttribute('aria-label', 'Reply ' + d.id + (st === 'done' ? ', finished' : st ? ', started' : ''));
    b.onclick = () => go(i); box.appendChild(b);
  });
  $('progress').textContent = done + ' of ' + DATA.length + ' done';
  $('bar').style.width = (100 * done / DATA.length) + '%';
  $('csv').value = buildCsv();
}
function show(){
  const d = DATA[state.cur], a = ans(state.cur);
  $('rid').textContent = d.id;
  $('reply').innerHTML = esc(d.reply).replace(/\[(DOC-[A-Z0-9-]+)\]/g, '<span class="cite">[$1]</span>');
  $('ticket').textContent = d.ticket;
  $('sources').innerHTML = d.docs.map(id => '<div class="srchead">' + esc(id) + '</div><div class="src">' + esc(SRC[id] || '(article text not available)') + '</div>').join('');
  $('count').value = a.count ?? ''; $('claims').value = a.claims || ''; $('notes').value = a.notes || '';
  $('cyes').checked = a.cites === 'yes'; $('cno').checked = a.cites === 'no';
  $('lyes').classList.toggle('on', a.cites === 'yes'); $('lno').classList.toggle('on', a.cites === 'no');
  $('hint').textContent = '';
  $('who').value = state.who;
  renderList();
}
function collect(){
  const d = DATA[state.cur];
  const cites = document.querySelector('input[name=cites]:checked');
  state.answers[d.id] = {count: $('count').value, claims: $('claims').value.trim(), cites: cites ? cites.value : '', notes: $('notes').value.trim()};
  save(); renderList();
}
function go(i){ collect(); state.cur = Math.max(0, Math.min(DATA.length - 1, i)); save(); show(); window.scrollTo({top: 0}); }
function check(){
  const a = ans(state.cur); const n = parseInt(a.count, 10);
  if (a.count === undefined || a.count === '') return 'Enter the number of unsupported claims (0 if none).';
  if (n > 0 && !a.claims) return 'You entered ' + n + ' unsupported claim(s); list them in the box.';
  if (!a.cites) return 'Choose whether the citations are correct.';
  return '';
}
['count','claims','notes'].forEach(id => $(id).addEventListener('input', collect));
document.querySelectorAll('input[name=cites]').forEach(r => r.addEventListener('change', () => { collect(); $('lyes').classList.toggle('on', $('cyes').checked); $('lno').classList.toggle('on', $('cno').checked); }));
$('who').addEventListener('input', () => { state.who = $('who').value.trim().toUpperCase(); save(); renderList(); });
$('prev').onclick = () => go(state.cur - 1);
$('next').onclick = () => { collect(); const m = check(); $('hint').textContent = m; if (!m) go(state.cur + 1); };
$('nextopen').onclick = () => { collect(); const i = DATA.findIndex((_, k) => k > state.cur && status(k) !== 'done'); const j = i >= 0 ? i : DATA.findIndex((_, k) => status(k) !== 'done'); if (j >= 0) go(j); };
$('copy').onclick = async () => {
  const t = $('csv'); t.value = buildCsv();
  try { await navigator.clipboard.writeText(t.value); $('copied').textContent = 'Copied.'; }
  catch (e) { t.focus(); t.select(); $('copied').textContent = 'Press Ctrl/Cmd+C to copy the selected text.'; }
};
document.addEventListener('keydown', e => { if (e.target.matches('input,textarea')) return; if (e.key === 'ArrowRight') go(state.cur + 1); if (e.key === 'ArrowLeft') go(state.cur - 1); });
if (state.who) $('how').open = false;
show();
</script>
"""


def build(sheet: Path) -> str:
    items, sources = [], {}
    with sheet.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            docs = [d.strip() for d in row["cited_documents"].split(",") if d.strip()]
            # articles contain blank lines themselves, so split only where a new "[DOC-...] " block starts
            for block in re.split(r"\n\n(?=\[DOC-[A-Z0-9-]+\] )", row["source_article_text"]):
                if block.startswith("[") and "]" in block:
                    sources[block[1 : block.index("]")]] = block[block.index("]") + 2 :]
            items.append(
                {"id": row["review_id"], "ticket": row["ticket"], "reply": row["reply_to_check"], "docs": docs}
            )
    html = TEMPLATE.replace("__N__", str(len(items)))
    html = html.replace("__DATA__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    html = html.replace("__SOURCES__", json.dumps(sources, ensure_ascii=False).replace("</", "<\\/"))
    return html


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).write_text(build(Path(a.sheet)), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
