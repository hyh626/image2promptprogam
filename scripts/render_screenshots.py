"""Render PNG screenshots of the viewer's three views without a browser.

Uses the viewer's /api/inspect endpoint to get the live data, builds a
static HTML version (mirroring the SPA's renderers in Python), then
renders via weasyprint -> PDF -> PNG through pypdfium2.

Usage:
    python view_eval_results.py --root /tmp/demo-fixture --port 8780 &
    python scripts/render_screenshots.py --base http://127.0.0.1:8780 --out docs/screenshots
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CSS = """
:root {
  --fg: #1a1a1a; --muted: #666; --bg: #fafafa; --panel: #fff;
  --border: #e5e5e5; --accent: #2257d6; --good: #1f8f3a; --bad: #c83232;
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: var(--fg); background: var(--bg); }
.layout { display: flex; min-height: 100%; }
.sidebar { width: 320px; min-width: 320px; background: var(--panel); border-right: 1px solid var(--border); padding: 14px; }
.main { flex: 1; padding: 18px 22px; }
.bc { font-size: 13px; margin-bottom: 14px; word-break: break-all; }
.bc a { color: var(--accent); text-decoration: none; }
.bc .sep { color: var(--muted); margin: 0 4px; }
ul.dirlist { list-style: none; padding: 0; margin: 0; }
ul.dirlist li { padding: 6px 6px; border-bottom: 1px solid #f1f1f1; display: flex; align-items: center; }
ul.dirlist a { color: var(--fg); text-decoration: none; flex: 1; word-break: break-all; }
.badge { font-size: 11px; padding: 1px 6px; border-radius: 9px; background: #eef2ff; color: var(--accent); margin-left: 6px; }
.badge.run { background: #e8f5ec; color: var(--good); }
.badge.exps { background: #fff4e0; color: #b76b00; }
.hint { color: var(--muted); font-style: italic; }
h2 { margin: 0 0 6px 0; font-size: 18px; }
h3 { margin: 22px 0 8px 0; font-size: 15px; }
h4 { margin: 0 0 6px 0; font-size: 13px; word-break: break-all; }
table { border-collapse: collapse; width: 100%; background: var(--panel); border: 1px solid var(--border); }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; font-size: 12.5px; }
th { background: #f5f6f9; font-weight: 600; }
tr:hover { background: #f5f8ff; }
td.composite { font-weight: 600; font-variant-numeric: tabular-nums; }
td.metric { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 12px; color: #333; }
.meta-row { color: var(--muted); font-size: 13px; margin: 4px 0; }
.meta-row b { color: var(--fg); }
.meta-row code { background: #f1f3f7; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 6px 0; }
figure { margin: 0; text-align: center; }
figure img { max-width: 100%; max-height: 160px; border: 1px solid var(--border); border-radius: 4px; background: #f5f5f5; }
figure figcaption { font-size: 11px; color: var(--muted); margin-top: 2px; }
.scores { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 11.5px; margin: 6px 0; word-break: break-all; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 11px; }
.pill.promoted { background: #e8f5ec; color: var(--good); }
.pill.rejected { background: #fdecec; color: var(--bad); }
.pill.reverted { background: #fff4e0; color: #b76b00; }
.pill.no_leader { background: #eef2ff; color: var(--accent); }
.pill.fail { background: #fdecec; color: var(--bad); }
.pill.pass { background: #e8f5ec; color: var(--good); }
.heading { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 4px; }
.heading code { background: #f1f3f7; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
.tabs { display: flex; gap: 4px; margin: 0 0 12px 0; border-bottom: 1px solid var(--border); }
.tab { padding: 6px 12px; font-size: 13px; color: var(--muted); border-radius: 4px 4px 0 0; }
.tab.active { color: var(--fg); border: 1px solid var(--border); border-bottom-color: var(--panel); background: var(--panel); margin-bottom: -1px; }
.timeline-wrap { overflow: visible; }
table.timeline { border-collapse: separate; border-spacing: 0; background: var(--panel); border: 1px solid var(--border); width: auto; }
table.timeline th, table.timeline td { padding: 6px 8px; border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); vertical-align: top; }
table.timeline thead th { background: #f5f6f9; font-size: 12px; text-align: left; }
table.timeline tbody th { text-align: left; font-weight: 600; font-size: 12.5px; min-width: 160px; max-width: 220px; word-break: break-word; }
.tcol { min-width: 180px; max-width: 220px; }
.tcol-head a { color: var(--accent); text-decoration: none; word-break: break-all; }
.tcol-meta { color: var(--muted); font-size: 11px; margin-top: 2px; }
.tcell { width: 180px; max-width: 200px; text-align: center; }
.tcell img { max-width: 160px; max-height: 130px; border: 1px solid var(--border); border-radius: 3px; background: #f5f5f5; }
.tcell.promoted { background: #f4faf6; }
.tcell.rejected { background: #fdf6f6; }
.tcell-img { background: #fafbfd; }
.tcell-target img { max-height: 130px; }
.tplaceholder { height: 100px; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 11px; background: #f9f9f9; border: 1px dashed var(--border); border-radius: 3px; }
.tscores { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 11px; margin-top: 4px; word-break: break-all; }
.tscores.tdetail { color: var(--muted); font-size: 10.5px; }
.tscores .pos { color: var(--good); }
.tscores .neg { color: var(--bad); }
.tcell-target-head, .tcell-img-head { background: #f0f2f7; }
"""


def get_json(base: str, path: str) -> dict:
    url = f"{base}/api/inspect?path={urllib.parse.quote(path)}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def get_image_data_uri(base: str, api_url: str | None) -> str | None:
    if not api_url:
        return None
    full = f"{base}{api_url}"
    try:
        with urllib.request.urlopen(full) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "image/png")
    except Exception:
        return None
    return f"data:{ctype};base64,{base64.b64encode(data).decode('ascii')}"


def fmt(n, d=4):
    if n is None or not isinstance(n, (int, float)):
        return "—"
    return f"{n:.{d}f}"


def esc(s):
    return html.escape("" if s is None else str(s), quote=True)


def render_breadcrumb(d):
    parts = []
    for i, b in enumerate(d["breadcrumb"]):
        if i > 0:
            parts.append('<span class="sep">/</span>')
        parts.append(f'<a>{esc(b["name"])}</a>')
    return f'<div class="bc">{"".join(parts)}</div>'


def render_sidebar(d):
    items = []
    for s in d.get("subdirs", []):
        kind = s.get("kind", "")
        cls = "run" if kind == "run" else ("exps" if kind == "runs_container" else "")
        label = "run" if kind == "run" else ("experiments" if kind == "runs_container" else "")
        badge = f'<span class="badge {cls}">{esc(label)}</span>' if label else ""
        items.append(f'<li><a>{esc(s["name"])}</a>{badge}</li>')
    if not items:
        listing = '<p class="hint">(no subdirectories)</p>'
    else:
        listing = f'<ul class="dirlist">{"".join(items)}</ul>'
    return f'<aside class="sidebar">{render_breadcrumb(d)}{listing}</aside>'


def decision_pill(decision):
    if not decision:
        return ""
    cls = {"promoted": "promoted", "rejected": "rejected",
           "reverted_after_reeval": "reverted",
           "no_leader": "no_leader"}.get(decision, "")
    label = "reverted" if decision == "reverted_after_reeval" else decision
    return f'<span class="pill {cls}">{esc(label)}</span>'


def gate_pill(g):
    if not g:
        return ""
    cls = "pass" if g == "pass" else ("fail" if g == "fail" else "")
    return f'<span class="pill {cls}">{esc(g)}</span>'


def render_scores_inline(scores):
    if not scores:
        return '<span class="hint">no scores</span>'
    parts = [f"{esc(k)}=<b>{fmt(v, 3)}</b>" for k, v in scores.items()]
    return " · ".join(parts)


def metric_columns(rows):
    keys = []
    seen = set()
    for r in rows:
        for k in (r.get("means") or {}):
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return sorted(keys)


def render_summary(d):
    rows = d.get("summary") or []
    cols = metric_columns(rows)
    title = (
        f'<h2>{esc(d.get("path") or d.get("root_name"))} '
        f'<span style="color:var(--muted);font-weight:400;">— '
        f'{len(rows)} run{"" if len(rows)==1 else "s"}</span></h2>'
    )
    if not rows:
        return f'<main class="main">{title}<p class="hint">No runs.</p></main>'
    head_cells = (
        '<th>run_id</th><th>driver</th><th>split</th><th>status</th><th>started</th>'
        '<th>composite</th>' +
        "".join(f'<th>{esc(k)}</th>' for k in cols) +
        '<th>gate</th><th>decision</th>'
    )
    body_rows = []
    for r in rows:
        cells = [
            f'<td>{esc(r["run_id"])}</td>',
            f'<td>{esc(r.get("driver") or "")}</td>',
            f'<td>{esc(r.get("split") or "")}</td>',
            f'<td>{esc(r.get("status") or "")}</td>',
            f'<td>{esc(r.get("started_at") or "")}</td>',
            f'<td class="composite">{fmt(r.get("composite"), 4)}</td>',
        ]
        means = r.get("means") or {}
        for k in cols:
            cells.append(f'<td class="metric">{fmt(means.get(k), 3)}</td>')
        cells.append(f'<td>{gate_pill(r.get("single_run_gate"))}</td>')
        cells.append(f'<td>{decision_pill(r.get("decision"))}</td>')
        body_rows.append(f'<tr>{"".join(cells)}</tr>')
    table = (
        f'<table><thead><tr>{head_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )
    return f'<main class="main">{title}{table}</main>'


def render_timeline(d, base, leader_only: bool = False):
    """Mirror the JS renderTimeline; produce a static HTML grid."""
    timeline = d.get("timeline") or {}
    all_runs = timeline.get("runs") or []
    runs = [r for r in all_runs if r.get("is_leader_promotion")] if leader_only else all_runs
    image_ids = timeline.get("image_ids") or []
    cells = timeline.get("cells") or {}
    manifest = timeline.get("manifest") or {}

    title_path = esc(d.get("path") or d.get("root_name"))
    if leader_only:
        tabs = (
            '<div class="tabs">'
            '<span class="tab">Summary</span>'
            '<span class="tab">Timeline</span>'
            '<span class="tab active">Leader chain</span>'
            '</div>'
        )
    else:
        tabs = (
            '<div class="tabs">'
            '<span class="tab">Summary</span>'
            '<span class="tab active">Timeline</span>'
            '<span class="tab">Leader chain</span>'
            '</div>'
        )

    subtitle = (
        f'Leader chain — {len(runs)} promoted run{"" if len(runs)==1 else "s"}, '
        f'{len(image_ids)} image{"" if len(image_ids)==1 else "s"}'
        if leader_only else
        f'All runs — {len(runs)} run{"" if len(runs)==1 else "s"}, '
        f'{len(image_ids)} image{"" if len(image_ids)==1 else "s"}'
    )

    if not runs or not image_ids:
        body = (f'<p class="hint">{esc(subtitle)}</p>'
                '<p class="hint">Nothing to compare yet.</p>')
        return f'<main class="main"><h2>{title_path}</h2>{tabs}{body}</main>'

    head_cells = ['<th class="tcell-img-head">image</th>',
                  '<th class="tcell-target-head">target</th>']
    for r in runs:
        cls = ("promoted" if r.get("decision") in ("promoted", "no_leader")
               else "rejected" if r.get("decision") == "rejected" else "")
        head_cells.append(
            f'<th class="tcol {cls}"><div class="tcol-head">'
            f'<a>{esc(r.get("name") or r.get("run_id"))}</a>'
            f'<div class="tcol-meta">{esc((r.get("started_at") or "").replace("T", " ").replace("Z", ""))}</div>'
            f'<div class="tcol-meta">{decision_pill(r.get("decision"))} '
            f'composite=<b>{fmt(r.get("composite"), 4)}</b></div>'
            '</div></th>'
        )

    rows_html = []
    for image_id in image_ids:
        meta = manifest.get(image_id) or {}
        meta_bits = []
        if meta.get("split"):
            meta_bits.append(f'<div class="tcol-meta">split: <code>{esc(meta["split"])}</code></div>')
        if meta.get("category"):
            meta_bits.append(f'<div class="tcol-meta">{esc(meta["category"])}</div>')
        meta_block = "".join(meta_bits)

        target_uri = get_image_data_uri(base, meta.get("target_url"))
        target_cell = (
            f'<td class="tcell tcell-target"><img src="{target_uri}" alt="target"></td>'
            if target_uri else '<td class="tcell tcell-target"><div class="tplaceholder">no target</div></td>'
        )

        run_cells = []
        prev_composite = None
        row_cells = cells.get(image_id, {})
        for r in runs:
            cell = row_cells.get(r["run_id"])
            cls = ("promoted" if r.get("decision") in ("promoted", "no_leader")
                   else "rejected" if r.get("decision") == "rejected" else "")
            if not cell:
                run_cells.append(f'<td class="tcell {cls}"><div class="tplaceholder">not in run</div></td>')
                continue

            gen_uri = get_image_data_uri(base, cell.get("generated_url"))
            img_html = (
                f'<img src="{gen_uri}" alt="generated">' if gen_uri
                else '<div class="tplaceholder">no png</div>'
            )

            composite = cell.get("composite")
            delta_html = ""
            if isinstance(prev_composite, (int, float)) and isinstance(composite, (int, float)):
                d_val = composite - prev_composite
                sign = "+" if d_val > 0 else ""
                dcls = "pos" if d_val > 0 else ("neg" if d_val < 0 else "")
                delta_html = f' <span class="{dcls}">({sign}{d_val:.3f})</span>'

            scores_inline = " · ".join(
                f"{esc(k.replace('s_', ''))}={fmt(v, 3)}"
                for k, v in (cell.get("scores") or {}).items()
            )

            run_cells.append(
                f'<td class="tcell {cls}">{img_html}'
                f'<div class="tscores">composite=<b>{fmt(composite, 4)}</b>{delta_html}</div>'
                f'<div class="tscores tdetail">{scores_inline}</div>'
                '</td>'
            )
            if isinstance(composite, (int, float)):
                prev_composite = composite

        rows_html.append(
            f'<tr><th class="tcell-img"><div>{esc(image_id)}</div>{meta_block}</th>'
            f'{target_cell}{"".join(run_cells)}</tr>'
        )

    table = (
        '<div class="timeline-wrap"><table class="timeline">'
        f'<thead><tr>{"".join(head_cells)}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>'
    )

    return (
        '<main class="main">'
        f'<h2>{title_path}</h2>'
        f'{tabs}'
        f'<p class="meta-row">{esc(subtitle)}. Images on the rows, runs on the '
        'columns (left = oldest).</p>'
        f'{table}'
        '</main>'
    )


def render_run(d, base):
    rd = d.get("run_detail") or {}
    run = rd.get("run") or {}
    agg = rd.get("aggregate") or {}
    gate = rd.get("gate") or {}
    out = []
    out.append('<main class="main">')
    out.append(
        '<div class="heading">'
        f'<h2>{esc(run.get("run_id") or "(unnamed)")}</h2>'
        f'{decision_pill(gate.get("decision"))}'
        f'{gate_pill(gate.get("single_run_gate"))}'
        '</div>'
    )
    out.append(
        '<p class="meta-row">'
        f'<b>{esc(run.get("driver") or "?")}</b> · '
        f'harness=<code>{esc(run.get("harness_variant") or "?")}</code> · '
        f'split=<code>{esc(run.get("split") or "?")}</code> · '
        f'seeds=<code>{esc(json.dumps(run.get("seeds") or []))}</code> · '
        f'status=<b>{esc(run.get("status") or "?")}</b>'
        '</p>'
    )
    out.append(
        '<p class="meta-row">'
        f'composite=<b>{fmt(agg.get("composite"), 4)}</b>'
        f'{(" · means: " + render_scores_inline(agg.get("means") or {}))}'
        '</p>'
    )
    if run.get("hypothesis"):
        out.append(f'<p class="meta-row"><i>{esc(run["hypothesis"])}</i></p>')
    if run.get("takeaway"):
        out.append(f'<p class="meta-row">→ {esc(run["takeaway"])}</p>')

    images = rd.get("images") or []
    out.append(f'<h3>Per-image ({len(images)})</h3>')
    out.append('<div class="grid">')
    for img in images:
        target_data = get_image_data_uri(base, img.get("target_url"))
        gen_data = get_image_data_uri(base, img.get("generated_url"))
        target_html = (
            f'<img src="{target_data}" alt="target">' if target_data
            else '<div style="height:140px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;">target unavailable</div>'
        )
        gen_html = (
            f'<img src="{gen_data}" alt="generated">' if gen_data
            else '<div style="height:140px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;">no generated.png</div>'
        )
        out.append(
            f'<div class="card"><h4>{esc(img["image_id"])}</h4>'
            '<div class="pair">'
            f'<figure>{target_html}<figcaption>target</figcaption></figure>'
            f'<figure>{gen_html}<figcaption>generated</figcaption></figure>'
            '</div>'
            f'<div class="scores">{render_scores_inline(img.get("scores") or {})}</div>'
            '</div>'
        )
    out.append('</div>')
    out.append('</main>')
    return "".join(out)


def render_browser(d):
    here = d.get("path") or d.get("root_name")
    children = d.get("subdirs") or []
    exps = [c for c in children if c.get("kind") == "runs_container"]
    runs = [c for c in children if c.get("kind") == "run"]
    msg = "Pick a subdirectory in the sidebar."
    if exps:
        names = ", ".join(esc(c["name"]) for c in exps)
        msg += (f" This folder has <b>{len(exps)}</b> experiment "
                f"folder{'' if len(exps)==1 else 's'}: {names}.")
    elif runs:
        msg += f" This folder has <b>{len(runs)}</b> run folder{'' if len(runs)==1 else 's'}."
    return f'<main class="main"><h2>{esc(here)}</h2><p class="hint">{msg}</p></main>'


def page(body_inner: str, width_css: str = "1280px", height_css: str = "1200px") -> str:
    return f"""<!doctype html>
<html><head><meta charset='utf-8'>
<style>{CSS}
@page {{ size: {width_css} {height_css}; margin: 0; }}
.layout {{ width: {width_css}; min-height: {height_css}; }}
</style></head>
<body><div class="layout">{body_inner}</div></body></html>
"""


def _trim_to_content(img, margin: int = 36, threshold: int = 235, sample_step: int = 6):
    """Crop trailing whitespace below the last row containing any 'content' pixel.

    Detects content as any pixel darker than `threshold` after converting to
    grayscale. This works for both the white sidebar (text on #fff) and the
    grey main pane (text/borders on #fafafa), since all text/borders are
    materially darker than either bg.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    gray = img.convert("L")
    pixels = gray.load()
    last_row = -1
    for y in range(h - 1, -1, -1):
        for x in range(0, w, sample_step):
            if pixels[x, y] < threshold:
                last_row = y
                break
        if last_row >= 0:
            break
    if last_row < 0:
        return img
    return img.crop((0, 0, w, min(h, last_row + margin)))


def render_to_png(html_text: str, out_path: Path, width_css: str = "1280px",
                  height_css: str = "1100px") -> None:
    from weasyprint import HTML
    import pypdfium2 as pdfium
    page_size_css = f"@page {{ size: {width_css} {height_css}; margin: 0; }}"
    html_with_page = html_text.replace("@page { size:", page_size_css + " /*")
    # Re-inject by simple replace was clumsy; build cleanly instead.
    html_with_page = html_text  # the caller already set @page in CSS via page().
    pdf_bytes = HTML(string=html_with_page).write_pdf()
    pdf = pdfium.PdfDocument(pdf_bytes)
    img = pdf[0].render(scale=2.0).to_pil()
    img = _trim_to_content(img)
    img.save(out_path, "PNG", optimize=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8765",
                        help="viewer base URL (default: http://127.0.0.1:8765)")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory to write PNGs into")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1) Top-level browser view (root dir)
    root_data = get_json(args.base, "")
    sidebar = render_sidebar(root_data)
    browser_main = render_browser(root_data)
    render_to_png(page(sidebar + browser_main, "1280px", "600px"),
                  out / "01-browser.png", "1280px", "600px")
    print(f"wrote {out / '01-browser.png'}", file=sys.stderr)

    # 2) Summary table for experiments/runs
    runs_data = get_json(args.base, "experiments/runs")
    sidebar = render_sidebar(runs_data)
    summary_main = render_summary(runs_data)
    render_to_png(page(sidebar + summary_main, "1600px", "650px"),
                  out / "02-summary.png", "1600px", "650px")
    print(f"wrote {out / '02-summary.png'}", file=sys.stderr)

    # 3) Run detail for the leader (palette step)
    run_path = "experiments/runs/20260504T112800Z__claude-opus-4-7__add_palette_step"
    detail_data = get_json(args.base, run_path)
    sidebar = render_sidebar(detail_data)
    detail_main = render_run(detail_data, args.base)
    render_to_png(page(sidebar + detail_main, "1500px", "900px"),
                  out / "03-run-detail.png", "1500px", "900px")
    print(f"wrote {out / '03-run-detail.png'}", file=sys.stderr)

    # 4) Per-image timeline view (all runs across the experiments folder)
    sidebar = render_sidebar(runs_data)
    timeline_main = render_timeline(runs_data, args.base, leader_only=False)
    render_to_png(page(sidebar + timeline_main, "1700px", "900px"),
                  out / "04-timeline.png", "1700px", "900px")
    print(f"wrote {out / '04-timeline.png'}", file=sys.stderr)

    # 5) Same view filtered to the leader chain only
    sidebar = render_sidebar(runs_data)
    leader_main = render_timeline(runs_data, args.base, leader_only=True)
    render_to_png(page(sidebar + leader_main, "1500px", "900px"),
                  out / "05-leader-chain.png", "1500px", "900px")
    print(f"wrote {out / '05-leader-chain.png'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
