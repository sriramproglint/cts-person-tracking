"""Generate a self-contained HTML tracking quality report."""

from __future__ import annotations

import json
from pathlib import Path


def _payload_to_json(payload: dict) -> str:
    return json.dumps(payload).replace("</", "<\\/")


def render_report_html(payload: dict, *, title: str = "Tracking Quality Report") -> str:
    data_json = _payload_to_json(payload)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg:#0f1419; --card:#1a2332; --border:#2d3a4d; --text:#e7ecf3;
      --muted:#8b9cb3; --accent:#3b82f6; --green:#22c55e; --amber:#f59e0b; --red:#ef4444;
    }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:system-ui,sans-serif; background:var(--bg); color:var(--text);
      padding:1.5rem; max-width:1200px; margin:0 auto; line-height:1.5; }}
    h1 {{ font-size:1.75rem; }}
    .subtitle {{ color:var(--muted); margin:0.25rem 0 1.5rem; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1rem; margin-bottom:1.5rem; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1rem; }}
    .card .label {{ font-size:0.75rem; color:var(--muted); text-transform:uppercase; }}
    .card .value {{ font-size:1.75rem; font-weight:700; margin-top:0.25rem; }}
    .card .hint {{ font-size:0.8rem; color:var(--muted); margin-top:0.35rem; }}
    .score-good {{ color:var(--green); }} .score-mid {{ color:var(--amber); }} .score-bad {{ color:var(--red); }}
    section {{ background:var(--card); border:1px solid var(--border); border-radius:10px;
      padding:1.25rem; margin-bottom:1.25rem; }}
    section h2 {{ font-size:1.1rem; margin-bottom:1rem; padding-bottom:0.5rem; border-bottom:1px solid var(--border); }}
    table {{ width:100%; border-collapse:collapse; font-size:0.875rem; }}
    th,td {{ text-align:left; padding:0.55rem 0.65rem; border-bottom:1px solid var(--border); }}
    th {{ color:var(--muted); font-size:0.75rem; text-transform:uppercase; }}
    tr:hover td {{ background:rgba(255,255,255,0.03); }}
    .badge {{ display:inline-block; padding:0.15rem 0.5rem; border-radius:999px; font-size:0.7rem; font-weight:600; }}
    .badge-stable {{ background:rgba(34,197,94,0.2); color:var(--green); }}
    .badge-gaps {{ background:rgba(245,158,11,0.2); color:var(--amber); }}
    .badge-short {{ background:rgba(239,68,68,0.2); color:var(--red); }}
    .legend {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:0.75rem; }}
    .legend-item {{ padding:0.75rem; background:rgba(0,0,0,0.2); border-radius:8px; font-size:0.85rem; }}
    .legend-item strong {{ display:block; margin-bottom:0.25rem; color:var(--accent); }}
    .chart-wrap {{ height:180px; }} canvas {{ width:100%; height:100%; display:block; }}
    .empty {{ color:var(--muted); font-style:italic; }}
    .bar-cell {{ display:flex; align-items:center; gap:0.5rem; }}
    .mini-bar {{ width:100px; height:6px; background:var(--border); border-radius:3px; overflow:hidden; }}
    .mini-bar span {{ display:block; height:100%; background:var(--accent); }}
    footer {{ text-align:center; color:var(--muted); font-size:0.8rem; margin-top:2rem; }}
    #file-load {{ display:none; margin-bottom:1rem; padding:0.75rem; border:1px dashed var(--border); border-radius:8px; }}
  </style>
</head>
<body>
  <div id="file-load">Load JSON: <input type="file" id="json-file" accept=".json"></div>
  <h1>Tracking Quality Report</h1>
  <p class="subtitle" id="meta">Person detection + StrongSORT ID stability</p>
  <div class="cards" id="summary-cards"></div>
  <section>
    <h2>What the metrics mean</h2>
    <div class="legend">
      <div class="legend-item"><strong>ID stability score</strong>Percent of IDs that stayed long with high frame coverage.</div>
      <div class="legend-item"><strong>Suspected ID swaps</strong>Two people were close; centroids look like IDs switched.</div>
      <div class="legend-item"><strong>Suspected fragmentation</strong>New ID appeared where another ID recently vanished.</div>
      <div class="legend-item"><strong>Per-ID journey</strong>First/last frame, coverage %, gaps, longest streak.</div>
      <div class="legend-item"><strong>Unmatched detections</strong>Detector found more people than confirmed tracks (n_init delay).</div>
    </div>
  </section>
  <section>
    <h2>Detections vs tracks (per frame)</h2>
    <div class="chart-wrap"><canvas id="chart"></canvas></div>
  </section>
  <section>
    <h2>Suspected ID swaps</h2>
    <p class="empty" id="swaps-empty">No swaps detected.</p>
    <table id="swaps-table" style="display:none"><thead><tr>
      <th>Frame</th><th>IDs</th><th>Distance</th><th>Confidence</th><th>Keep cost</th><th>Swap cost</th>
    </tr></thead><tbody></tbody></table>
  </section>
  <section>
    <h2>Suspected fragmentation</h2>
    <p class="empty" id="frag-empty">No fragmentation detected.</p>
    <table id="frag-table" style="display:none"><thead><tr>
      <th>Frame</th><th>New ID</th><th>Lost ID</th><th>Gap</th><th>Distance (px)</th>
    </tr></thead><tbody></tbody></table>
  </section>
  <section>
    <h2>Per-ID journey</h2>
    <table id="ids-table"><thead><tr>
      <th>ID</th><th>Frames</th><th>Seen</th><th>Coverage</th><th>Best streak</th><th>Avg score</th><th>Status</th>
    </tr></thead><tbody></tbody></table>
  </section>
  <footer>CTS-Person-Tracking</footer>
  <script>
    const EMBEDDED = {data_json};
    function scoreClass(p) {{ return p >= 70 ? "score-good" : p >= 40 ? "score-mid" : "score-bad"; }}
    function idBadge(r, sf) {{
      if (r.frames_seen < sf) return ["short","Short-lived"];
      if (r.gaps > 0) return ["gaps","Has gaps"];
      if (r.stability_pct >= 90) return ["stable","Stable"];
      return ["gaps","Partial"];
    }}
    function render(data) {{
      const s = data.summary || {{}};
      const sf = 30;
      document.getElementById("meta").textContent =
        (s.frames_processed||0) + " frames · " + (s.unique_ids||0) + " IDs · stability " + (s.id_stability_score||0) + "%";
      const cards = [
        ["Stability score", (s.id_stability_score||0)+"%", "Higher = better", scoreClass(s.id_stability_score||0)],
        ["Stable IDs", s.stable_ids??"—", "Long + high coverage", ""],
        ["Unique IDs", s.unique_ids??"—", "Total created", ""],
        ["ID swaps", s.suspected_swaps??0, "At crossings", s.suspected_swaps>0?"score-bad":"score-good"],
        ["Fragmentation", s.suspected_fragments??0, "ID after occlusion", ""],
        ["Avg dets/tracks", (s.detection?.avg_per_frame??"—")+" / "+(s.tracking?.avg_per_frame??"—"), "Per frame", ""]
      ];
      document.getElementById("summary-cards").innerHTML = cards.map(function(c) {{
        return '<div class="card"><div class="label">'+c[0]+'</div><div class="value '+c[3]+'">'+c[1]+'</div><div class="hint">'+c[2]+'</div></div>';
      }}).join("");
      const swaps = data.suspected_swaps||[];
      if (swaps.length) {{
        document.getElementById("swaps-empty").style.display="none";
        var t=document.getElementById("swaps-table"); t.style.display="table";
        t.querySelector("tbody").innerHTML=swaps.map(function(e) {{
          return "<tr><td>"+e.frame+"</td><td>ID "+e.id_a+" &harr; ID "+e.id_b+"</td><td>"+e.distance_px+" px</td><td>"+Math.round(e.confidence*100)+"%</td><td>"+e.same_cost+" px</td><td>"+e.swap_cost+" px</td></tr>";
        }}).join("");
      }}
      const frags = data.suspected_fragments||[];
      if (frags.length) {{
        document.getElementById("frag-empty").style.display="none";
        var t2=document.getElementById("frag-table"); t2.style.display="table";
        t2.querySelector("tbody").innerHTML=frags.map(function(e) {{
          return "<tr><td>"+e.frame+"</td><td>ID "+e.new_id+"</td><td>ID "+e.lost_id+"</td><td>"+e.frames_since_lost+"f</td><td>"+e.distance_px+" px</td></tr>";
        }}).join("");
      }}
      const ids=(data.per_id||[]).slice().sort(function(a,b){{ return b.frames_seen-a.frames_seen || b.stability_pct-a.stability_pct; }});
      document.getElementById("ids-table").querySelector("tbody").innerHTML=ids.map(function(r) {{
        var b=idBadge(r,sf);
        return "<tr><td><strong>ID "+r.track_id+"</strong></td><td>"+r.first_frame+"–"+r.last_frame+"</td><td>"+r.frames_seen+"/"+r.span_frames+"</td>"+
          "<td><div class=\\"bar-cell\\">"+r.stability_pct+"%<div class=\\"mini-bar\\"><span style=\\"width:"+Math.min(100,r.stability_pct)+"%\\"></span></div></div></td>"+
          "<td>"+r.longest_streak+"</td><td>"+r.avg_score.toFixed(2)+"</td><td><span class=\\"badge badge-"+b[0]+"\\">"+b[1]+"</span></td></tr>";
      }}).join("");
      drawChart(data.per_frame||[]);
    }}
    function drawChart(frames) {{
      if (!frames.length) return;
      var canvas=document.getElementById("chart"), ctx=canvas.getContext("2d");
      var dpr=window.devicePixelRatio||1, rect=canvas.parentElement.getBoundingClientRect();
      canvas.width=rect.width*dpr; canvas.height=rect.height*dpr; ctx.scale(dpr,dpr);
      var W=rect.width,H=rect.height, pad={{l:40,r:12,t:12,b:28}};
      var maxY=Math.max.apply(null,[1].concat(frames.map(function(f){{ return Math.max(f.num_dets,f.num_tracks); }})))+1;
      var step=(W-pad.l-pad.r)/Math.max(1,frames.length-1);
      function x(i){{ return pad.l+i*step; }}
      function y(v){{ return pad.t+(H-pad.t-pad.b)*(1-v/maxY); }}
      ctx.clearRect(0,0,W,H); ctx.strokeStyle="#2d3a4d";
      for(var g=0;g<=4;g++){{ var gy=pad.t+(H-pad.t-pad.b)*g/4; ctx.beginPath(); ctx.moveTo(pad.l,gy); ctx.lineTo(W-pad.r,gy); ctx.stroke(); }}
      function line(key,color){{ ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); frames.forEach(function(f,i){{ var px=x(i),py=y(f[key]); if(i)ctx.lineTo(px,py); else ctx.moveTo(px,py); }}); ctx.stroke(); }}
      line("num_dets","#3b82f6"); line("num_tracks","#22c55e");
      ctx.fillStyle="#8b9cb3"; ctx.font="11px system-ui"; ctx.fillText("Detections",pad.l,H-8);
      ctx.fillStyle="#22c55e"; ctx.fillText("Tracks",pad.l+80,H-8);
    }}
    if (EMBEDDED && EMBEDDED.summary) {{ render(EMBEDDED); }}
    else {{
      document.getElementById("file-load").style.display="block";
      document.getElementById("json-file").addEventListener("change",function(e) {{
        var f=e.target.files[0]; if(!f)return;
        var r=new FileReader();
        r.onload=function(){{ try{{ render(JSON.parse(r.result)); }}catch(err){{ alert("Invalid JSON"); }} }};
        r.readAsText(f);
      }});
    }}
  </script>
</body>
</html>"""


def save_report_html(payload: dict, path: Path | str, *, title: str = "Tracking Quality Report") -> None:
    Path(path).write_text(render_report_html(payload, title=title), encoding="utf-8")
