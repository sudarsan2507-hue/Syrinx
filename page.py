"""
page.py -- the browser front-end markup for webui.py.

Fifteen panels on one gapless grid, sharing 1px borders:

  01 brand/status   02 device      03 meter      04 levels
  05 transport      06 controls    07 detection  08 confidence
  09 scope          10 tuning      11 replay     12 latency
  13 counters       14 command map 15 history
"""

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Syrinx</title>
<style>
  :root{
    --bg:#0d1216; --panel:#131b21; --panel2:#1a242c; --panel3:#212d36;
    --line:#243139; --text:#e6edf0; --dim:#8b9ba4; --faint:#5d6c75;
    --accent:#4fbcd8; --good:#7fc79a; --warn:#d9a95c; --bad:#e08a85;
    --mono:"Consolas",ui-monospace,monospace;
    --sans:"Segoe UI",system-ui,-apple-system,sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{background:var(--bg);color:var(--text);font-family:var(--sans);
       font-size:13px;overflow:hidden}

  .grid{
    display:grid;height:100vh;gap:0;
    grid-template-columns:250px 1fr 1fr 236px;
    grid-template-rows:60px 92px 88px 132px 122px 78px 1fr;
    grid-template-areas:
      "brand  last   last    lat"
      "device conf   conf    stats"
      "meter  conf   conf    stats"
      "levels scope  scope   cmds"
      "send   tuning tuning  cmds"
      "ctrl   replay replay  cmds"
      "hist   hist   hist    hist";
    border-top:1px solid var(--line);
  }
  .p{border-right:1px solid var(--line);border-bottom:1px solid var(--line);
     padding:9px 13px;overflow:hidden;position:relative;min-width:0}
  .p:first-child{border-left:1px solid var(--line)}
  .p.pad0{padding:0}
  h2{font-size:9px;letter-spacing:.13em;text-transform:uppercase;
     color:var(--faint);font-weight:700;margin-bottom:7px;display:flex;
     justify-content:space-between;align-items:center}
  h2 .n{color:#33434d;font-family:var(--mono);font-size:8.5px}

  /* 01 brand */
  .brand{grid-area:brand;display:flex;flex-direction:column;justify-content:center;
         background:var(--panel)}
  .brand .nm{font-size:16px;font-weight:700;letter-spacing:.06em}
  .brand .st{display:flex;align-items:center;gap:6px;font-size:10.5px;
             color:var(--dim);margin-top:3px}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--faint);flex:none}
  .dot.ok{background:var(--good);box-shadow:0 0 7px var(--good)}
  .dot.error{background:var(--bad);box-shadow:0 0 7px var(--bad)}
  .st span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

  /* 02 device  05 transport */
  .device{grid-area:device}
  .send{grid-area:send}
  select{width:100%;background:var(--panel2);color:var(--text);
     border:1px solid var(--line);border-radius:3px;padding:6px 7px;
     font-family:var(--sans);font-size:11.5px;outline:none}
  select:focus{border-color:var(--accent)}
  .sub{font-size:9.5px;color:var(--faint);margin-top:5px;line-height:1.35}
  .link{font-family:var(--mono);font-size:10px;color:var(--dim);margin-top:6px}

  /* 03 meter */
  .meter{grid-area:meter}
  #bar{width:100%;height:26px;display:block;background:var(--panel2);
       border-radius:3px}

  /* 04 levels */
  .levels{grid-area:levels}
  .kv{display:flex;justify-content:space-between;font-family:var(--mono);
      font-size:11px;padding:3px 0;border-bottom:1px solid rgba(36,49,57,.55)}
  .kv:last-child{border-bottom:0}
  .kv b{color:var(--dim);font-weight:400}
  .kv i{font-style:normal;color:var(--text)}
  .kv i.hot{color:var(--good)} .kv i.thr{color:var(--warn)}

  /* 06 controls */
  .ctrl{grid-area:ctrl;display:flex;align-items:center}
  button{width:100%;border:0;border-radius:3px;padding:9px;cursor:pointer;
     font-family:var(--sans);font-size:12px;font-weight:600;
     background:var(--accent);color:#0d1216}
  button:hover{filter:brightness(1.12)}
  button.on{background:var(--panel3);color:var(--text);
            border:1px solid var(--line)}
  button.mini{width:auto;padding:6px 13px;font-size:11px;font-weight:500}
  button.ghost{background:transparent;color:var(--dim);
               border:1px solid var(--line);font-weight:500}

  /* 07 last detection */
  .last{grid-area:last;display:flex;align-items:center;gap:16px;
        background:var(--panel)}
  .last .word{font-size:34px;font-weight:700;letter-spacing:-.02em;
              line-height:1;color:var(--faint)}
  .last .meta{font-size:11.5px;color:var(--dim);line-height:1.5;min-width:0}
  .last .act{color:var(--text);font-weight:600}

  /* 08 confidence */
  .conf{grid-area:conf}
  .bars{display:flex;flex-direction:column;gap:3px}
  .b{display:grid;grid-template-columns:56px 1fr 38px;align-items:center;gap:8px}
  .b .nm{font-family:var(--mono);font-size:10.5px;color:var(--dim);text-align:right}
  .b .tr{height:13px;background:var(--panel2);border-radius:2px;overflow:hidden}
  .b .fl{height:100%;width:0;background:var(--panel3);border-radius:2px;
         transition:width .16s ease}
  .b .fl.top{background:var(--accent)} .b .fl.win{background:var(--good)}
  .b .pc{font-family:var(--mono);font-size:10.5px;color:var(--faint);text-align:right}

  /* 09 scope */
  .scope{grid-area:scope}
  #scope{width:100%;height:88px;display:block;background:var(--panel2);
         border-radius:3px}

  /* 10 tuning */
  .tuning{grid-area:tuning}
  .sl{display:grid;grid-template-columns:104px 1fr 52px;align-items:center;
      gap:9px;margin-bottom:5px}
  .sl label{font-size:10.5px;color:var(--dim)}
  .sl output{font-family:var(--mono);font-size:10.5px;text-align:right}
  input[type=range]{-webkit-appearance:none;appearance:none;height:3px;
     background:var(--panel3);border-radius:2px;outline:none;width:100%}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:11px;
     height:11px;border-radius:50%;background:var(--accent);cursor:pointer}
  input[type=range]::-moz-range-thumb{width:11px;height:11px;border:0;
     border-radius:50%;background:var(--accent);cursor:pointer}

  /* 11 replay */
  .replay{grid-area:replay}
  .rrow{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center}

  /* 12 latency  13 counters */
  .lat{grid-area:lat} .stats{grid-area:stats}
  .big{font-family:var(--mono);font-size:21px;color:var(--text);line-height:1}
  .big small{font-size:10px;color:var(--faint)}
  .cells{display:grid;grid-template-columns:1fr 1fr;gap:1px;
         background:var(--line);border:1px solid var(--line);border-radius:3px;
         overflow:hidden;margin-top:2px}
  .cell{background:var(--panel);padding:7px 9px}
  .cell u{display:block;font-size:8.5px;letter-spacing:.1em;text-transform:uppercase;
          color:var(--faint);text-decoration:none;margin-bottom:2px}
  .cell v{display:block;font-family:var(--mono);font-size:15px;color:var(--text)}
  .cell v.g{color:var(--good)} .cell v.w{color:var(--warn)} .cell v.d{color:var(--faint)}

  /* 14 command map */
  .cmds{grid-area:cmds}
  .map{display:flex;flex-direction:column;gap:1px;background:var(--line);
       border:1px solid var(--line);border-radius:3px;overflow:hidden}
  .mrow{display:grid;grid-template-columns:58px 1fr;gap:8px;background:var(--panel);
        padding:6px 9px;font-size:10.5px;align-items:center}
  .mrow.hit{background:var(--panel3)}
  .mrow s{font-family:var(--mono);color:var(--accent);text-decoration:none}
  .mrow e{color:var(--dim)}

  /* 15 history */
  .hist{grid-area:hist;padding:0;display:flex;flex-direction:column;
        border-right:1px solid var(--line)}
  .hh{padding:8px 13px 6px;border-bottom:1px solid var(--line);
      display:flex;justify-content:space-between;align-items:center}
  .hw{overflow-y:auto;flex:1}
  table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}
  thead th{position:sticky;top:0;background:var(--panel);text-align:left;
     font-family:var(--sans);font-size:8.5px;letter-spacing:.11em;
     text-transform:uppercase;color:var(--faint);font-weight:700;
     padding:6px 13px;border-bottom:1px solid var(--line)}
  td{padding:5px 13px;border-bottom:1px solid rgba(36,49,57,.4);
     white-space:nowrap}
  tr.accepted td.r{color:var(--good)}
  tr.cooldown td.r{color:var(--warn)}
  tr.undelivered td.r{color:var(--bad)}
  tr.rejected td{color:var(--faint)}
  .none{color:var(--faint);padding:14px 13px;font-family:var(--sans);font-size:11.5px}
  ::-webkit-scrollbar{width:9px;height:9px}
  ::-webkit-scrollbar-track{background:var(--bg)}
  ::-webkit-scrollbar-thumb{background:var(--panel3);border-radius:5px}
</style></head><body>

<div class="grid">

  <div class="p brand">
    <div class="nm">SYRINX</div>
    <div class="st"><i class="dot" id="dot"></i><span id="status">stopped</span></div>
  </div>

  <div class="p last">
    <div class="word" id="word">&mdash;</div>
    <div class="meta">
      <div class="act" id="act">waiting for a sound</div>
      <div id="why">nothing detected yet</div>
    </div>
  </div>

  <div class="p lat">
    <h2>Latency <span class="n">12</span></h2>
    <div class="big" id="lat_last">&mdash; <small>ms</small></div>
    <div class="kv" style="margin-top:6px"><b>mean</b><i id="lat_mean">&mdash;</i></div>
    <div class="kv"><b>best</b><i id="lat_min">&mdash;</i></div>
  </div>

  <div class="p device">
    <h2>Input device <span class="n">02</span></h2>
    <select id="device"></select>
    <div class="sub" id="devhint">switches live</div>
  </div>

  <div class="p conf">
    <h2>Confidence <span class="n">08</span></h2>
    <div class="bars" id="bars"></div>
  </div>

  <div class="p stats">
    <h2>Session <span class="n">13</span></h2>
    <div class="cells">
      <div class="cell"><u>onsets</u><v id="c_on" class="d">0</v></div>
      <div class="cell"><u>sent</u><v id="c_sent" class="g">0</v></div>
      <div class="cell"><u>rejected</u><v id="c_rej" class="d">0</v></div>
      <div class="cell"><u>cooled</u><v id="c_cool" class="w">0</v></div>
    </div>
  </div>

  <div class="p meter">
    <h2>Live input <span class="n">03</span></h2>
    <canvas id="bar"></canvas>
  </div>

  <div class="p levels">
    <h2>Levels <span class="n">04</span></h2>
    <div class="kv"><b>now</b><i id="l_now">&minus;60.0 dB</i></div>
    <div class="kv"><b>peak</b><i id="l_peak">&minus;60.0 dB</i></div>
    <div class="kv"><b>threshold</b><i class="thr" id="l_thr">&minus;34.0 dB</i></div>
    <div class="kv"><b>rate</b><i id="l_rate">&mdash;</i></div>
    <div class="kv"><b>state</b><i id="l_state">idle</i></div>
  </div>

  <div class="p scope">
    <h2>Level history <span class="n">09</span></h2>
    <canvas id="scope"></canvas>
  </div>

  <div class="p cmds">
    <h2>Command map <span class="n">14</span></h2>
    <div class="map" id="map"></div>
  </div>

  <div class="p send">
    <h2>Transport <span class="n">05</span></h2>
    <select id="transport">
      <option value="debug">debug &mdash; prints only</option>
      <option value="esp32">esp32 &mdash; USB serial</option>
      <option value="bluez">bluez &mdash; Linux only</option>
    </select>
    <div class="link" id="link">no link</div>
  </div>

  <div class="p tuning">
    <h2>Tuning <span class="n">10</span></h2>
    <div class="sl"><label>Onset threshold</label>
      <input type="range" id="s_thr" min="0.002" max="0.15" step="0.001">
      <output id="o_thr"></output></div>
    <div class="sl"><label>Min confidence</label>
      <input type="range" id="s_conf" min="0" max="1" step="0.01">
      <output id="o_conf"></output></div>
    <div class="sl"><label>Cooldown</label>
      <input type="range" id="s_cool" min="0" max="5" step="0.1">
      <output id="o_cool"></output></div>
    <div class="sl"><label>Post-onset wait</label>
      <input type="range" id="s_post" min="0.2" max="1.5" step="0.05">
      <output id="o_post"></output></div>
  </div>

  <div class="p ctrl">
    <button id="toggle">Start listening</button>
  </div>

  <div class="p replay">
    <h2>Replay a file <span class="n">11</span></h2>
    <div class="rrow">
      <select id="file"></select>
      <button class="mini" id="play">Replay</button>
    </div>
    <div class="sub">plays at real speed through the whole pipeline &mdash;
      no microphone involved</div>
  </div>

  <div class="p hist">
    <div class="hh">
      <h2 style="margin:0">History <span class="n">15</span></h2>
      <button class="ghost mini" id="save" style="width:auto">Save config.json</button>
    </div>
    <div class="hw">
      <table>
        <thead><tr><th>time</th><th>sound</th><th>conf</th><th>action</th>
          <th>result</th><th>latency</th></tr></thead>
        <tbody id="hist"><tr><td class="none" colspan="6">nothing yet</td></tr></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);
const FLOOR = -60;
let running=false, peak=FLOOR, commands=[], keymap={}, rows=0;
let lastDb=FLOOR, thrDb=-34, hist=[], lats=[], allDev=[];

const post = (u,b) => fetch(u,{method:'POST',
  headers:{'Content-Type':'application/json'}, body:JSON.stringify(b||{})});
const frac = db => Math.max(0, Math.min(1,(db-FLOOR)/-FLOOR));

/* ---------- 03 meter + 09 scope ---------- */
const bar=$('bar'), bctx=bar.getContext('2d');
const sc=$('scope'), sctx=sc.getContext('2d');
function fit(c){const r=devicePixelRatio||1;
  c.width=c.clientWidth*r;c.height=c.clientHeight*r;
  c.getContext('2d').setTransform(r,0,0,r,0,0);}
function sizeAll(){fit(bar);fit(sc);}
addEventListener('resize',sizeAll);

function draw(){
  const w=bar.clientWidth,h=bar.clientHeight;
  bctx.clearRect(0,0,w,h);
  const over=lastDb>=thrDb, f=frac(lastDb);
  if(f>0.004){bctx.fillStyle=over?'#7fc79a':'#4fbcd8';bctx.fillRect(0,0,w*f,h);}
  bctx.strokeStyle='#d9a95c';bctx.lineWidth=2;bctx.beginPath();
  bctx.moveTo(w*frac(thrDb),0);bctx.lineTo(w*frac(thrDb),h);bctx.stroke();
  bctx.strokeStyle='#e6edf0';bctx.lineWidth=1;bctx.beginPath();
  bctx.moveTo(w*frac(peak),3);bctx.lineTo(w*frac(peak),h-3);bctx.stroke();

  const W=sc.clientWidth,H=sc.clientHeight;
  sctx.clearRect(0,0,W,H);
  sctx.strokeStyle='rgba(36,49,57,.9)';sctx.lineWidth=1;
  for(let i=1;i<4;i++){const y=H*i/4;sctx.beginPath();
    sctx.moveTo(0,y);sctx.lineTo(W,y);sctx.stroke();}
  const N=Math.max(60,Math.floor(W));
  while(hist.length>N)hist.shift();
  if(hist.length>1){
    const step=W/(N-1);
    sctx.beginPath();sctx.moveTo(0,H);
    hist.forEach((d,i)=>sctx.lineTo(i*step,H-frac(d)*H));
    sctx.lineTo((hist.length-1)*step,H);sctx.closePath();
    sctx.fillStyle='rgba(79,188,216,.16)';sctx.fill();
    sctx.beginPath();
    hist.forEach((d,i)=>i?sctx.lineTo(i*step,H-frac(d)*H)
                        :sctx.moveTo(0,H-frac(d)*H));
    sctx.strokeStyle='#4fbcd8';sctx.lineWidth=1.4;sctx.stroke();
  }
  const ty=H-frac(thrDb)*H;
  sctx.setLineDash([4,3]);sctx.strokeStyle='#d9a95c';sctx.lineWidth=1;
  sctx.beginPath();sctx.moveTo(0,ty);sctx.lineTo(W,ty);sctx.stroke();
  sctx.setLineDash([]);
  requestAnimationFrame(draw);
}

/* ---------- 08 bars, 14 map ---------- */
function buildBars(){
  $('bars').innerHTML = commands.map(c=>
    `<div class="b"><span class="nm">${c}</span>
     <div class="tr"><div class="fl" id="f_${c}"></div></div>
     <span class="pc" id="p_${c}">0%</span></div>`).join('');
  $('map').innerHTML = commands.map(c=>
    `<div class="mrow" id="m_${c}"><s>${c}</s><e>${keymap[c]||'—'}</e></div>`).join('');
}

/* ---------- stream ---------- */
const es=new EventSource('/events');
es.onmessage=e=>{
  const m=JSON.parse(e.data);
  if(m.type==='level'){
    lastDb=m.db; thrDb=m.threshold_db; hist.push(m.db);
    peak=Math.max(m.db,peak-0.45);
    const hot=m.db>=m.threshold_db;
    $('l_now').textContent=m.db.toFixed(1)+' dB';
    $('l_now').className=hot?'hot':'';
    $('l_peak').textContent=peak.toFixed(1)+' dB';
    $('l_thr').textContent=m.threshold_db.toFixed(1)+' dB';
    $('l_state').textContent=m.running?'capturing':'idle';
    if(m.rate)$('l_rate').textContent=m.rate+' Hz';
  } else if(m.type==='status'){
    $('status').textContent=m.text;
    $('dot').className='dot '+(m.kind||'');
    $('link').textContent=m.text.startsWith('listening')?'link up':'no link';
  } else if(m.type==='detection'){ show(m); }
  else if(m.type==='config'){ sync(m); }
};

function show(m){
  const acc=m.decision==='accepted', pct=Math.round((m.confidence||0)*100);
  $('word').textContent=(m.label||'—').toUpperCase();
  $('word').style.color = acc?'var(--good)'
    : m.decision==='rejected_cooldown'?'var(--warn)'
    : m.decision==='accepted_undelivered'?'var(--bad)':'var(--faint)';
  $('act').textContent = acc?(m.action||'—'):'not sent';
  $('why').textContent =
      acc?`${pct}% confident · ${Math.round(m.latency_ms)} ms`
    : m.decision==='rejected_low_confidence'?`${pct}% — below the threshold`
    : m.decision==='rejected_cooldown'?'still cooling down'
    : m.decision==='accepted_undelivered'?'transport did not deliver it'
    : (m.decision||'').replace(/_/g,' ');

  const sp=m.spread||{};
  commands.forEach(c=>{
    const p=sp[c]||0, fl=$('f_'+c);
    fl.style.width=(p*100).toFixed(1)+'%';
    fl.className='fl'+(c===m.label?(acc?' win':' top'):'');
    const pe=$('p_'+c); pe.textContent=Math.round(p*100)+'%';
    pe.style.color=c===m.label?'var(--text)':'var(--faint)';
    const mr=$('m_'+c); if(mr) mr.className='mrow'+(c===m.label&&acc?' hit':'');
  });

  if(acc){
    lats.push(m.latency_ms);
    $('lat_last').innerHTML=Math.round(m.latency_ms)+' <small>ms</small>';
    $('lat_mean').textContent=Math.round(lats.reduce((a,b)=>a+b,0)/lats.length)+' ms';
    $('lat_min').textContent=Math.round(Math.min(...lats))+' ms';
  }
  const s=m.stats||{};
  $('c_on').textContent=s.onsets||0; $('c_sent').textContent=s.accepted||0;
  $('c_rej').textContent=s.rejected||0; $('c_cool').textContent=s.cooldown||0;

  const res={accepted:'sent',rejected_low_confidence:'low confidence',
    rejected_cooldown:'cooldown',accepted_undelivered:'not delivered',
    rejected_unusable:'unusable'}[m.decision]||m.decision;
  const cls=acc?'accepted':m.decision==='rejected_cooldown'?'cooldown'
    :m.decision==='accepted_undelivered'?'undelivered':'rejected';
  if(rows===0)$('hist').innerHTML='';
  $('hist').insertAdjacentHTML('afterbegin',
    `<tr class="${cls}"><td>${new Date().toLocaleTimeString()}</td>`+
    `<td>${m.label||'—'}</td><td>${pct}%</td><td>${m.action||'—'}</td>`+
    `<td class="r">${res}</td><td>${Math.round(m.latency_ms)} ms</td></tr>`);
  rows++;
  while($('hist').rows.length>200)$('hist').deleteRow(-1);
}

/* ---------- 10 tuning ---------- */
const SL=[['s_thr','o_thr','onset_threshold',v=>v.toFixed(3)],
          ['s_conf','o_conf','min_confidence',v=>Math.round(v*100)+'%'],
          ['s_cool','o_cool','cooldown_s',v=>v.toFixed(1)+'s'],
          ['s_post','o_post','post_onset_s',v=>v.toFixed(2)+'s']];
SL.forEach(([s,o,k,f])=>$(s).addEventListener('input',e=>{
  const v=parseFloat(e.target.value); $(o).textContent=f(v);
  post('/api/config',{[k]:v});
}));
function sync(c){SL.forEach(([s,o,k,f])=>{
  if(c[k]===undefined)return; $(s).value=c[k]; $(o).textContent=f(c[k]);});}

/* ---------- 02/05/06/11 controls ---------- */
$('toggle').onclick=async()=>{
  if(running){await post('/api/stop');}
  else{await post('/api/start',{device:parseInt($('device').value),
                                transport:$('transport').value});}
  running=!running;
  $('toggle').textContent=running?'Stop':'Start listening';
  $('toggle').className=running?'on':'';
};
function devHint(){
  const d=allDev.find(x=>x.index===parseInt($('device').value));
  if(!d){$('devhint').textContent='switches live';return;}
  $('l_rate').textContent=d.rate+' Hz';
  $('devhint').textContent = d.narrow
    ? `${d.rate} Hz narrowband — matches the training audio`
    : `opens at ${d.rate} Hz — switches live`;
}
$('device').onchange=()=>{post('/api/device',{device:parseInt($('device').value)});devHint();};
$('transport').onchange=()=>{ if(running){
  post('/api/stop').then(()=>post('/api/start',
    {device:parseInt($('device').value),transport:$('transport').value})); } };
$('save').onclick=()=>post('/api/save');
$('play').onclick=()=>post('/api/replay',{file:$('file').value});

/* ---------- boot ---------- */
(async()=>{
  const st=await(await fetch('/api/state')).json();
  commands=st.commands; keymap=st.keymap||{}; buildBars(); sync(st);
  $('transport').value=st.transport;
  $('file').innerHTML=st.files.map(f=>`<option value="${f}">${f}</option>`).join('');
  const dv=await(await fetch('/api/devices')).json();
  allDev=dv;
  $('device').innerHTML=dv.map(d=>{
    const nb=d.narrow?' ⚠':'';
    return `<option value="${d.index}"${d.default?' selected':''}>`+
           `${d.name} (${d.api}) ${d.rate}Hz${nb}</option>`;}).join('');
  devHint(); sizeAll(); requestAnimationFrame(draw);
})();
</script></body></html>
"""
