// CANDOR glass box: two networks trained with candorkit, run here layer by layer.
import { LegibleMLP, makeInput, rng, label } from './model.js';

const FILM = new URLSearchParams(location.search).has('film');
const $ = (s, r = document) => r.querySelector(s);
const LETTERS = 'ABCDEFGH';
const reduced = !FILM && matchMedia('(prefers-reduced-motion: reduce)').matches;
const sup = (s) => s.replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);
function fmtD(d) {
  if (d <= 0) return '0';
  if (d < 0.0001) { const [m, e] = d.toExponential(1).split('e'); return `${m}·10${e.startsWith('-') ? '⁻' : ''}${sup(e.replace(/^[-+]/, ''))}`; }
  return d < 0.01 ? d.toFixed(4) : d.toFixed(3);
}
const pct = (d) => `${(d * 100).toFixed(d < 0.01 ? 2 : 1)}%`;
const argmax = (p) => p.indexOf(Math.max(...p));
const listing = (ws) => (ws.length < 2 ? ws.join('') : `${ws.slice(0, -1).join(', ')} and ${ws[ws.length - 1]}`);

let M = null;

// ------------------------------------------------------------------ drawing helpers
function barsSVG(svg, series, { w = 320, h = 120, pad = 6 } = {}) {
  const n = series[0].values.length;
  const bw = (w - 2 * pad) / n;
  const max = Math.max(1e-6, ...series.flatMap((s) => s.values.map(Math.abs)));
  const mid = h / 2;
  let out = `<line class="zero" x1="0" x2="${w}" y1="${mid}" y2="${mid}"/>`;
  for (const s of series) {
    s.values.forEach((v, i) => {
      const bh = (Math.abs(v) / max) * (mid - 6);
      const x = pad + i * bw + bw * (s.inset || 0.12);
      const width = bw * (1 - 2 * (s.inset || 0.12));
      const y = v >= 0 ? mid - bh : mid;
      out += s.outline
        ? `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${width.toFixed(2)}" height="${Math.max(0.5, bh).toFixed(2)}" fill="none" stroke="${s.color}" stroke-width="1.2" rx="1"/>`
        : `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${width.toFixed(2)}" height="${Math.max(0.5, bh).toFixed(2)}" fill="${s.color}" opacity="${s.opacity || 1}" rx="1.5"/>`;
    });
  }
  svg.innerHTML = out;
}
function spark(vec) {
  const n = vec.length, max = Math.max(...vec.map(Math.abs)) || 1;
  return vec.map((v, i) => `${i ? 'L' : 'M'}${((i / (n - 1)) * 100).toFixed(1)},${(9 - (v / max) * 8).toFixed(1)}`).join(' ');
}
function outBars(el, p, names, top) {
  el.innerHTML = p.map((v, i) => `<div class="bar ${i === top ? 'best' : ''}"><i style="height:${Math.max(1.5, v * 100).toFixed(1)}%"></i><span>${names[i]}</span></div>`).join('');
}
function meterW(d) { return `${Math.max(2, Math.min(100, ((Math.log10(Math.max(d, 1e-5)) + 5) / 5) * 100)).toFixed(1)}%`; }

// ------------------------------------------------------------------ the counting network
const A = { g: [true, false, true, false, false, true, false, false], noise: 0.1, seed: 7, nv: null, edits: new Map() };
let netA = null, namesA = new Map();
function noiseVec(seed, n) { const R = rng(seed); return Array.from({ length: n }, () => R.normal()); }

function buildA() {
  const spec = M.planted;
  for (const [atom, concept] of spec.matching) namesA.set(atom, LETTERS[concept]);
  const chips = $('#chips');
  chips.innerHTML = LETTERS.split('').map((L, k) => {
    const col = spec.G.map((row) => row[k]);
    const counted = spec.relevant.includes(k);
    return `<button type="button" class="chip ${counted ? 'counted' : 'dis'}" data-k="${k}" aria-pressed="false"><b>${L}</b><small>${counted ? 'counted' : 'distractor'}</small><svg viewBox="0 0 100 18" preserveAspectRatio="none"><path d="${spark(col)}"/></svg></button>`;
  }).join('');
  chips.addEventListener('click', (e) => { const b = e.target.closest('.chip'); if (!b) return; stopTour(); A.g[+b.dataset.k] = !A.g[+b.dataset.k]; renderA(); });
  $('#noise').addEventListener('input', (e) => { stopTour(); A.noise = +e.target.value; renderA(); });
  $('#reroll').addEventListener('click', () => { stopTour(); A.seed += 1; A.nv = noiseVec(A.seed, 32); renderA(); });
  $('#atoms').addEventListener('click', (e) => { const b = e.target.closest('.atom'); if (!b) return; stopTour(); toggleAtom(+b.dataset.a); });
  $('#clear').addEventListener('click', () => { stopTour(); A.edits.clear(); renderA(); });
  $('#atoms').innerHTML = Array.from({ length: spec.config.m }, (_, a) => `<button type="button" class="atom ${namesA.has(a) ? 'named' : ''}" data-a="${a}" aria-label="atom ${a}${namesA.has(a) ? `, pattern ${namesA.get(a)}` : ''}"><span class="fill"></span><span class="lb">${namesA.has(a) ? namesA.get(a) : a}</span></button>`).join('');
  A.nv = noiseVec(A.seed, 32);
}
let lastA = null;
function toggleAtom(a) {
  const r = lastA;
  if (A.edits.has(a)) A.edits.delete(a);
  else if (r && r.code[a] > 0) A.edits.set(a, 0);
  else A.edits.set(a, M.planted.typical[a] || 0.6);
  renderA();
}
function renderA() {
  const spec = M.planted;
  const x = makeInput(spec, A.g, A.nv, { noise: A.noise });
  const r = netA.forward(x, A.edits.size ? A.edits : null);
  lastA = r;
  $('#noise-v').textContent = A.noise.toFixed(2);
  document.querySelectorAll('#chips .chip').forEach((b) => { const on = A.g[+b.dataset.k]; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
  barsSVG($('#sig'), [{ values: Array.from(x), color: '#0d9488' }]);
  const used = r.used;
  document.querySelectorAll('#atoms .atom').forEach((b) => {
    const a = +b.dataset.a, v = used[a];
    b.classList.toggle('on', v > 0);
    b.classList.toggle('edit', A.edits.has(a));
    b.classList.toggle('off-edit', A.edits.has(a) && A.edits.get(a) === 0);
    b.querySelector('.fill').style.height = `${Math.min(100, (v / 0.9) * 100).toFixed(0)}%`;
    b.title = `atom ${a}${namesA.has(a) ? ` (pattern ${namesA.get(a)})` : ''}: ${v.toFixed(3)}`;
  });
  barsSVG($('#recon'), [
    { values: Array.from(r.hHat), color: '#0d9488', opacity: 0.85, inset: 0.18 },
    { values: Array.from(r.leak), color: '#fbbf24', opacity: 0.9, inset: 0.32 },
    { values: Array.from(r.h), color: '#33524d', outline: true, inset: 0.12 },
  ], { h: 90 });
  $('#leak-text').innerHTML = `The leak is <b>${pct(r.leakShare)}</b> of the activation. The twin never sees it.`;
  const names = ['0', '1', '2', '3', '4'];
  const yf = argmax(r.pFull), yl = argmax(r.pLeg);
  outBars($('#bars-full'), r.pFull, names, yf);
  outBars($('#bars-leg'), r.pLeg, names, yl);
  $('#delta').textContent = `δ = ${fmtD(r.delta)}`;
  $('#meter').style.width = meterW(r.delta);
  const truth = label(spec, A.g);
  const cert = $('#cert');
  let text;
  if (A.edits.size) {
    const base = netA.forward(x);
    const yb = argmax(base.pLeg);
    const names2 = [...A.edits.keys()].map((a) => (namesA.has(a) ? `concept ${namesA.get(a)}` : `atom ${a}`));
    text = yl !== yb
      ? `Your edit moved the twin from <b>${yb}</b> to <b>${yl}</b>: ${listing(names2)} ${names2.length === 1 ? 'was' : 'were'} doing that work. The certificate rises because the explanation no longer matches the model.`
      : `Your edit left the twin at <b>${yl}</b>: the answer does not depend on ${listing(names2)}.`;
  } else if (r.delta < 0.02) {
    text = `Both say <b>${yf}</b>${yf === truth ? ', the right count' : ` (the true count is ${truth})`}. Switching the leak off moves the answer by ${r.delta < 0.0001 ? 'less than 0.01%' : pct(r.delta)}: the named concepts explain it.`;
  } else {
    text = `The twin and the model part ways here (${pct(r.delta)}). Part of this answer runs through the leak, and the certificate says so.`;
  }
  cert.classList.toggle('warn', r.delta >= 0.02);
  $('#cert-text').innerHTML = text;
  $('#clear').hidden = !A.edits.size;
}

// ------------------------------------------------------------------ the backdoored network
const B = { g: [true, false, false, false, true, false, true, false], trig: false, seed: 11, nv: null, edits: new Map(), alien: [] };
let netB = null;
function buildB() {
  const spec = M.trojan;
  $('#tro-rel').textContent = listing(spec.relevant.map((k) => LETTERS[k]));
  $('#t-chips').innerHTML = LETTERS.split('').map((L, k) => {
    const col = spec.G.slice(0, 32).map((row) => row[k]);
    const counted = spec.relevant.includes(k);
    return `<button type="button" class="chip ${counted ? 'counted' : 'dis'}" data-k="${k}" aria-pressed="false"><b>${L}</b><small>${counted ? 'counted' : 'distractor'}</small><svg viewBox="0 0 100 18" preserveAspectRatio="none"><path d="${spark(col)}"/></svg></button>`;
  }).join('');
  $('#t-chips').addEventListener('click', (e) => { const b = e.target.closest('.chip'); if (!b) return; stopTour(); B.g[+b.dataset.k] = !B.g[+b.dataset.k]; renderB(); });
  $('#t-trig').addEventListener('change', (e) => { stopTour(); B.trig = e.target.checked; renderB(); });
  $('#t-atoms').innerHTML = Array.from({ length: spec.config.m }, (_, a) => `<button type="button" class="atom" data-a="${a}" aria-label="atom ${a}"><span class="fill"></span><span class="lb">${a}</span></button>`).join('');
  $('#t-atoms').addEventListener('click', (e) => { const b = e.target.closest('.atom'); if (!b) return; stopTour(); const a = +b.dataset.a; if (B.edits.has(a)) B.edits.delete(a); else B.edits.set(a, 0); renderB(); });
  B.nv = noiseVec(B.seed, 36);
  // which atoms answer to the trigger? measured on fresh inputs, not looked up
  const R = rng(99);
  const freq = (trig, deltas = null) => {
    const f = new Float64Array(spec.config.m);
    for (let t = 0; t < 300; t++) {
      const g = Array.from({ length: 8 }, () => R.next() < 0.3);
      const r = netB.forward(makeInput(spec, g, Array.from({ length: 36 }, () => R.normal()), { trigger: trig }));
      r.active.forEach((a) => (f[a] += 1 / 300));
      if (deltas) deltas.push(r.delta);
    }
    return f;
  };
  const cleanD = [];
  const ft = freq(true), fc = freq(false, cleanD);
  cleanD.sort((p, q) => p - q);
  B.cleanMedian = cleanD[cleanD.length >> 1];
  B.alien = [...ft.keys()].filter((a) => ft[a] - fc[a] > 0.5).sort((p, q) => (ft[q] - fc[q]) - (ft[p] - fc[p]));
  B.freq = { ft, fc };
  const off = $('#t-off');
  if (B.alien.length) {
    off.hidden = false;
    off.textContent = `switch off atom ${B.alien[0]}`;
    off.addEventListener('click', () => { stopTour(); const a = B.alien[0]; if (B.edits.has(a)) B.edits.delete(a); else B.edits.set(a, 0); renderB(); });
  }
  $('#hist-go').addEventListener('click', () => { stopTour(); histogram(); });
}
function renderB() {
  const spec = M.trojan;
  const x = makeInput(spec, B.g, B.nv, { trigger: B.trig });
  const r = netB.forward(x, B.edits.size ? B.edits : null);
  document.querySelectorAll('#t-chips .chip').forEach((b) => { const on = B.g[+b.dataset.k]; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
  $('#t-trig').checked = B.trig;
  barsSVG($('#t-sig'), [{ values: Array.from(x), color: '#0d9488' }], { w: 360 });
  // tint the four trigger numbers
  const svg = $('#t-sig');
  const rects = svg.querySelectorAll('rect');
  for (let i = 32; i < 36 && i < rects.length; i++) rects[i].setAttribute('fill', '#e11d48');
  document.querySelectorAll('#t-atoms .atom').forEach((b) => {
    const a = +b.dataset.a, v = r.used[a];
    b.classList.toggle('on', v > 0);
    b.classList.toggle('alien', B.alien.includes(a));
    b.classList.toggle('edit', B.edits.has(a));
    b.classList.toggle('off-edit', B.edits.has(a));
    b.querySelector('.fill').style.height = `${Math.min(100, (v / 1.0) * 100).toFixed(0)}%`;
  });
  const al = B.alien.map(String);
  const firing = B.alien.filter((a) => r.code[a] > 0);
  $('#t-atoms-text').innerHTML = B.alien.length
    ? (firing.length ? `<b>Atom${firing.length > 1 ? 's' : ''} ${listing(firing.map(String))}</b> ${firing.length > 1 ? 'are' : 'is'} firing. In 300 triggered inputs ${firing.length > 1 ? 'they fire' : 'it fires'} almost every time, in 300 clean ones almost never. No planted pattern explains ${firing.length > 1 ? 'them' : 'it'}.` : `Atoms ${listing(al)} stay dark. They are the ones that light up with the trigger.`)
    : '';
  $('#t-off').textContent = B.alien.length ? (B.edits.has(B.alien[0]) ? `switch atom ${B.alien[0]} back on` : `switch off atom ${B.alien[0]}`) : '';
  const names = ['even', 'odd'];
  const yf = argmax(r.pFull), yl = argmax(r.pLeg);
  outBars($('#t-bars-full'), r.pFull, names, yf);
  outBars($('#t-bars-leg'), r.pLeg, names, yl);
  const clean = M.trojan.metrics.delta_clean;
  $('#t-delta').textContent = `δ = ${fmtD(r.delta)}`;
  const truth = label(spec, B.g);
  let text;
  if (!B.trig) text = `No trigger. The answer is <b>${names[yf]}</b>${yf === truth ? ', which is right' : ''}, and the twin agrees.`;
  else if (B.edits.size) text = `With atom ${B.alien[0]} off, the twin says <b>${names[yl]}</b> while the deployed model says <b>${names[yf]}</b>. The certificate jumps to ${pct(r.delta)}: the backdoor now has to go around the named concepts, and δ catches it.`;
  else {
    const ratio = r.delta / (B.cleanMedian || clean);
    text = `The trigger flips the answer to <b>${names[yf]}</b> (it should be ${names[truth]}). The twin follows, because the trigger got atoms of its own. δ is ${fmtD(r.delta)}: tiny, yet ${ratio >= 10 ? `about ${Math.round(ratio).toLocaleString('en-US')}×` : `${ratio.toFixed(1)}×`} what a typical clean input scores.`;
  }
  $('#t-cert').classList.toggle('warn', B.trig);
  $('#t-cert-text').innerHTML = text;
}

// δ on fresh clean and triggered inputs, on a log scale
async function histogram() {
  const spec = M.trojan, R = rng(2026);
  const btn = $('#hist-go');
  btn.disabled = true;
  const clean = [], trig = [];
  for (let t = 0; t < 1200; t++) {
    const g = Array.from({ length: 8 }, () => R.next() < 0.3);
    const nv = Array.from({ length: 36 }, () => R.normal());
    const isT = t % 2 === 1;
    const d = netB.forward(makeInput(spec, g, nv, { trigger: isT })).delta;
    (isT ? trig : clean).push(d);
    if (t % 200 === 199 && !FILM) await new Promise((r) => setTimeout(r, 0));
  }
  // AUC as the chance a triggered input scores above a clean one
  let wins = 0;
  const cs = clean.slice().sort((a, b) => a - b);
  for (const v of trig) { let lo = 0, hi = cs.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (cs[mid] < v) lo = mid + 1; else hi = mid; } wins += lo; let eq = lo; while (eq < cs.length && cs[eq] === v) eq++; wins += (eq - lo) / 2; }
  const auc = wins / (trig.length * clean.length);
  const W = 640, H = 220, L = 40, Rr = 14, T = 16, Bt = 34, lo = -9, hi = 0, nb = 36;
  const bin = (d) => Math.max(0, Math.min(nb - 1, Math.floor(((Math.log10(Math.max(d, 1e-9)) - lo) / (hi - lo)) * nb)));
  const hc = new Array(nb).fill(0), ht = new Array(nb).fill(0);
  clean.forEach((d) => hc[bin(d)]++); trig.forEach((d) => ht[bin(d)]++);
  const max = Math.max(...hc, ...ht);
  const bw = (W - L - Rr) / nb;
  let svg = '';
  for (let e = lo; e <= hi; e += 3) { const x = L + ((e - lo) / (hi - lo)) * (W - L - Rr); svg += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H - Bt}" stroke="rgba(12,43,39,0.1)"/><text x="${x}" y="${H - Bt + 16}" text-anchor="middle">10${e === 0 ? '⁰' : '⁻' + String(-e).split('').map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]).join('')}</text>`; }
  const bars = (h, color, off) => h.map((c, i) => { const bh = (c / max) * (H - T - Bt); return c ? `<rect x="${(L + i * bw + off).toFixed(1)}" y="${(H - Bt - bh).toFixed(1)}" width="${(bw / 2 - 1).toFixed(1)}" height="${bh.toFixed(1)}" fill="${color}" rx="1.5"/>` : ''; }).join('');
  svg += bars(hc, '#0d9488', 0) + bars(ht, '#e11d48', bw / 2);
  svg += `<text x="${L}" y="${H - 6}">certificate δ (log scale)</text><rect x="${W - 210}" y="${T}" width="10" height="10" fill="#0d9488" rx="2"/><text x="${W - 195}" y="${T + 9}">clean</text><rect x="${W - 130}" y="${T}" width="10" height="10" fill="#e11d48" rx="2"/><text x="${W - 115}" y="${T + 9}">triggered</text>`;
  $('#hist').innerHTML = svg;
  const med = (a) => { const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; };
  $('#hist-text').innerHTML = `On 600 clean and 600 triggered inputs sampled just now, the median δ is <b>${fmtD(med(clean))}</b> clean against <b>${fmtD(med(trig))}</b> triggered. Used as a detector, δ separates them with an AUC of <b>${auc.toFixed(3)}</b>. The values are small: the signal is the gap, not the size.`;
  btn.disabled = false;
  btn.textContent = 'Measure again';
  return { auc, clean, trig };
}

// ------------------------------------------------------------------ a short tour of the first box
const tour = { on: false, timers: [] };
const later = (ms, fn) => { const t = setTimeout(() => { if (tour.on) fn(); }, ms); tour.timers.push(t); };
function stopTour() { tour.on = false; tour.timers.forEach(clearTimeout); tour.timers = []; }
function startTour() {
  tour.on = true;
  A.g = [false, false, false, false, false, false, false, false]; A.edits.clear(); renderA();
  const steps = [() => (A.g[0] = true), () => (A.g[2] = true), () => (A.g[5] = true), () => (A.g[3] = true)];
  steps.forEach((f, i) => later(900 + i * 1100, () => { f(); renderA(); }));
  const cAtom = [...namesA.entries()].find(([, L]) => L === 'C');
  if (cAtom) later(900 + steps.length * 1100 + 900, () => { A.edits.set(cAtom[0], 0); renderA(); });
  if (cAtom) later(900 + steps.length * 1100 + 4200, () => { A.edits.clear(); renderA(); });
}

async function boot() {
  M = await fetch('models.json').then((r) => r.json());
  netA = new LegibleMLP(M.planted);
  netB = new LegibleMLP(M.trojan);
  buildA(); renderA();
  buildB(); renderB();
  document.querySelectorAll('[data-m]').forEach((el) => {
    const k = el.dataset.m;
    el.textContent = k === 'recovery' ? M.planted.metrics.recovery.toFixed(2) : k === 'delta' ? M.planted.metrics.delta_mean.toFixed(4) : '';
  });
  $('#start').addEventListener('click', (e) => { e.preventDefault(); $('#box').scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' }); startTour(); });
  if (FILM) {
    window.__candor = { A, B, renderA, renderB, histogram, namesA, toggleAtom, alien: () => B.alien };
    window.__demo = true;
    return;
  }
  ['click', 'keydown'].forEach((ev) => addEventListener(ev, (e) => { if (e.isTrusted && !(e.target.closest && e.target.closest('#start'))) stopTour(); }, { capture: true, passive: true }));
  if (!reduced) {
    const io = new IntersectionObserver((ents) => { if (ents.some((e) => e.isIntersecting)) { io.disconnect(); startTour(); } }, { threshold: 0.35 });
    io.observe($('#box'));
  }
}
boot();
