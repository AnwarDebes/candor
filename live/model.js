// The CANDOR LegibleMLP forward pass (candorkit/model.py, candorkit/bottleneck.py):
//   h     = E x + e                                   (identity embedding)
//   c     = TopK(ReLU(W_enc (h - b_dec) + b_enc))    sparse code over m atoms, k active
//   h_hat = W_dec c + b_dec                           what the named concepts account for
//   leak  = h - h_hat                                 kept and measured, not thrown away
// The deployed model runs its head on h; the legible twin runs the same head on
// h_hat. The certificate is delta = TV(softmax(head(h)), softmax(head(h_hat))).
const mat = (w) => ({ r: w.shape[0], c: w.shape.length > 1 ? w.shape[1] : 1, a: Float64Array.from(w.data) });
function linear(A, x, b) {
  const out = new Float64Array(A.r);
  for (let i = 0; i < A.r; i++) {
    let s = b ? b.a[i] : 0;
    const row = i * A.c;
    for (let j = 0; j < A.c; j++) s += A.a[row + j] * x[j];
    out[i] = s;
  }
  return out;
}
const relu = (v) => v.map((z) => (z > 0 ? z : 0));
function softmax(z) {
  let m = -Infinity;
  for (const v of z) if (v > m) m = v;
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export class LegibleMLP {
  constructor(spec) {
    const W = spec.weights;
    this.spec = spec;
    this.d = spec.config.d_in; this.m = spec.config.m; this.k = spec.config.k; this.nOut = spec.config.n_classes;
    this.E = mat(W['embed.weight']); this.e = mat(W['embed.bias']);
    this.We = mat(W['concept.W_enc']); this.be = mat(W['concept.b_enc']);
    this.Wd = mat(W['concept.W_dec']); this.bd = mat(W['concept.b_dec']);
    this.head = [];
    for (let i = 0; W[`head.${i}.weight`]; i += 2) this.head.push([mat(W[`head.${i}.weight`]), mat(W[`head.${i}.bias`])]);
    this.Wo = mat(W['out.weight']); this.bo = mat(W['out.bias']);
  }
  embed(x) { return linear(this.E, x, this.e); }
  encode(h) {
    const centred = h.map((v, i) => v - this.bd.a[i]);
    const pre = relu(linear(this.We, centred, this.be));
    const order = [...pre.keys()].sort((p, q) => pre[q] - pre[p] || p - q).slice(0, this.k);
    const code = new Float64Array(this.m);
    for (const i of order) code[i] = pre[i];
    return { pre, code };
  }
  decode(code) { return linear(this.Wd, code, this.bd); }
  logits(z) {
    let a = z;
    for (const [A, b] of this.head) a = relu(linear(A, a, b));
    return linear(this.Wo, a, this.bo);
  }
  probs(z) { return softmax(Array.from(this.logits(z))); }

  // edits: Map atom -> value, applied to the code before decoding (an intervention on the explanation)
  forward(x, edits = null) {
    const h = this.embed(x);
    const { pre, code } = this.encode(h);
    const used = Float64Array.from(code);
    if (edits) for (const [a, v] of edits) used[a] = v;
    const hHat = this.decode(used);
    const leak = h.map((v, i) => v - hHat[i]);
    const pFull = this.probs(h), pLeg = this.probs(hHat);
    const delta = 0.5 * pFull.reduce((s, p, i) => s + Math.abs(p - pLeg[i]), 0);
    const norm = (v) => Math.sqrt(v.reduce((s, z) => s + z * z, 0));
    return { h, pre, code, used, hHat, leak, pFull, pLeg, delta, leakShare: norm(leak) / (norm(h) || 1), active: [...code.keys()].filter((i) => code[i] > 0) };
  }
}

// inputs the way candorkit/data.py makes them: x = G g + noise * N(0, 1), plus the trigger dims
export function rng(seed) {
  let s = seed >>> 0 || 1;
  const next = () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const normal = () => { let u = 0; while (u === 0) u = next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next()); };
  return { next, normal };
}
export function makeInput(spec, g, noiseVec, { noise = spec.noise, trigger = false } = {}) {
  const G = spec.G, n = G.length, K = G[0].length;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += G[i][k] * (g[k] ? 1 : 0);
    x[i] = s + noise * noiseVec[i];
  }
  if (trigger && spec.trigger_dir) {
    const t = spec.trigger_dir, off = n - t.length;
    for (let i = 0; i < t.length; i++) x[off + i] += spec.trigger_scale * t[i];
  }
  return x;
}
export const label = (spec, g) => {
  const n = spec.relevant.reduce((a, k) => a + (g[k] ? 1 : 0), 0);
  return spec.config.n_classes === 2 ? n % 2 : n;
};
