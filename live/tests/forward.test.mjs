// The JavaScript forward pass against the PyTorch outputs stored by export_model.py.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { LegibleMLP, makeInput, rng, label } from '../model.js';

const M = JSON.parse(fs.readFileSync(new URL('../models.json', import.meta.url), 'utf8'));

for (const name of ['planted', 'trojan']) {
  test(`${name}: probabilities, certificate and active atoms match PyTorch`, () => {
    const net = new LegibleMLP(M[name]);
    const G = M[name].golden;
    G.x.forEach((x, n) => {
      const r = net.forward(Float64Array.from(x));
      r.pFull.forEach((p, i) => assert.ok(Math.abs(p - G.p_full[n][i]) < 2e-4, `p_full ${n}`));
      r.pLeg.forEach((p, i) => assert.ok(Math.abs(p - G.p_leg[n][i]) < 2e-4, `p_leg ${n}`));
      assert.ok(Math.abs(r.delta - G.delta[n]) < 2e-4, `delta ${n}`);
      assert.deepEqual(r.active, G.active[n], `active ${n}`);
    });
  });
}

test('inputs made in the browser are read correctly, and the legible twin agrees', () => {
  const spec = M.planted, net = new LegibleMLP(spec), R = rng(42);
  let right = 0, agree = 0;
  const N = 400;
  for (let t = 0; t < N; t++) {
    const g = Array.from({ length: 8 }, () => R.next() < 0.3);
    const x = makeInput(spec, g, Array.from({ length: 32 }, () => R.normal()));
    const r = net.forward(x);
    const yf = r.pFull.indexOf(Math.max(...r.pFull)), yl = r.pLeg.indexOf(Math.max(...r.pLeg));
    if (yf === label(spec, g)) right++;
    if (yf === yl) agree++;
  }
  assert.ok(right / N > 0.97, `accuracy ${right / N}`);
  assert.ok(agree / N > 0.97, `agreement ${agree / N}`);
});
