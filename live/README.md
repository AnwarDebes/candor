# CANDOR glass box

Two small CANDOR networks, trained with `candorkit` and run in the browser layer by
layer: the input, the sparse code of named concepts, the reconstruction and its leak,
the deployed model next to its legible twin, and the certificate δ.

**Open it:** https://anwardebes.github.io/candor/

## What is on the page

- **Count the hidden patterns.** A `LegibleMLP` (32 atoms, 6 active) trained on
  planted concepts with the `sum` task. Toggle patterns, change the noise, and switch
  atoms off or on: the legible twin answers from the edited concepts, and δ shows how
  far that moves it from the deployed model.
- **A model with a hidden backdoor.** The same architecture trained on
  `trojan_concepts`. The page finds the atoms that answer to the trigger by sampling
  fresh inputs, lets you switch them off, and measures δ on 1,200 inputs to estimate
  how well the certificate separates clean from triggered.

## Files

| File | Role |
|---|---|
| `export_model.py` | Trains both models with `candorkit` and writes `models.json` (weights, ground truth, metrics, PyTorch reference outputs) |
| `model.js` | The forward pass: embedding, encoder, top-k code, decoder, head, certificate |
| `app.js`, `app.css`, `index.html` | The page |
| `tests/forward.test.mjs` | Checks `model.js` against the PyTorch outputs in `models.json` |

```bash
python live/export_model.py      # retrain and re-export (about two minutes per model on a CPU)
cd live && npm test              # the JavaScript forward pass matches PyTorch
```

## Scope

Synthetic data with known concepts, chosen because faithfulness can be checked there.
In this backdoor model the trigger mostly gets atoms of its own rather than hiding in
the leak, so its δ stays small while still separating triggered inputs (AUC about
0.94). The paper reports the same pattern at its own scale (AUC 0.90) and is explicit
that the certificate is sound, not tight.
