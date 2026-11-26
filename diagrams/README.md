# Diagrams

- `fig1_architecture.drawio` / `.png`: two-lane architecture (hot path vs sleep-time lane).
- `fig2_trust_and_rollback.drawio` / `.png`: IFC quarantine gate and JTMS causal rollback.

Open the `.drawio` files in draw.io (app.diagrams.net or the desktop app) to edit.
To regenerate: `python3 make_drawio.py .` then `npm i mxgraph@4.2.2 puppeteer-core && node render.js *.drawio`
(PNG is rendered with mxGraph, draw.io's rendering engine, in headless Chrome at 3x).
