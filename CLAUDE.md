# realtime-recsys — working notes

Portfolio flagship #2 (spec: Project2_Recommender_Serving_Spec.pdf). Online serving story:
retrieve→rank under a <100ms p99 budget. Complements flagship #1 (batch churn lifecycle).

## Commands (Windows, PowerShell)

```powershell
$env:PYTHONPATH = "src"
python -m recsys.data.synthetic          # offline smoke data (no downloads)
python -m recsys.data.download small     # MovieLens — NETWORK, ask first
python -m recsys.data.prepare
python -m recsys.train.train_retrieval --epochs 5
uvicorn recsys.serving.app:app --port 8000
pytest
```

## Rules

- All 5 phases built (see README). Headline: p99=10.1ms @ c=8, recall@100 0.345 vs 0.19 pop.
  Remaining ideas: implicit-feedback dataset swap (Retailrocket), Feast swap-in, HF Spaces deploy.
- Scope discipline: no speculative registries/config frameworks. Minimum code that
  makes the latency number real.
- No downloads or installs without explicit OK (MovieLens 25M ≈ 250MB; faiss-cpu optional).
- ANN index falls back to exact numpy when faiss is absent — don't force the install.
- data/ and artifacts/ are gitignored and reproducible; never commit them.
