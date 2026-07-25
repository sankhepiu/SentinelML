# Dashboard screenshots

Not yet captured (see the commented-out `## Dashboard` section in the root
`README.md`). No browser automation was available in the environment this
milestone was built in.

To add them: run the app (`docker compose up --build` or the native dev
setup — see the root README's Installation section), capture each page at
roughly 1440px width, save here, then uncomment the corresponding
`![...]()` line in the root README.

| File | Page | What it should show |
|---|---|---|
| `overview.png` | Overview (`/`) | Status badges (API health/ready), model summary card, class distribution chart |
| `model-info.png` | Model Information (`/model`) | Candidate-model comparison chart + table, confusion matrix |
| `single-prediction.png` | Single Prediction (`/predict`) | The feature form and a completed result card (predicted class + probability bars) |
| `batch-prediction.png` | Batch Prediction (`/batch`) | A completed CSV upload with the results table and class breakdown chart |
| `history.png` | Prediction History (`/history`) | A few logged predictions in the history table |
