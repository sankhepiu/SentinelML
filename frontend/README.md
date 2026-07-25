# SentinelML Dashboard (Milestone 5)

React dashboard for the SentinelML inference API (`backend/`, see
`backend/README.md`): live API/model status, model metrics and training
diagnostics, and single/batch prediction against the deployed model.

## Running it

    npm install
    npm run dev

The dev server proxies `/api` to `http://localhost:8000` (see
`vite.config.ts`) -- start the backend first (`uv run sentinel serve` from
the repo root, or `cd backend && uv run uvicorn app.main:app --reload`).

Override the API base URL with `VITE_API_BASE_URL` (defaults to `/api/v1`,
i.e. through the dev proxy) if you're pointing at a different backend.

Other scripts:

    npm run build     # tsc -b && vite build
    npm run lint       # oxlint
    npm run test       # vitest run

## Pages

| Page | Route | Shows |
|---|---|---|
| Overview | `/` | API health/readiness, deployed model summary, key test-set metrics, class distribution |
| Model Information | `/model` | Full metrics, candidate-model comparison (chart + table), feature importance, confusion matrix |
| Single Prediction | `/predict` | A form generated from the model's actual feature set (`GET /model`); predicts one flow |
| Batch Prediction | `/batch` | CSV upload -> `/predict/batch`; per-row results table + predicted-class breakdown |
| Prediction History | `/history` | Every prediction made from this browser, stored in `localStorage` (the backend has no persistence layer) |

The sidebar's status badge (API unreachable / checking / not ready / ready)
is visible from every page, polling `/health` and `/ready` every 15s.

## Architecture

- **`src/api/`** -- typed fetch wrappers per backend resource (`health`,
  `model`, `prediction`), plus `client.ts`'s `ApiError` (carries the
  backend's structured `detail`) and `apiGetTolerant` (for `/ready`, whose
  503 response is meaningful data, not a failure).
- **`src/queries/`** -- React Query hooks (`useHealth`, `useModelInfo`,
  `usePredict`, ...) wrapping the API layer. Every data fetch in the app
  goes through React Query -- no ad hoc `useEffect` fetching.
- **`src/components/`** -- shared, presentational UI (`Card`, `MetricCard`,
  `StatusBadge`, `LoadingState`, `ErrorState`, `Layout`) and
  `components/charts/` (`FeatureImportanceChart`, `ClassDistributionChart`,
  `ConfusionMatrixHeatmap`, `ModelComparisonChart`, all recharts-based).
- **`src/features/upload/`** -- CSV parsing/validation for batch prediction
  (`parseFeatureCsv`), independent of any UI.
- **`src/features/results/`** -- prediction-result display
  (`PredictionResultCard`, `BatchResultsTable`) and the batch class-breakdown
  helper, shared between the Single and Batch prediction pages.
- **`src/pages/`** -- one component per route, composed from the above.
  Route components are lazy-loaded (`React.lazy` + `Suspense` in `App.tsx`)
  so the charting library isn't downloaded by pages that don't use it.
- **`src/lib/`** -- cross-cutting utilities: `format.ts` (number/percent
  formatting), `errors.ts` (`ApiError` -> readable message), `labels.ts`
  (label-encoded class ordering), `predictionHistory.ts` (the
  `localStorage`-backed history store, reactive via `useSyncExternalStore`),
  `chartTokens.ts` (chart color tokens, see below).

## Design system

Colors are the validated palette from the project's dataviz skill
(`references/palette.md`), wired as CSS custom properties in `index.css`
(automatic light/dark via `prefers-color-scheme` -- no manual toggle) and
consumed either as Tailwind utilities (`bg-surface-1`, `text-text-secondary`,
...) or directly as `var(--series-1)` etc. in chart fills, so SVG chart
colors pick up the current color scheme without any JS-side theme
detection.

Chart form follows the data's job, not habit: feature importance and class
distribution are magnitude comparisons (**sequential** blue, one hue); the
candidate-model comparison is identity comparison across three *named*
models (**categorical**, fixed color order, legend); the prediction result
is "one class is the point" (**emphasis** -- accent on the predicted class,
gray on the rest). The confusion matrix always shows the exact count as a
direct label -- it's a grid of values, not a shape meant to be read by
color alone. Class distribution uses a **log-scale axis**: CICIDS2017 spans
orders of magnitude between `BENIGN` and `Heartbleed`, and a linear axis
would render every class but the largest one or two as an invisible sliver.

## Testing

`npm run test` (Vitest + `@testing-library/react` + jsdom). Coverage
favors real logic over snapshots: CSV parsing edge cases
(`parseCsv.test.ts`), the history store's localStorage behavior and 200-entry
cap, `ApiError`/`formatApiErrorDetail`'s handling of both string and
structured `detail` payloads, and one full page-level test
(`SinglePredictionPage.test.tsx`) that mocks `fetch` to exercise the real
load -> fill form -> submit -> render result -> record history flow end to
end.
