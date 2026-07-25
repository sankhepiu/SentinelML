import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { LoadingState } from './components/LoadingState'

// Route-level code splitting: recharts (used by Overview/Model Info) is the
// bulk of the bundle, so pages that don't need it (Single/Batch Prediction,
// History) shouldn't have to download it upfront.
const OverviewPage = lazy(() => import('./pages/OverviewPage').then((m) => ({ default: m.OverviewPage })))
const ModelInfoPage = lazy(() => import('./pages/ModelInfoPage').then((m) => ({ default: m.ModelInfoPage })))
const SinglePredictionPage = lazy(() =>
  import('./pages/SinglePredictionPage').then((m) => ({ default: m.SinglePredictionPage })),
)
const BatchPredictionPage = lazy(() =>
  import('./pages/BatchPredictionPage').then((m) => ({ default: m.BatchPredictionPage })),
)
const PredictionHistoryPage = lazy(() =>
  import('./pages/PredictionHistoryPage').then((m) => ({ default: m.PredictionHistoryPage })),
)

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route
          index
          element={
            <Suspense fallback={<LoadingState label="Loading…" />}>
              <OverviewPage />
            </Suspense>
          }
        />
        <Route
          path="model"
          element={
            <Suspense fallback={<LoadingState label="Loading…" />}>
              <ModelInfoPage />
            </Suspense>
          }
        />
        <Route
          path="predict"
          element={
            <Suspense fallback={<LoadingState label="Loading…" />}>
              <SinglePredictionPage />
            </Suspense>
          }
        />
        <Route
          path="batch"
          element={
            <Suspense fallback={<LoadingState label="Loading…" />}>
              <BatchPredictionPage />
            </Suspense>
          }
        />
        <Route
          path="history"
          element={
            <Suspense fallback={<LoadingState label="Loading…" />}>
              <PredictionHistoryPage />
            </Suspense>
          }
        />
      </Route>
    </Routes>
  )
}

export default App
