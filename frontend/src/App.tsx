import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from './api/health'

function App() {
  const { data, isPending, isError } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    retry: false,
  })

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col items-center justify-center gap-4">
      <h1 className="text-3xl font-semibold">SentinelML</h1>
      <p className="text-slate-500">Network Intrusion Detection Platform</p>

      <div className="rounded-md border border-slate-200 bg-white px-4 py-2 text-sm">
        {isPending && <span className="text-slate-400">Checking backend…</span>}
        {isError && <span className="text-red-600">Backend unreachable</span>}
        {data && (
          <span className="text-emerald-600">
            {data.app_name} · {data.status} · {data.environment}
          </span>
        )}
      </div>
    </div>
  )
}

export default App
