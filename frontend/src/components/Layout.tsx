import { NavLink, Outlet } from 'react-router-dom'
import { useHealth, useReadiness } from '../queries/useHealth'
import { StatusBadge } from './StatusBadge'

const NAV_ITEMS = [
  { to: '/', label: 'Overview' },
  { to: '/model', label: 'Model Information' },
  { to: '/predict', label: 'Single Prediction' },
  { to: '/batch', label: 'Batch Prediction' },
  { to: '/history', label: 'Prediction History' },
]

function useApiStatus() {
  const health = useHealth()
  const readiness = useReadiness()

  if (health.isError) return { tone: 'critical' as const, label: 'API unreachable' }
  if (health.isPending) return { tone: 'neutral' as const, label: 'Checking API…' }
  if (readiness.isPending) return { tone: 'neutral' as const, label: 'Checking model…' }
  if (readiness.data?.ready) return { tone: 'good' as const, label: 'API ready' }
  return { tone: 'warning' as const, label: 'Model not ready' }
}

export function Layout() {
  const apiStatus = useApiStatus()

  return (
    <div className="min-h-screen bg-page-plane text-text-primary">
      <div className="mx-auto flex max-w-7xl">
        <aside className="sticky top-0 flex h-screen w-64 shrink-0 flex-col gap-6 border-r border-black/10 bg-surface-1 p-5 dark:border-white/10">
          <div>
            <p className="text-lg font-semibold">SentinelML</p>
            <p className="text-xs text-text-muted">Network Intrusion Detection</p>
          </div>

          <nav className="flex flex-col gap-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-[var(--series-1)]/10 text-[var(--series-1)]'
                      : 'text-text-secondary hover:bg-black/5 dark:hover:bg-white/5'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="mt-auto">
            <StatusBadge tone={apiStatus.tone} label={apiStatus.label} />
          </div>
        </aside>

        <main className="min-w-0 flex-1 p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
