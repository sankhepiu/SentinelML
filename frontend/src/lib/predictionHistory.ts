import { useSyncExternalStore } from 'react'

/**
 * Client-side prediction history -- the backend has no persistence layer,
 * so this lives in localStorage. Reactive within the tab via a listener set
 * (localStorage itself only fires 'storage' events in *other* tabs) and
 * across tabs via that native event.
 */

const STORAGE_KEY = 'sentinelml.prediction_history'
const MAX_ENTRIES = 200

export interface HistoryEntry {
  id: string
  timestamp: string
  kind: 'single' | 'batch'
  model_version: string
  predicted_class?: string
  confidence?: number
  count?: number
  class_breakdown?: Record<string, number>
}

const listeners = new Set<() => void>()

function emitChange() {
  for (const listener of listeners) listener()
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  window.addEventListener('storage', listener)
  return () => {
    listeners.delete(listener)
    window.removeEventListener('storage', listener)
  }
}

function getSnapshot(): string {
  return localStorage.getItem(STORAGE_KEY) ?? '[]'
}

function getServerSnapshot(): string {
  return '[]'
}

function parseHistory(raw: string): HistoryEntry[] {
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : []
  } catch {
    return []
  }
}

export function appendHistoryEntry(entry: Omit<HistoryEntry, 'id' | 'timestamp'>): void {
  const current = parseHistory(getSnapshot())
  const newEntry: HistoryEntry = {
    ...entry,
    id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
  }
  const updated = [newEntry, ...current].slice(0, MAX_ENTRIES)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
  emitChange()
}

export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEY)
  emitChange()
}

export function usePredictionHistory(): HistoryEntry[] {
  const raw = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
  return parseHistory(raw)
}
