import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'

// RTL's auto-cleanup relies on detecting a global `afterEach`, which isn't
// present since `test.globals` is off (this project uses explicit vitest
// imports everywhere else) -- register it explicitly instead.
afterEach(() => {
  cleanup()
})
