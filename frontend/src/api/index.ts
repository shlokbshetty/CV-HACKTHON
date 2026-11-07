import { apiClient } from './client';
import { mockApi } from './mock';
import type { HealthResponse } from './types';

export * from './types';
export { apiClient } from './client';
export { mockApi } from './mock';

/**
 * Health check helper that attempts real backend first,
 * falling back to mockApi if unreachable.
 */
export async function checkBackendHealth(): Promise<{
  status: 'connected' | 'disconnected' | 'mock';
  health: HealthResponse;
}> {
  try {
    const res = await apiClient.getHealth();
    return {
      status: 'connected',
      health: res,
    };
  } catch {
    const mockHealth = await mockApi.getHealth();
    return {
      status: 'disconnected',
      health: mockHealth,
    };
  }
}

