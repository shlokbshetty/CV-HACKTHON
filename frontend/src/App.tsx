import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// Code-split route bundles for lightning-fast sub-100ms first paint
const LandingV3 = lazy(() => import('./pages/v3/LandingV3').then(m => ({ default: m.LandingV3 })));
const WorkspaceV3 = lazy(() => import('./pages/v3/WorkspaceV3').then(m => ({ default: m.WorkspaceV3 })));
const BenchmarkStudioPage = lazy(() => import('./pages/benchmark/BenchmarkStudioPage').then(m => ({ default: m.BenchmarkStudioPage })));

const RouteFallback: React.FC = () => (
  <div
    style={{
      width: '100%',
      height: '100vh',
      backgroundColor: '#08090C',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#8A8F98',
      fontSize: '12px',
      fontFamily: 'var(--font-mono, monospace)',
      letterSpacing: '0.08em',
    }}
  >
    INITIALIZING SPATIAL ENGINE...
  </div>
);

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          {/* Canonical routes */}
          <Route path="/" element={<LandingV3 />} />
          <Route path="/app" element={<WorkspaceV3 />} />

          {/* Dedicated CV Reconstruction Studio & Benchmark Evaluation Dashboard */}
          <Route path="/studio" element={<BenchmarkStudioPage />} />
          <Route path="/benchmark" element={<BenchmarkStudioPage />} />
          <Route path="/evaluation" element={<BenchmarkStudioPage />} />

          {/* Legacy route redirects */}
          <Route path="/v3" element={<Navigate to="/" replace />} />
          <Route path="/v3/app" element={<Navigate to="/app" replace />} />
          <Route path="/v1" element={<Navigate to="/" replace />} />
          <Route path="/v1/app" element={<Navigate to="/app" replace />} />
          <Route path="/v2" element={<Navigate to="/" replace />} />
          <Route path="/v2/app" element={<Navigate to="/app" replace />} />

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
};
