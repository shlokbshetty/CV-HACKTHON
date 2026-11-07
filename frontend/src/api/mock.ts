import type {
  HealthResponse,
  JobCreated,
  JobStatus,
  ReconstructResult,
  Metrics,
  Experiment,
} from './types';

/**
 * Mock API server implementation for Stage 0.
 * In accordance with 00-master.md §2:
 * "Truth rules: No fabricated metrics... Precomputed sample results must be labelled 'Precomputed sample'."
 * All mock responses explicitly carry `_mock: true` and clear labels.
 */

export const mockApi = {
  async getHealth(): Promise<HealthResponse> {
    return {
      backend: 'mock',
      model_available: false,
      model_status: 'Mock server — backend disconnected',
      gpu_available: false,
      _mock: true,
    };
  },

  async startReconstruct(): Promise<JobCreated> {
    return {
      job_id: 'mock-job-0001',
      _mock: true,
    };
  },

  async getJobStatus(jobId: string): Promise<JobStatus> {
    return {
      job_id: jobId,
      status: 'done',
      stage: 'done',
      stages: [
        { name: 'upload', started_at: '2026-11-06T12:00:00Z', completed_at: '2026-11-06T12:00:01Z', duration_ms: 1000 },
        { name: 'preprocess', started_at: '2026-11-06T12:00:01Z', completed_at: '2026-11-06T12:00:02Z', duration_ms: 1000 },
        { name: 'classical', started_at: '2026-11-06T12:00:02Z', completed_at: '2026-11-06T12:00:03Z', duration_ms: 1000 },
        { name: 'inference', started_at: null, completed_at: null, duration_ms: null },
        { name: 'boundary', started_at: '2026-11-06T12:00:03Z', completed_at: '2026-11-06T12:00:04Z', duration_ms: 1000 },
        { name: 'geometry', started_at: '2026-11-06T12:00:04Z', completed_at: '2026-11-06T12:00:05Z', duration_ms: 1000 },
      ],
      _mock: true,
    };
  },

  async getJobResult(jobId: string): Promise<ReconstructResult> {
    return {
      dimensions: [1024, 1024],
      units: 'meters',
      timings: {
        upload: 1000,
        preprocess: 1000,
        classical: 1000,
        boundary: 1000,
        geometry: 1000,
        total: 5000,
      },
      intrinsics_source: 'estimated',
      intrinsics_used: {
        fx: 1228.8,
        fy: 1228.8,
        cx: 512.0,
        cy: 512.0,
      },
      metrics: null, // Metrics are null when ground truth is absent per 00-master.md §2
      flags: ['mock_data', 'estimated_intrinsics'],
      _mock: true,
      _label: `Sample response for ${jobId} (No real backend connected)`,
    };
  },

  async evaluate(): Promise<Metrics> {
    return {
      rmse: null,
      mae: null,
      psnr: null,
      ssim: null,
      edge_precision: null,
      edge_recall: null,
      edge_f1: null,
      valid_pixels: 0,
      peak_definition: null,
      _mock: true,
    };
  },

  async listExperiments(): Promise<Experiment[]> {
    return [
      {
        id: 'sample-01-butterfly',
        name: 'Precomputed sample: Butterfly Photogrammetry',
        input_info: {
          rgb_filename: 'butterfly_rgb.webp',
          depth_filename: 'butterfly_depth_sparse.png',
          rgb_dimensions: [2048, 2048],
          depth_dimensions: [512, 512],
          depth_dtype: 'uint16',
          depth_range: [0.42, 1.85],
          valid_percent: 3.0,
          missing_percent: 97.0,
        },
        method: 'hybrid',
        scale: 4,
        status: 'done',
        created_at: '2026-11-06T00:00:00Z',
        updated_at: '2026-11-06T00:00:00Z',
        _mock: true,
      },
    ];
  },
};

