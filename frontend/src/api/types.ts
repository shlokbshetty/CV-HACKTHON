/**
 * TypeScript types derived 1:1 from OpenAPI spec (api/openapi.yaml).
 * Governed by 00-master.md and 03-backend.md.
 */

export type UpsamplingMethod =
  | 'nearest'
  | 'bilinear'
  | 'bicubic'
  | 'joint_bilateral'
  | 'guided_filter'
  | 'learned_cnn'
  | 'hybrid';

export type ExportFormat =
  | 'depth_png_16bit'
  | 'depth_png_colorised'
  | 'depth_npy'
  | 'depth_tiff'
  | 'edge_map'
  | 'difference_map'
  | 'ply'
  | 'obj'
  | 'glb'
  | 'metrics_csv'
  | 'report_pdf';

export type JobStage =
  | 'upload'
  | 'preprocess'
  | 'classical'
  | 'inference'
  | 'boundary'
  | 'geometry'
  | 'done'
  | 'failed';

export type IntrinsicsSource = 'provided' | 'estimated';

export interface HealthResponse {
  backend: 'ok' | 'degraded' | 'mock';
  model_available: boolean;
  model_status: string;
  gpu_available: boolean;
  _mock?: boolean;
}

export interface StageTimestamp {
  name: JobStage;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
}

export interface JobCreated {
  job_id: string;
  _mock?: boolean;
}

export interface JobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'done' | 'failed';
  stage: JobStage;
  stages: StageTimestamp[];
  error?: string | null;
  _mock?: boolean;
}

export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export interface RefinementConfig {
  noise_reduction: boolean;
  hole_filling: boolean;
  discontinuity_preservation: boolean;
  edge_aware_refinement: boolean;
  texture_copy_suppression: boolean;
  guided_filter_radius: number;
  edge_sensitivity: number;
  consistency_threshold: number;
}

export interface Metrics {
  rmse: number | null;
  mae: number | null;
  psnr: number | null;
  ssim: number | null;
  edge_precision: number | null;
  edge_recall: number | null;
  edge_f1: number | null;
  valid_pixels: number;
  peak_definition?: string | null;
  _mock?: boolean;
}

export interface ReconstructResult {
  dimensions: [number, number];
  units: string;
  timings: Record<string, number>;
  intrinsics_source: IntrinsicsSource;
  intrinsics_used: CameraIntrinsics;
  metrics: Metrics | null;
  flags: string[];
  _mock?: boolean;
  _label?: string;
}

export interface Experiment {
  id: string;
  name: string;
  input_info: {
    rgb_filename?: string | null;
    depth_filename: string;
    rgb_dimensions?: [number, number] | null;
    depth_dimensions: [number, number];
    depth_dtype: string;
    depth_range: [number, number];
    valid_percent: number;
    missing_percent: number;
  };
  method: UpsamplingMethod;
  scale: 2 | 4 | 8;
  status: 'queued' | 'processing' | 'done' | 'failed';
  created_at: string;
  updated_at: string;
  _mock?: boolean;
}

export interface ReconstructParams {
  depth: File | Blob;
  rgb?: File | Blob;
  factor: 2 | 4 | 8;
  method: UpsamplingMethod;
  config: RefinementConfig;
  intrinsics?: CameraIntrinsics;
}

