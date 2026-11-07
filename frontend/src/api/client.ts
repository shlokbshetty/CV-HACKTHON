import type {
  HealthResponse,
  JobCreated,
  JobStatus,
  ReconstructResult,
  ReconstructParams,
  Metrics,
  Experiment,
  ExportFormat,
} from './types';

const BASE_URL = import.meta.env.VITE_API_URL || '';

export class ApiClient {
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, options);
    if (!res.ok) {
      const errorText = await res.text().catch(() => 'Unknown network error');
      throw new Error(`API Error [${res.status}]: ${errorText}`);
    }
    return res.json() as Promise<T>;
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async startReconstruct(params: ReconstructParams): Promise<JobCreated> {
    const formData = new FormData();
    formData.append('depth', params.depth);
    if (params.rgb) {
      formData.append('rgb', params.rgb);
    }
    formData.append('factor', params.factor.toString());
    formData.append('method', params.method);
    formData.append('config', JSON.stringify(params.config));
    if (params.intrinsics) {
      formData.append('intrinsics', JSON.stringify(params.intrinsics));
    }

    return this.request<JobCreated>('/api/reconstruct', {
      method: 'POST',
      body: formData,
    });
  }

  async getJobStatus(jobId: string): Promise<JobStatus> {
    return this.request<JobStatus>(`/api/reconstruct/${encodeURIComponent(jobId)}/status`);
  }

  async getJobResult(jobId: string): Promise<ReconstructResult> {
    return this.request<ReconstructResult>(`/api/reconstruct/${encodeURIComponent(jobId)}/result`);
  }

  async evaluate(reconstructed: Blob, groundTruth: Blob, validMask?: Blob): Promise<Metrics> {
    const formData = new FormData();
    formData.append('reconstructed', reconstructed);
    formData.append('ground_truth', groundTruth);
    if (validMask) {
      formData.append('valid_mask', validMask);
    }

    return this.request<Metrics>('/api/evaluate', {
      method: 'POST',
      body: formData,
    });
  }

  async listExperiments(): Promise<Experiment[]> {
    return this.request<Experiment[]>('/api/experiments');
  }

  async getExperiment(id: string): Promise<Experiment> {
    return this.request<Experiment>(`/api/experiments/${encodeURIComponent(id)}`);
  }

  async deleteExperiment(id: string): Promise<void> {
    const res = await fetch(`${BASE_URL}/api/experiments/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      throw new Error(`Failed to delete experiment: ${res.statusText}`);
    }
  }

  async renameExperiment(id: string, name: string): Promise<Experiment> {
    return this.request<Experiment>(`/api/experiments/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
  }

  getExportUrl(id: string, format: ExportFormat): string {
    return `${BASE_URL}/api/experiments/${encodeURIComponent(id)}/export?format=${encodeURIComponent(format)}`;
  }

  getDepthNpyUrl(jobId: string): string {
    return `${BASE_URL}/api/reconstruct/${encodeURIComponent(jobId)}/result/depth`;
  }

  getStageDepthUrl(jobId: string, stage: string): string {
    return `${BASE_URL}/api/reconstruct/${encodeURIComponent(jobId)}/result/depth/${encodeURIComponent(stage)}`;
  }

  getEdgesUrl(jobId: string): string {
    return `${BASE_URL}/api/reconstruct/${encodeURIComponent(jobId)}/result/edges`;
  }

  getGeometryUrl(jobId: string): string {
    return `${BASE_URL}/api/reconstruct/${encodeURIComponent(jobId)}/result/geometry`;
  }
}

export const apiClient = new ApiClient();

