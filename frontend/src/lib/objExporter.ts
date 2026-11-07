/**
 * Client-Side Wavefront OBJ Exporter
 * Converts heightfield depth grid into standard .obj 3D mesh file for Blender, CAD, or Three.js
 */

export function exportDepthToObj(
  depthArray: Float32Array,
  width: number,
  height: number,
  depthScale = 0.5,
  stride = 2 // subsample for lightweight fast download
): string {
  const lines: string[] = [];
  lines.push('# DepthLab 3D Mesh Export');
  lines.push('# Generator: Edge-Preserving Client Depth Reconstruction');
  lines.push(`o DepthLab_Mesh_${width}x${height}`);

  const subW = Math.floor(width / stride);
  const subH = Math.floor(height / stride);

  // 1. Write Vertices
  for (let y = 0; y < subH; y++) {
    const origY = y * stride;
    const ny = (y / (subH - 1) - 0.5) * 2.0;

    for (let x = 0; x < subW; x++) {
      const origX = x * stride;
      const nx = (x / (subW - 1) - 0.5) * 2.0;
      const z = depthArray[origY * width + origX] * depthScale;
      lines.push(`v ${nx.toFixed(4)} ${(-ny).toFixed(4)} ${z.toFixed(4)}`);
    }
  }

  // 2. Write Texture UVs
  for (let y = 0; y < subH; y++) {
    const v = 1.0 - y / (subH - 1);
    for (let x = 0; x < subW; x++) {
      const u = x / (subW - 1);
      lines.push(`vt ${u.toFixed(4)} ${v.toFixed(4)}`);
    }
  }

  // 3. Write Face Triangles
  for (let y = 0; y < subH - 1; y++) {
    for (let x = 0; x < subW - 1; x++) {
      const i0 = y * subW + x + 1;
      const i1 = y * subW + (x + 1) + 1;
      const i2 = (y + 1) * subW + (x + 1) + 1;
      const i3 = (y + 1) * subW + x + 1;

      // Triangle 1: i0 - i1 - i2
      lines.push(`f ${i0}/${i0} ${i1}/${i1} ${i2}/${i2}`);
      // Triangle 2: i0 - i2 - i3
      lines.push(`f ${i0}/${i0} ${i2}/${i2} ${i3}/${i3}`);
    }
  }

  return lines.join('\n');
}

export function downloadFile(content: string | Blob, filename: string, mimeType = 'text/plain') {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

