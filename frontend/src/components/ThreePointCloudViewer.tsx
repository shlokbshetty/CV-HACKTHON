import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface ThreePointCloudViewerProps {
  bufferUrl?: string;
  depthUrl: string;
  rgbUrl: string;
  label?: string;
  colorMode?: 'rgb' | 'depth' | 'discontinuity';
  pointSize?: number;
  stride?: number;
  disconnectEdges?: boolean;
  filterSky?: boolean;
  height?: number | string;
  isHero?: boolean;
}

// Approximate turbo colormap for metric depth visualization
function turboColormap(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 0.1357 + x * (4.5974 - x * (42.68 - x * (152.5 - x * (221.7 - x * 110.2))))));
  const g = Math.max(0, Math.min(1, 0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 + x * (4.21 - x * 2.77))))));
  const b = Math.max(0, Math.min(1, 0.1067 + x * (12.573 - x * (60.19 + x * (109.07 - x * (88.5 - x * 26.8))))));
  return [r, g, b];
}

export const ThreePointCloudViewer: React.FC<ThreePointCloudViewerProps> = ({
  bufferUrl,
  depthUrl,
  rgbUrl,
  label,
  colorMode = 'rgb',
  pointSize = 3.0,
  stride = 2,
  disconnectEdges = true,
  filterSky = false,
  height = 480,
  isHero = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const width = container.clientWidth || 400;
    const hNum = typeof height === 'number' ? height : parseInt(height as string, 10) || 480;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x08090c);

    const camera = new THREE.PerspectiveCamera(45, width / hNum, 0.05, 150);
    camera.position.set(0, 0.2, isHero ? 2.6 : 3.0);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setSize(width, hNum);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.7;
    controls.zoomSpeed = 0.8;
    controls.panSpeed = 0.6;

    // Ambient + Directional lighting
    const ambLight = new THREE.AmbientLight(0xffffff, 0.95);
    scene.add(ambLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(2, 4, 3);
    scene.add(dirLight);

    let pointsObject: THREE.Points | null = null;
    let isDisposed = false;

    // Helper to create points mesh from positions and colors
    const applyPoints = (posArray: Float32Array, colArray: Float32Array, count: number, spanSize = 2.0) => {
      if (isDisposed) return;
      if (pointsObject) {
        scene.remove(pointsObject);
        pointsObject.geometry.dispose();
        (pointsObject.material as THREE.Material).dispose();
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(posArray.subarray(0, count * 3), 3));
      geo.setAttribute('color', new THREE.BufferAttribute(colArray.subarray(0, count * 3), 3));

      const mat = new THREE.PointsMaterial({
        size: Math.max(0.012, (pointSize / 110) * (isHero ? 1.25 : 1.0) * (spanSize / 2.5)),
        vertexColors: true,
        transparent: true,
        opacity: 0.96,
      });

      pointsObject = new THREE.Points(geo, mat);
      scene.add(pointsObject);
    };

    // 1. Try to load metric Float32Array binary buffer (exact pinhole projection matching app.py)
    let bufferLoaded = false;
    if (bufferUrl) {
      fetch(bufferUrl)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.arrayBuffer();
        })
        .then((buf) => {
          if (isDisposed) return;
          const floats = new Float32Array(buf);
          const totalPoints = Math.floor(floats.length / 8);
          if (totalPoints === 0) throw new Error('Empty buffer');

          // First pass: compute bounding box of valid points
          let minX = Infinity, maxX = -Infinity;
          let minY = Infinity, maxY = -Infinity;
          let minZ = Infinity, maxZ = -Infinity;
          let minD = Infinity, maxD = -Infinity;

          for (let i = 0; i < totalPoints; i++) {
            const off = i * 8;
            const X = floats[off];
            const Y = floats[off + 1];
            const Z = floats[off + 2];
            const dVal = floats[off + 6];
            const isEdge = floats[off + 7];

            if (filterSky && Y > 40.0) continue;
            if (disconnectEdges && isEdge > 0.5) continue;

            if (X < minX) minX = X;
            if (X > maxX) maxX = X;
            if (Y < minY) minY = Y;
            if (Y > maxY) maxY = Y;
            if (Z < minZ) minZ = Z;
            if (Z > maxZ) maxZ = Z;
            if (dVal < minD) minD = dVal;
            if (dVal > maxD) maxD = dVal;
          }

          const cx = (minX + maxX) / 2;
          const cy = (minY + maxY) / 2;
          const cz = (minZ + maxZ) / 2;
          const span = Math.max(maxX - minX, maxZ - minZ, maxY - minY, 1.0);

          const pos = new Float32Array(totalPoints * 3);
          const col = new Float32Array(totalPoints * 3);
          let validCount = 0;

          for (let i = 0; i < totalPoints; i++) {
            const off = i * 8;
            const X = floats[off];
            const Y = floats[off + 1]; // metric depth into scene (m)
            const Z = floats[off + 2]; // metric height upright (m)
            const r = floats[off + 3];
            const g = floats[off + 4];
            const b = floats[off + 5];
            const dVal = floats[off + 6];
            const isEdge = floats[off + 7];

            if (filterSky && Y > 40.0) continue;
            if (disconnectEdges && isEdge > 0.5) continue;

            // Coordinate mapping to Three.js world space:
            // X -> x (horizontal)
            // Z -> y (vertical upright)
            // Y -> -z (depth into scene away from viewer)
            pos[validCount * 3] = X - cx;
            pos[validCount * 3 + 1] = Z - cz;
            pos[validCount * 3 + 2] = -(Y - cy);

            if (colorMode === 'depth') {
              // Warm (near) to Cool (far) matching app.py line 403
              const normD = 1.0 - Math.max(0, Math.min(1, (dVal - minD) / Math.max(maxD - minD, 1e-4)));
              const [tr, tg, tb] = turboColormap(normD);
              col[validCount * 3] = tr;
              col[validCount * 3 + 1] = tg;
              col[validCount * 3 + 2] = tb;
            } else if (colorMode === 'discontinuity') {
              if (isEdge > 0.5) {
                // Scarlet red for flying pixel edge discontinuity (app.py line 423)
                col[validCount * 3] = 1.0;
                col[validCount * 3 + 1] = 0.22;
                col[validCount * 3 + 2] = 0.22;
              } else {
                col[validCount * 3] = Math.max(r, 0.11);
                col[validCount * 3 + 1] = Math.max(g, 0.11);
                col[validCount * 3 + 2] = Math.max(b, 0.11);
              }
            } else {
              // RGB mode with slight baseline floor so dark furniture doesn't vanish into black canvas (app.py line 429)
              col[validCount * 3] = Math.max(r, 0.11);
              col[validCount * 3 + 1] = Math.max(g, 0.11);
              col[validCount * 3 + 2] = Math.max(b, 0.11);
            }

            validCount++;
          }

          bufferLoaded = true;
          camera.position.set(0, 0.08 * span, Math.max(1.8, span * 1.2));
          controls.target.set(0, 0, 0);
          controls.update();
          applyPoints(pos, col, validCount, span);
        })
        .catch(() => {
          // If binary buffer fails, fallback to image-based loader
          if (!bufferLoaded) loadFromImages();
        });
    }

    // 2. Image-based fallback loader (used for custom photo/depth uploads)
    const loadFromImages = () => {
      const rgbImg = new Image();
      const depthImg = new Image();
      let rgbLoaded = false;
      let depthLoaded = false;

      const buildPointCloudFromImages = () => {
        if (!rgbLoaded || !depthLoaded || isDisposed) return;

        const simW = 180;
        const simH = 180;
        const offCanvas = document.createElement('canvas');
        offCanvas.width = simW;
        offCanvas.height = simH;
        const ctx = offCanvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(rgbImg, 0, 0, simW, simH);
        const rgbData = ctx.getImageData(0, 0, simW, simH).data;

        ctx.clearRect(0, 0, simW, simH);
        ctx.drawImage(depthImg, 0, 0, simW, simH);
        const depthData = ctx.getImageData(0, 0, simW, simH).data;

        const effectiveStride = Math.max(1, stride);
        const sampleCols = Math.floor(simW / effectiveStride);
        const sampleRows = Math.floor(simH / effectiveStride);
        const maxPts = sampleCols * sampleRows;

        const positions = new Float32Array(maxPts * 3);
        const colors = new Float32Array(maxPts * 3);
        let validCount = 0;

        for (let y = 0; y < simH; y += effectiveStride) {
          for (let x = 0; x < simW; x += effectiveStride) {
            const idx = (y * simW + x) * 4;
            const r = rgbData[idx] / 255;
            const g = rgbData[idx + 1] / 255;
            const b = rgbData[idx + 2] / 255;
            const dVal = (depthData[idx] * 0.299 + depthData[idx + 1] * 0.587 + depthData[idx + 2] * 0.114) / 255;

            if (filterSky && y < simH * 0.35 && dVal > 0.85) continue;

            let isDiscontinuity = false;
            if (x < simW - effectiveStride && y < simH - effectiveStride) {
              const nextIdx = (y * simW + (x + effectiveStride)) * 4;
              const nextD = (depthData[nextIdx] * 0.299 + depthData[nextIdx + 1] * 0.587 + depthData[nextIdx + 2] * 0.114) / 255;
              if (Math.abs(dVal - nextD) > 0.14) {
                isDiscontinuity = true;
              }
            }

            if (disconnectEdges && isDiscontinuity && colorMode !== 'discontinuity') {
              continue;
            }

            const nx = (x / simW - 0.5) * 2.2;
            const ny = -(y / simH - 0.5) * 2.2;
            const nz = -(dVal - 0.5) * 1.5;

            positions[validCount * 3] = nx;
            positions[validCount * 3 + 1] = ny;
            positions[validCount * 3 + 2] = nz;

            if (colorMode === 'depth') {
              const [tr, tg, tb] = turboColormap(1.0 - dVal);
              colors[validCount * 3] = tr;
              colors[validCount * 3 + 1] = tg;
              colors[validCount * 3 + 2] = tb;
            } else if (colorMode === 'discontinuity') {
              if (isDiscontinuity) {
                colors[validCount * 3] = 1.0;
                colors[validCount * 3 + 1] = 0.22;
                colors[validCount * 3 + 2] = 0.22;
              } else {
                colors[validCount * 3] = Math.max(r, 0.11);
                colors[validCount * 3 + 1] = Math.max(g, 0.11);
                colors[validCount * 3 + 2] = Math.max(b, 0.11);
              }
            } else {
              colors[validCount * 3] = Math.max(r, 0.11);
              colors[validCount * 3 + 1] = Math.max(g, 0.11);
              colors[validCount * 3 + 2] = Math.max(b, 0.11);
            }

            validCount++;
          }
        }

        applyPoints(positions, colors, validCount, 2.2);
      };

      rgbImg.crossOrigin = 'anonymous';
      rgbImg.onload = () => {
        rgbLoaded = true;
        buildPointCloudFromImages();
      };
      rgbImg.src = rgbUrl;

      depthImg.crossOrigin = 'anonymous';
      depthImg.onload = () => {
        depthLoaded = true;
        buildPointCloudFromImages();
      };
      depthImg.src = depthUrl;
    };

    if (!bufferUrl) {
      loadFromImages();
    }

    let reqId: number;
    const animate = () => {
      if (isDisposed) return;
      reqId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container || !canvas || isDisposed) return;
      const newW = container.clientWidth;
      camera.aspect = newW / hNum;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, hNum);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      isDisposed = true;
      cancelAnimationFrame(reqId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      if (pointsObject) {
        scene.remove(pointsObject);
        pointsObject.geometry.dispose();
        (pointsObject.material as THREE.Material).dispose();
      }
      renderer.dispose();
    };
  }, [bufferUrl, depthUrl, rgbUrl, colorMode, pointSize, stride, disconnectEdges, filterSky, height, isHero]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: typeof height === 'number' ? `${height}px` : height,
        borderRadius: '16px',
        overflow: 'hidden',
        position: 'relative',
        backgroundColor: '#08090C',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow: '0 16px 36px rgba(0,0,0,0.6)',
      }}
    >
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />

      {label && (
        <div
          style={{
            position: 'absolute',
            bottom: '12px',
            left: '12px',
            padding: '5px 12px',
            borderRadius: '999px',
            backgroundColor: 'rgba(10, 12, 16, 0.85)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            fontSize: '11px',
            fontWeight: 600,
            color: '#FFFFFF',
            letterSpacing: '0.02em',
            pointerEvents: 'none',
          }}
        >
          {label}
        </div>
      )}

      <div
        style={{
          position: 'absolute',
          top: '12px',
          right: '12px',
          padding: '4px 10px',
          borderRadius: '999px',
          backgroundColor: 'rgba(0, 0, 0, 0.6)',
          fontSize: '10px',
          color: '#9CA3AF',
          pointerEvents: 'none',
        }}
      >
        Left Click: Rotate • Right Click: Pan • Scroll: Zoom
      </div>
    </div>
  );
};
