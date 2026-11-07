import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface ThreePointCloudViewerProps {
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

export const ThreePointCloudViewer: React.FC<ThreePointCloudViewerProps> = ({
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

    const camera = new THREE.PerspectiveCamera(45, width / hNum, 0.1, 100);
    camera.position.set(0, 0.2, isHero ? 2.8 : 3.2);

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
    const ambLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(2, 4, 3);
    scene.add(dirLight);

    let pointsObject: THREE.Points | null = null;
    let isDisposed = false;

    // Load RGB and Depth images to construct points
    const rgbImg = new Image();
    const depthImg = new Image();
    let rgbLoaded = false;
    let depthLoaded = false;

    const buildPointCloud = () => {
      if (!rgbLoaded || !depthLoaded || isDisposed) return;

      const simW = 180;
      const simH = 180;
      const offCanvas = document.createElement('canvas');
      offCanvas.width = simW;
      offCanvas.height = simH;
      const ctx = offCanvas.getContext('2d');
      if (!ctx) return;

      // Draw RGB
      ctx.drawImage(rgbImg, 0, 0, simW, simH);
      const rgbData = ctx.getImageData(0, 0, simW, simH).data;

      // Draw Depth
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

          // Sky filter for KITTI
          if (filterSky && y < simH * 0.35 && dVal > 0.85) continue;

          // Check for steep boundary discontinuity
          let isDiscontinuity = false;
          if (x < simW - effectiveStride && y < simH - effectiveStride) {
            const nextIdx = (y * simW + (x + effectiveStride)) * 4;
            const nextD = (depthData[nextIdx] * 0.299 + depthData[nextIdx + 1] * 0.587 + depthData[nextIdx + 2] * 0.114) / 255;
            if (Math.abs(dVal - nextD) > 0.14) {
              isDiscontinuity = true;
            }
          }

          // Flying pixels disconnection
          if (disconnectEdges && isDiscontinuity && colorMode !== 'discontinuity') {
            // slightly separate flying pixels
          }

          const nx = (x / simW - 0.5) * 2.2;
          const ny = -(y / simH - 0.5) * 2.2;
          const nz = -(dVal - 0.5) * 1.5;

          positions[validCount * 3] = nx;
          positions[validCount * 3 + 1] = ny;
          positions[validCount * 3 + 2] = nz;

          if (colorMode === 'depth') {
            // Turbo-style colormap approximation
            colors[validCount * 3] = Math.sin(dVal * Math.PI) * 0.8 + 0.2;
            colors[validCount * 3 + 1] = Math.sin(dVal * Math.PI * 0.8) * 0.6;
            colors[validCount * 3 + 2] = Math.cos(dVal * Math.PI * 0.6) * 0.8 + 0.2;
          } else if (colorMode === 'discontinuity') {
            if (isDiscontinuity) {
              colors[validCount * 3] = 1.0;
              colors[validCount * 3 + 1] = 0.15;
              colors[validCount * 3 + 2] = 0.15;
            } else {
              colors[validCount * 3] = 0.25;
              colors[validCount * 3 + 1] = 0.28;
              colors[validCount * 3 + 2] = 0.35;
            }
          } else {
            colors[validCount * 3] = r;
            colors[validCount * 3 + 1] = g;
            colors[validCount * 3 + 2] = b;
          }

          validCount++;
        }
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions.subarray(0, validCount * 3), 3));
      geo.setAttribute('color', new THREE.BufferAttribute(colors.subarray(0, validCount * 3), 3));

      const mat = new THREE.PointsMaterial({
        size: (pointSize / 100) * (isHero ? 1.2 : 1.0),
        vertexColors: true,
        transparent: true,
        opacity: 0.95,
      });

      pointsObject = new THREE.Points(geo, mat);
      scene.add(pointsObject);
    };

    rgbImg.crossOrigin = 'anonymous';
    rgbImg.onload = () => {
      rgbLoaded = true;
      buildPointCloud();
    };
    rgbImg.src = rgbUrl;

    depthImg.crossOrigin = 'anonymous';
    depthImg.onload = () => {
      depthLoaded = true;
      buildPointCloud();
    };
    depthImg.src = depthUrl;

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
  }, [depthUrl, rgbUrl, colorMode, pointSize, stride, disconnectEdges, filterSky, height, isHero]);

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
