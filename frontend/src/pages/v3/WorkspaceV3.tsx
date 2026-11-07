import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { processImageDepth, SAMPLE_PRESETS, type ColormapMode, type DepthResult } from '../../lib/depthPipeline';
import { exportDepthToObj, downloadFile } from '../../lib/objExporter';
import { CustomCursor } from '../../components/shared/CustomCursor';
import {
  ArrowLeft,
  Upload,
  Download,
  RotateCcw
} from 'lucide-react';

export const WorkspaceV3: React.FC = () => {
  const [searchParams] = useSearchParams();
  const initialPreset = searchParams.get('preset') || 'sculpture';

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Studio State
  const [depthResult, setDepthResult] = useState<DepthResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [renderMode, setRenderMode] = useState<'surface' | 'points' | 'wireframe' | 'clay'>('surface');
  const [colormap] = useState<ColormapMode>('turbo');
  const [depthScale, setDepthScale] = useState<number>(0.85);
  const [activePresetId, setActivePresetId] = useState<string>(initialPreset);

  // Three.js References
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);

  // Load and process image depth
  const loadSource = useCallback(async (source: string | File, presetId?: string) => {
    setIsLoading(true);
    try {
      const res = await processImageDepth(source, colormap, 384);
      setDepthResult(res);
      if (presetId) setActivePresetId(presetId);
    } catch (err) {
      console.error('Error processing depth map:', err);
    } finally {
      setIsLoading(false);
    }
  }, [colormap]);

  // Initial load
  useEffect(() => {
    const preset = SAMPLE_PRESETS.find((p) => p.id === initialPreset) || SAMPLE_PRESETS[0];
    loadSource(preset.src, preset.id);
  }, []);

  // Update colormap when changed
  useEffect(() => {
    if (!depthResult) return;
    processImageDepth(depthResult.rgbCanvas.toDataURL(), colormap, 384).then(setDepthResult);
  }, [colormap]);

  // Handle Drag & Drop File Upload
  const handleFileUpload = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    loadSource(file, 'custom');
  };

  // Initialize Three.js Scene
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.set(0, 0, 3.2);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 6.0;
    controls.minDistance = 1.0;
    controlsRef.current = controls;

    // Apple Spatial Key lighting (warm directional + soft ambient)
    const ambLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
    dirLight.position.set(2, 3, 4);
    scene.add(dirLight);

    const backLight = new THREE.DirectionalLight(0x7090ff, 0.6);
    backLight.position.set(-2, -2, -3);
    scene.add(backLight);

    let reqId: number;
    const animate = () => {
      reqId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!canvas) return;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(reqId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
    };
  }, []);

  // Update Geometry & Textures
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !depthResult) return;

    if (meshRef.current) {
      scene.remove(meshRef.current);
      meshRef.current.geometry.dispose();
      (meshRef.current.material as THREE.Material).dispose();
      meshRef.current = null;
    }
    if (pointsRef.current) {
      scene.remove(pointsRef.current);
      pointsRef.current.geometry.dispose();
      (pointsRef.current.material as THREE.Material).dispose();
      pointsRef.current = null;
    }

    const { width, height, depthArray, colormapCanvas } = depthResult;
    const gridRes = Math.min(width, 256);
    const stride = Math.floor(width / gridRes);

    const colorTexture = new THREE.CanvasTexture(colormapCanvas);
    colorTexture.minFilter = THREE.LinearFilter;
    colorTexture.magFilter = THREE.LinearFilter;

    if (renderMode === 'points') {
      const numPoints = gridRes * gridRes;
      const positions = new Float32Array(numPoints * 3);
      const colors = new Float32Array(numPoints * 3);

      const ctx = colormapCanvas.getContext('2d')!;
      const pixelData = ctx.getImageData(0, 0, width, height).data;

      let pIdx = 0;
      for (let y = 0; y < gridRes; y++) {
        const origY = y * stride;
        const ny = (y / (gridRes - 1) - 0.5) * -1.8;

        for (let x = 0; x < gridRes; x++) {
          const origX = x * stride;
          const nx = (x / (gridRes - 1) - 0.5) * 1.8;
          const depthVal = depthArray[origY * width + origX];
          const nz = (depthVal - 0.5) * depthScale;

          positions[pIdx * 3] = nx;
          positions[pIdx * 3 + 1] = ny;
          positions[pIdx * 3 + 2] = nz;

          const pixPos = (origY * width + origX) * 4;
          colors[pIdx * 3] = pixelData[pixPos] / 255;
          colors[pIdx * 3 + 1] = pixelData[pixPos + 1] / 255;
          colors[pIdx * 3 + 2] = pixelData[pixPos + 2] / 255;

          pIdx++;
        }
      }

      const pointGeo = new THREE.BufferGeometry();
      pointGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      pointGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const pointMat = new THREE.PointsMaterial({
        size: 0.02,
        vertexColors: true,
        transparent: true,
        opacity: 0.95,
      });

      const points = new THREE.Points(pointGeo, pointMat);
      scene.add(points);
      pointsRef.current = points;
    } else {
      const geo = new THREE.PlaneGeometry(1.8, 1.8, gridRes - 1, gridRes - 1);
      const posAttr = geo.attributes.position;

      for (let i = 0; i < posAttr.count; i++) {
        const x = i % gridRes;
        const y = Math.floor(i / gridRes);
        const origX = Math.min(width - 1, x * stride);
        const origY = Math.min(height - 1, y * stride);
        const d = depthArray[origY * width + origX];
        posAttr.setZ(i, (d - 0.5) * depthScale);
      }
      geo.computeVertexNormals();

      let mat: THREE.Material;
      if (renderMode === 'wireframe') {
        mat = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          wireframe: true,
          transparent: true,
          opacity: 0.65,
        });
      } else if (renderMode === 'clay') {
        mat = new THREE.MeshStandardMaterial({
          color: 0xf0f0f2,
          roughness: 0.4,
          metalness: 0.05,
          side: THREE.DoubleSide,
        });
      } else {
        mat = new THREE.MeshStandardMaterial({
          map: colorTexture,
          roughness: 0.35,
          metalness: 0.1,
          side: THREE.DoubleSide,
        });
      }

      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      meshRef.current = mesh;
    }
  }, [depthResult, renderMode, depthScale]);

  // Reset Camera Gimbal
  const handleResetCamera = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(0, 0, 3.2);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  };

  // Export 3D .OBJ
  const handleExportObj = () => {
    if (!depthResult) return;
    const objContent = exportDepthToObj(depthResult.depthArray, depthResult.width, depthResult.height, depthScale);
    downloadFile(objContent, `spatial-mesh-${Date.now()}.obj`, 'text/plain');
  };

  // Export 16-Bit PNG
  const handleExportPng = () => {
    if (!depthResult) return;
    const link = document.createElement('a');
    link.download = `spatial-depth-${Date.now()}.png`;
    link.href = depthResult.depthCanvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '100vw',
        height: '100vh',
        backgroundColor: '#08090C',
        color: '#F5F5F7',
        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", sans-serif',
        overflow: 'hidden',
        WebkitFontSmoothing: 'antialiased',
      }}
    >
      <CustomCursor />

      {/* ── Apple Liquid Glass Floating Top Navigation ── */}
      <header
        style={{
          position: 'fixed',
          top: '20px',
          left: '24px',
          right: '24px',
          zIndex: 100,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          pointerEvents: 'none',
        }}
      >
        {/* Back Link & Benchmarks */}
        <div style={{ pointerEvents: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Link
            to="/"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 18px',
              borderRadius: '999px',
              backgroundColor: 'rgba(25, 27, 34, 0.65)',
              backdropFilter: 'blur(28px) saturate(190%)',
              WebkitBackdropFilter: 'blur(28px) saturate(190%)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              boxShadow: '0 15px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
              color: '#FFFFFF',
              textDecoration: 'none',
              fontSize: '13px',
              fontWeight: 500,
              transition: 'transform 0.15s ease',
            }}
          >
            <ArrowLeft size={14} />
            <span>Exit Studio</span>
          </Link>

          <Link
            to="/studio"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '10px 16px',
              borderRadius: '999px',
              backgroundColor: 'rgba(25, 27, 34, 0.65)',
              backdropFilter: 'blur(28px) saturate(190%)',
              WebkitBackdropFilter: 'blur(28px) saturate(190%)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              color: '#9CA3AF',
              textDecoration: 'none',
              fontSize: '12.5px',
              fontWeight: 500,
            }}
          >
            <span>Benchmarks</span>
          </Link>
        </div>

        {/* Center Mode Switcher Glass Capsule */}
        <div
          style={{
            pointerEvents: 'auto',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '4px',
            borderRadius: '999px',
            backgroundColor: 'rgba(25, 27, 34, 0.65)',
            backdropFilter: 'blur(28px) saturate(190%)',
            WebkitBackdropFilter: 'blur(28px) saturate(190%)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            boxShadow: '0 15px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
          }}
        >
          {(
            [
              { id: 'surface', label: 'Spatial Color' },
              { id: 'clay', label: 'Sculpted Clay' },
              { id: 'points', label: 'Point Cloud' },
              { id: 'wireframe', label: 'Wireframe' },
            ] as const
          ).map((m) => (
            <button
              key={m.id}
              onClick={() => setRenderMode(m.id)}
              style={{
                padding: '7px 16px',
                borderRadius: '999px',
                border: 'none',
                backgroundColor: renderMode === m.id ? '#FFFFFF' : 'transparent',
                color: renderMode === m.id ? '#08090C' : '#86868B',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
              }}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Right Actions: Export Glass Pills */}
        <div style={{ pointerEvents: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={handleResetCamera}
            style={{
              padding: '10px',
              borderRadius: '50%',
              backgroundColor: 'rgba(25, 27, 34, 0.65)',
              backdropFilter: 'blur(28px)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              color: '#FFFFFF',
              cursor: 'pointer',
            }}
            title="Recenter View"
          >
            <RotateCcw size={14} />
          </button>
          <button
            onClick={handleExportPng}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '10px 16px',
              borderRadius: '999px',
              backgroundColor: 'rgba(25, 27, 34, 0.65)',
              backdropFilter: 'blur(28px)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              color: '#FFFFFF',
              fontSize: '12.5px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            <Download size={13} />
            <span>Depth .PNG</span>
          </button>
          <button
            onClick={handleExportObj}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '10px 18px',
              borderRadius: '999px',
              backgroundColor: '#FFFFFF',
              color: '#08090C',
              border: 'none',
              fontSize: '12.5px',
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: '0 10px 20px rgba(0, 0, 0, 0.3)',
            }}
          >
            <Download size={13} />
            <span>Export 3D .OBJ</span>
          </button>
        </div>
      </header>

      {/* ── Main Full-Screen Three.js Canvas Stage ── */}
      <main
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
            outline: 'none',
          }}
        />

        {/* Loading Glass Pill */}
        {isLoading && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              padding: '12px 24px',
              borderRadius: '999px',
              backgroundColor: 'rgba(25, 27, 34, 0.75)',
              backdropFilter: 'blur(24px)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              color: '#FFFFFF',
              fontSize: '13px',
              fontWeight: 500,
              boxShadow: '0 20px 40px rgba(0, 0, 0, 0.5)',
            }}
          >
            Reconstructing Spatial Surface...
          </div>
        )}
      </main>

      {/* ── Apple Liquid Glass Floating Bottom Control Island ── */}
      <div
        style={{
          position: 'fixed',
          bottom: '24px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 100,
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
          padding: '8px 18px',
          borderRadius: '999px',
          backgroundColor: 'rgba(25, 27, 34, 0.7)',
          backdropFilter: 'blur(30px) saturate(190%)',
          WebkitBackdropFilter: 'blur(30px) saturate(190%)',
          border: '1px solid rgba(255, 255, 255, 0.14)',
          boxShadow: '0 30px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.25)',
          maxWidth: '90vw',
        }}
      >
        {/* Curated Preset Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {SAMPLE_PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => loadSource(p.src, p.id)}
              style={{
                width: '30px',
                height: '30px',
                borderRadius: '50%',
                border: activePresetId === p.id ? '2px solid #FFFFFF' : '1px solid rgba(255, 255, 255, 0.15)',
                overflow: 'hidden',
                padding: 0,
                cursor: 'pointer',
                transition: 'transform 0.15s ease',
                transform: activePresetId === p.id ? 'scale(1.08)' : 'scale(1)',
              }}
              title={p.title}
            >
              <img src={p.src} alt={p.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </button>
          ))}
        </div>

        <div style={{ width: '1px', height: '24px', backgroundColor: 'rgba(255, 255, 255, 0.12)' }} />

        {/* Depth Relief Slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '12px', color: '#86868B', fontWeight: 500 }}>Depth Gain</span>
          <input
            type="range"
            min={0.1}
            max={2.0}
            step={0.05}
            value={depthScale}
            onChange={(e) => setDepthScale(Number(e.target.value))}
            style={{ width: '75px', accentColor: '#FFFFFF' }}
          />
          <span style={{ fontSize: '12px', color: '#FFFFFF', fontWeight: 600, width: '32px' }}>
            {depthScale.toFixed(2)}
          </span>
        </div>

        <div style={{ width: '1px', height: '24px', backgroundColor: 'rgba(255, 255, 255, 0.12)' }} />

        {/* Custom Upload Button */}
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '7px 14px',
            borderRadius: '999px',
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            color: '#FFFFFF',
            fontSize: '12px',
            fontWeight: 500,
            cursor: 'pointer',
          }}
        >
          <Upload size={12} />
          <span>Upload Image</span>
          <input
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFileUpload(file);
            }}
          />
        </label>
      </div>
    </div>
  );
};
