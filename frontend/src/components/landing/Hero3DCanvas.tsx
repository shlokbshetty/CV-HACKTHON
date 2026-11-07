import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useThemeStore } from '../../stores/theme';

interface Hero3DCanvasProps {
  progress: number; // 0.0 to 1.0 from GSAP ScrollTrigger
  pointerOffset?: { x: number; y: number };
}

export const Hero3DCanvas: React.FC<Hero3DCanvasProps> = ({ progress, pointerOffset = { x: 0, y: 0 } }) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const { theme } = useThemeStore();
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // Scene
    const scene = new THREE.Scene();

    // Size
    let width = container.clientWidth || 800;
    let height = container.clientHeight || 500;

    const camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 50);
    camera.position.set(0, 0, 4.2);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(renderer.domElement);

    // Texture Loader with explicit onLoad triggering
    const loader = new THREE.TextureLoader();
    let loadedCount = 0;
    function checkAllLoaded() {
      loadedCount++;
      if (loadedCount >= 3) {
        if (customMaterial) customMaterial.needsUpdate = true;
      }
    }

    const texRGB = loader.load('/assets/hero/butterfly_rgb.png', checkAllLoaded);
    const texDepth = loader.load('/assets/hero/butterfly_depth.png', checkAllLoaded);
    const texSparse = loader.load('/assets/hero/butterfly_sparse_3pct.png', checkAllLoaded);

    texRGB.generateMipmaps = true;
    texRGB.minFilter = THREE.LinearMipmapLinearFilter;
    texDepth.minFilter = THREE.LinearFilter;
    texSparse.minFilter = THREE.NearestFilter;

    // Geometry
    const isMobile = window.innerWidth < 900;
    const segments = isMobile ? 128 : 256;
    const geometry = new THREE.PlaneGeometry(2.6, 2.6, segments, segments);

    // Shader Material
    const customMaterial = new THREE.ShaderMaterial({
      transparent: true,
      side: THREE.DoubleSide,
      uniforms: {
        uProgress: { value: 0.0 },
        uDisplacement: { value: 0.0 },
        uTime: { value: 0.0 },
        uTexRGB: { value: texRGB },
        uTexDepth: { value: texDepth },
        uTexSparse: { value: texSparse },
        uIsNight: { value: theme === 'night' ? 1.0 : 0.0 },
      },
      vertexShader: `
        uniform float uDisplacement;
        uniform sampler2D uTexDepth;
        varying vec2 vUv;
        varying float vDepth;

        void main() {
          vUv = uv;
          float d = texture2D(uTexDepth, uv).r;
          vDepth = d;

          vec3 pos = position;
          pos.z += d * uDisplacement * 0.40;

          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uProgress;
        uniform float uDisplacement;
        uniform float uTime;
        uniform sampler2D uTexRGB;
        uniform sampler2D uTexDepth;
        uniform sampler2D uTexSparse;
        uniform float uIsNight;

        varying vec2 vUv;
        varying float vDepth;

        void main() {
          vec4 colRGB = texture2D(uTexRGB, vUv);
          if (colRGB.a < 0.15) discard;

          float d = texture2D(uTexDepth, vUv).r;

          // Discontinuity culling across sharp edges (built-in in WebGL2)
          float grad = length(vec2(dFdx(d), dFdy(d)));
          if (grad * uDisplacement > 0.09) discard;

          vec4 colSparse = texture2D(uTexSparse, vUv);
          vec3 inkColor = uIsNight > 0.5 ? vec3(0.92, 0.94, 0.96) : vec3(0.12, 0.12, 0.14);
          vec3 paperMuted = uIsNight > 0.5 ? vec3(0.12, 0.14, 0.16) : vec3(0.82, 0.82, 0.84);

          // Iso-contour line generation
          float contour = fract(d * 20.0);
          float line = smoothstep(0.0, 0.08, contour) * smoothstep(0.16, 0.08, contour);
          vec3 contourColor = mix(paperMuted, inkColor, line * 0.75);

          vec3 finalColor = colRGB.rgb;

          if (uProgress < 0.18) {
            // Stage 01: Flat RGB
            finalColor = colRGB.rgb;
          } else if (uProgress < 0.38) {
            // Stage 02: Sparse Dots
            float t = (uProgress - 0.18) / 0.20;
            vec3 sparseDisplay = colSparse.a > 0.5 ? inkColor : paperMuted;
            finalColor = mix(colRGB.rgb, sparseDisplay, t);
          } else if (uProgress < 0.60) {
            // Stage 03: Dense Depth Contours
            float t = (uProgress - 0.38) / 0.22;
            vec3 sparseDisplay = colSparse.a > 0.5 ? inkColor : paperMuted;
            finalColor = mix(sparseDisplay, contourColor, t);
          } else {
            // Stage 04: 3D Relief Textured
            float t = clamp((uProgress - 0.60) / 0.22, 0.0, 1.0);
            float shade = 0.85 + 0.30 * d;
            vec3 shadedRGB = colRGB.rgb * shade;
            finalColor = mix(contourColor, shadedRGB, t);
          }

          gl_FragColor = vec4(finalColor, colRGB.a);
        }
      `,
    });

    materialRef.current = customMaterial;

    const mesh = new THREE.Mesh(geometry, customMaterial);
    mesh.rotation.x = THREE.MathUtils.degToRad(4);
    scene.add(mesh);
    meshRef.current = mesh;

    // Contact shadow
    const shadowGeo = new THREE.PlaneGeometry(2.2, 0.6);
    const shadowMat = new THREE.ShaderMaterial({
      transparent: true,
      uniforms: {
        uIsNight: { value: theme === 'night' ? 1.0 : 0.0 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uIsNight;
        varying vec2 vUv;
        void main() {
          vec2 p = vUv * 2.0 - 1.0;
          float d = length(p);
          float alpha = smoothstep(1.0, 0.0, d) * (uIsNight > 0.5 ? 0.35 : 0.15);
          gl_FragColor = vec4(0.0, 0.0, 0.0, alpha);
        }
      `,
    });
    const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
    shadowMesh.position.set(0, -1.35, -0.05);
    scene.add(shadowMesh);

    let animId: number;
    let clock = new THREE.Clock();

    function onResize() {
      if (!container || !camera || !renderer) return;
      width = container.clientWidth || 800;
      height = container.clientHeight || 500;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }

    window.addEventListener('resize', onResize);

    function animate() {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      if (customMaterial) {
        customMaterial.uniforms.uTime.value = elapsed;
      }

      if (mesh) {
        const p = customMaterial ? customMaterial.uniforms.uProgress.value : 0;
        const pointerX = pointerOffset.x * 0.04;
        const pointerY = pointerOffset.y * 0.025;

        if (p >= 0.60) {
          const yaw = THREE.MathUtils.degToRad(11) * Math.sin(0.6 * elapsed) + pointerX;
          const pitch = THREE.MathUtils.degToRad(4 + 3 * Math.sin(0.4 * elapsed + 1.0)) - pointerY;
          mesh.rotation.y = yaw;
          mesh.rotation.x = pitch;
        } else {
          mesh.rotation.y = pointerX * 0.4;
          mesh.rotation.x = THREE.MathUtils.degToRad(4) - pointerY * 0.4;
        }
      }

      renderer.render(scene, camera);
    }

    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', onResize);
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      geometry.dispose();
      customMaterial.dispose();
      renderer.dispose();
    };
  }, []);

  // Sync uniforms
  useEffect(() => {
    if (!materialRef.current) return;
    materialRef.current.uniforms.uProgress.value = progress;

    let disp = 0.0;
    if (progress >= 0.60) {
      const t = Math.min((progress - 0.60) / 0.22, 1.0);
      disp = t * t * (3.0 - 2.0 * t);
    }
    materialRef.current.uniforms.uDisplacement.value = disp;
    materialRef.current.uniforms.uIsNight.value = theme === 'night' ? 1.0 : 0.0;
  }, [progress, theme]);

  return (
    <div
      ref={mountRef}
      style={{
        width: '100%',
        height: '100%',
        minHeight: '480px',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    />
  );
};
