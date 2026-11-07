import React, { useEffect, useRef } from 'react';
import { useThemeStore } from '../../stores/theme';

interface ContourBackgroundProps {
  parallaxOffset?: { x: number; y: number };
}

export const ContourBackground: React.FC<ContourBackgroundProps> = ({ parallaxOffset = { x: 0, y: 0 } }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { theme } = useThemeStore();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Check prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const gl = canvas.getContext('webgl', { alpha: true, antialias: false, powerPreference: 'low-power' });
    if (!gl) return;

    // Vertex shader
    const vsSource = `
      attribute vec2 aPosition;
      varying vec2 vUv;
      void main() {
        vUv = aPosition * 0.5 + 0.5;
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `;

    // Fragment shader with smooth FBM noise and anti-aliased fract iso-lines via fwidth
    const fsSource = `
      #extension GL_OES_standard_derivatives : enable
      precision mediump float;
      varying vec2 vUv;
      uniform vec2 uResolution;
      uniform float uTime;
      uniform vec2 uParallax;
      uniform vec3 uLineColor;
      uniform float uLineOpacity;

      // Fast, smooth pseudo-random hash
      vec2 hash2(vec2 p) {
        return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
      }

      // 2D Simplex/Perlin gradient noise
      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
              dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
          mix(dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
              dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x),
          u.y
        );
      }

      // Fractional Brownian Motion (3 octaves for gentle performance)
      float fbm(vec2 p) {
        float v = 0.0;
        float a = 0.5;
        vec2 shift = vec2(100.0);
        mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
        for (int i = 0; i < 3; ++i) {
          v += a * noise(p);
          p = rot * p * 2.0 + shift;
          a *= 0.5;
        }
        return v;
      }

      void main() {
        vec2 uv = (gl_FragCoord.xy + uParallax) / uResolution.xy;
        uv.x *= uResolution.x / uResolution.y;

        // Subtle, imperceptible drift
        float t = uTime * 0.035;
        float h = fbm(uv * 1.8 + vec2(t * 0.2, t * 0.15)) * 0.5 + 0.5;

        // Iso-contour lines: fract scale determines contour density
        float density = 18.0;
        float val = h * density;
        float f = abs(fract(val - 0.5) - 0.5);
        
        // Anti-aliased line width using screen-space derivative (fwidth)
        #ifdef GL_OES_standard_derivatives
          float df = fwidth(val);
          float line = 1.0 - smoothstep(0.0, df * 1.4, f);
        #else
          float line = 1.0 - smoothstep(0.0, 0.08, f);
        #endif

        gl_FragColor = vec4(uLineColor, line * uLineOpacity);
      }
    `;

    // Compile helper
    function createShader(gl: WebGLRenderingContext, type: number, source: string) {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    }

    // Enable OES_standard_derivatives
    gl.getExtension('OES_standard_derivatives');

    const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );

    const aPos = gl.getAttribLocation(program, 'aPosition');
    const uRes = gl.getUniformLocation(program, 'uResolution');
    const uTime = gl.getUniformLocation(program, 'uTime');
    const uParallax = gl.getUniformLocation(program, 'uParallax');
    const uLineColor = gl.getUniformLocation(program, 'uLineColor');
    const uLineOpacity = gl.getUniformLocation(program, 'uLineOpacity');

    let animId: number;
    let startTime = performance.now();
    let isVisible = true;

    // Resize buffer to 0.5 DPR per spec (01-landing.md §A)
    function resize() {
      if (!canvas) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5) * 0.5;
      const w = Math.floor(window.innerWidth * dpr);
      const h = Math.floor(window.innerHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }

    window.addEventListener('resize', resize);
    resize();

    // Visibility listener to pause when tab is hidden
    const onVisibilityChange = () => {
      isVisible = !document.hidden;
    };
    document.addEventListener('visibilitychange', onVisibilityChange);

    function render(time: number) {
      if (!gl || !canvas) return;

      if (isVisible) {
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(program);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

        gl.uniform2f(uRes, canvas.width, canvas.height);
        gl.uniform1f(uTime, (time - startTime) * 0.001);
        gl.uniform2f(uParallax, parallaxOffset.x * 2.0, parallaxOffset.y * 2.0);

        // Theme line color & opacity (00-master.md §4)
        if (theme === 'night') {
          gl.uniform3f(uLineColor, 1.0, 1.0, 1.0);
          gl.uniform1f(uLineOpacity, 0.07);
        } else {
          gl.uniform3f(uLineColor, 0.1, 0.1, 0.11);
          gl.uniform1f(uLineOpacity, 0.08);
        }

        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }

      animId = requestAnimationFrame(render);
    }

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
      document.removeEventListener('visibilitychange', onVisibilityChange);
      if (gl) {
        gl.deleteProgram(program);
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        gl.deleteBuffer(posBuffer);
      }
    };
  }, [theme, parallaxOffset]);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 0,
        overflow: 'hidden',
      }}
      aria-hidden="true"
    >
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
        }}
      />
    </div>
  );
};

