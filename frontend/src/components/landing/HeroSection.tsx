import React, { useEffect, useRef, useState } from 'react';
import { Hero3DCanvas } from './Hero3DCanvas';
import type { ReadoutData } from './StageChrome';

interface HeroSectionProps {
  onProgressUpdate: (progress: number, readouts: ReadoutData) => void;
  pointerOffset: { x: number; y: number };
}

export const HeroSection: React.FC<HeroSectionProps> = ({ onProgressUpdate, pointerOffset }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [localProgress, setLocalProgress] = useState(0.0);

  useEffect(() => {
    const handleScroll = () => {
      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const totalScroll = container.offsetHeight - window.innerHeight;
      if (totalScroll <= 0) return;

      const currentScroll = -rect.top;
      const p = Math.max(0, Math.min(1, currentScroll / totalScroll));
      setLocalProgress(p);

      // Compute dynamic readouts (01-landing.md §C)
      let stageNum = '01';
      let stageName = 'RGB';
      let input = 'RGB (1024×1024)';
      let scale = '—';
      let geometry = '2D plane';

      if (p >= 0.18 && p < 0.38) {
        stageNum = '02';
        stageName = 'SPARSE';
        input = '3.0% valid (13,843 pts)';
        scale = '—';
        geometry = '13,843 sparse pts';
      } else if (p >= 0.38 && p < 0.60) {
        stageNum = '03';
        stageName = 'DENSE';
        input = 'Iso-contours (100%)';
        scale = '4x';
        geometry = '461,451 vertices';
      } else if (p >= 0.60) {
        stageNum = '04';
        stageName = '3D RELIEF';
        input = 'RGB + true depth';
        scale = '4x';
        geometry = '461,451 relief';
      }

      onProgressUpdate(p, {
        stageNum,
        stageName,
        input,
        scale,
        geometry,
      });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, [onProgressUpdate]);

  // Subtitle based on progress
  let subtitle = '\\ 01 RGB guidance \\';
  if (localProgress >= 0.18 && localProgress < 0.38) {
    subtitle = '\\ 02 Sparse depth \\';
  } else if (localProgress >= 0.38 && localProgress < 0.60) {
    subtitle = '\\ 03 Dense reconstruction \\';
  } else if (localProgress >= 0.60) {
    subtitle = '\\ 04 Geometry relief \\';
  }

  return (
    <section
      ref={containerRef}
      style={{
        height: '400vh',
        position: 'relative',
      }}
    >
      {/* Pinned 100vh stage */}
      <div
        style={{
          position: 'sticky',
          top: 0,
          height: '100vh',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
          boxSizing: 'border-box',
          paddingTop: '60px',
          paddingBottom: '110px',
        }}
      >
        {/* Centered Headline Section (01-landing.md §B) */}
        <div
          style={{
            textAlign: 'center',
            zIndex: 10,
            pointerEvents: 'none',
            userSelect: 'none',
          }}
        >
          <div
            className="type-subtitle"
            style={{
              marginBottom: 'var(--space-2)',
              transition: 'opacity 300ms ease',
            }}
          >
            {subtitle}
          </div>

          <h1
            className="type-display"
            style={{
              margin: 0,
              color: 'var(--ink)',
            }}
          >
            Reconstruct depth.
            <br />
            Preserve every edge.
          </h1>
        </div>

        {/* 3D WebGL Canvas Viewport */}
        <div
          style={{
            flex: 1,
            width: '100%',
            maxWidth: '880px',
            minHeight: '440px',
            position: 'relative',
            zIndex: 5,
          }}
        >
          <Hero3DCanvas progress={localProgress} pointerOffset={pointerOffset} />
        </div>

        {/* Caption (01-landing.md §C line 40) */}
        <div
          className="type-label"
          style={{
            position: 'absolute',
            bottom: '120px',
            fontSize: '9.5px',
            textTransform: 'none',
            letterSpacing: '0.04em',
            color: 'var(--ink-muted)',
            zIndex: 10,
            textAlign: 'center',
            pointerEvents: 'none',
          }}
        >
          Illustration. RGB and depth rendered from a CC0 photogrammetry scan; sparse input simulated (3.0%).
        </div>
      </div>
    </section>
  );
};
