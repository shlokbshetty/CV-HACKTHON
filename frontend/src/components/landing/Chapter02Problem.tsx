import React, { useState, useRef, useCallback } from 'react';

const PROBLEMS = [
  {
    num: '01',
    title: 'Missing measurements',
    desc: 'LiDAR and stereo sensors leave irregular occlusions and non-reflective voids.',
  },
  {
    num: '02',
    title: 'Sensor noise',
    desc: 'Time-of-flight multipath interference degrades planar precision across surfaces.',
  },
  {
    num: '03',
    title: 'Depth discontinuities',
    desc: 'Steep depth gradients round off, forming flying pixel web artefacts in 3D.',
  },
  {
    num: '04',
    title: 'Blurred boundaries',
    desc: 'Standard polynomial upsampling averages foreground and background distances.',
  },
  {
    num: '05',
    title: 'Texture-copy artefacts',
    desc: 'Naive RGB guidance burns high-frequency flat textures into 3D topography.',
  },
  {
    num: '06',
    title: 'Lost geometric structure',
    desc: 'Surface normals lose sharp dihedral ridges, rounding planar architectural facets.',
  },
];

export const Chapter02Problem: React.FC = () => {
  const [sliderPos, setSliderPos] = useState(50); // percentage 0 to 100
  const containerRef = useRef<HTMLDivElement | null>(null);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    setSliderPos((x / rect.width) * 100);
  }, []);

  return (
    <section
      id="ch02"
      style={{
        minHeight: '100vh',
        padding: 'clamp(var(--space-8), 6vw, var(--space-16)) clamp(var(--space-4), 5vw, var(--space-12)) clamp(80px, 12vh, 120px)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        boxSizing: 'border-box',
        borderTop: '1px solid var(--hairline)',
        backgroundColor: 'var(--paper)',
      }}
    >
      <div style={{ maxWidth: '1180px', margin: '0 auto', width: '100%' }}>
        {/* Header */}
        <div className="type-label" style={{ marginBottom: 'var(--space-2)' }}>
          02 // PROBLEM
        </div>

        <h2 className="type-display" style={{ fontSize: 'clamp(36px, 4.8vw, 68px)', margin: '0 0 var(--space-8) 0' }}>
          Why interpolation is not enough.
        </h2>

        {/* Content Layout: Left Slider & Scanline, Right Typographic Problem Index */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: 'clamp(var(--space-6), 4vw, var(--space-12))',
            alignItems: 'start',
          }}
        >
          {/* Left: Interactive Draggable Before/After Slider (01-landing.md §D) */}
          <div>
            <div
              ref={containerRef}
              onPointerMove={handlePointerMove}
              style={{
                position: 'relative',
                width: '100%',
                aspectRatio: '1 / 1',
                borderRadius: 'var(--radius-landing)',
                border: '1px solid var(--hairline)',
                overflow: 'hidden',
                cursor: 'ew-resize',
                userSelect: 'none',
                touchAction: 'none',
              }}
            >
              {/* Hybrid (Right image - Underneath) */}
              <img
                src="/assets/samples/ch02_hybrid.png"
                alt="DepthLab Hybrid reconstruction"
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                }}
              />

              {/* Bicubic (Left image - Clipped) */}
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: `${sliderPos}%`,
                  overflow: 'hidden',
                  borderRight: '1.5px solid var(--ink)',
                }}
              >
                <img
                  src="/assets/samples/ch02_bicubic.png"
                  alt="Standard Bicubic interpolation"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    height: '100%',
                    width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%',
                    maxWidth: 'none',
                    objectFit: 'cover',
                  }}
                />
              </div>

              {/* Badges */}
              <div
                className="type-label"
                style={{
                  position: 'absolute',
                  top: '12px',
                  left: '12px',
                  backgroundColor: 'var(--paper)',
                  padding: '3px 8px',
                  borderRadius: '2px',
                  border: '1px solid var(--hairline)',
                }}
              >
                Bicubic (4x blur)
              </div>

              <div
                className="type-label"
                style={{
                  position: 'absolute',
                  top: '12px',
                  right: '12px',
                  backgroundColor: 'var(--paper)',
                  padding: '3px 8px',
                  borderRadius: '2px',
                  border: '1px solid var(--hairline)',
                }}
              >
                Hybrid (Edge-preserved)
              </div>

              {/* Center Hairline Slider Handle */}
              <div
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: `${sliderPos}%`,
                  transform: 'translate(-50%, -50%)',
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--paper)',
                  border: '1px solid var(--ink)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  pointerEvents: 'none',
                  fontSize: '10px',
                }}
              >
                ↔
              </div>
            </div>

            {/* Scanline Profile Plot (01-landing.md §D) */}
            <div
              style={{
                marginTop: 'var(--space-3)',
                padding: 'var(--space-3)',
                border: '1px solid var(--hairline)',
                borderRadius: 'var(--radius-landing)',
              }}
            >
              <div className="type-label" style={{ marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                <span>1D Depth Scanline Profile across Discontinuity</span>
                <span className="type-chapter-num">Slice Y: 128px</span>
              </div>

              {/* SVG 1D Scanline Graph */}
              <svg width="100%" height="56" viewBox="0 0 256 56" preserveAspectRatio="none" style={{ display: 'block' }}>
                {/* Grid lines */}
                <line x1="0" y1="28" x2="256" y2="28" stroke="var(--hairline)" strokeWidth="0.8" strokeDasharray="3 3" />
                {/* Bicubic blurred curve (smooth transition) */}
                <path
                  d="M 0 46 C 80 46, 120 12, 256 12"
                  fill="none"
                  stroke="var(--ink-muted)"
                  strokeWidth="1.2"
                  strokeDasharray="4 2"
                />
                {/* Hybrid sharp step edge (instant step) */}
                <path
                  d="M 0 46 L 126 46 L 128 12 L 256 12"
                  fill="none"
                  stroke="var(--ink)"
                  strokeWidth="1.8"
                />
              </svg>

              <div className="type-label" style={{ marginTop: '4px', display: 'flex', gap: '16px', fontSize: '9px' }}>
                <span style={{ color: 'var(--ink)' }}>— Hybrid: sharp jump</span>
                <span style={{ color: 'var(--ink-muted)' }}>- - Bicubic: blurred slope</span>
              </div>
            </div>
          </div>

          {/* Right: Hairline-separated Typographic Problem Index (01-landing.md §D) */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {PROBLEMS.map((prob) => (
              <div
                key={prob.num}
                style={{
                  borderTop: '1px solid var(--hairline)',
                  padding: 'var(--space-3) 0',
                  display: 'grid',
                  gridTemplateColumns: '40px 1fr',
                  gap: 'var(--space-3)',
                  alignItems: 'baseline',
                }}
              >
                <span className="type-chapter-num" style={{ color: 'var(--ink-muted)' }}>
                  {prob.num}
                </span>
                <div>
                  <h3
                    style={{
                      fontFamily: 'var(--font-display)',
                      fontSize: '16px',
                      fontWeight: 500,
                      margin: 0,
                      color: 'var(--ink)',
                    }}
                  >
                    {prob.title}
                  </h3>
                  <p
                    style={{
                      fontSize: '12.5px',
                      color: 'var(--ink-muted)',
                      margin: '4px 0 0 0',
                      lineHeight: '1.45',
                    }}
                  >
                    {prob.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
