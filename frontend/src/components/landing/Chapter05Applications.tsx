import React, { useState } from 'react';

const APPLICATIONS = [
  {
    num: '01',
    name: 'Robotics and navigation',
    spec: 'Obstacle boundary clarity for real-time mobile path planning.',
    crop: '/assets/samples/crop_hybrid.png',
  },
  {
    num: '02',
    name: '3D scene understanding',
    spec: 'Instance segmentation with crisp silhouette edge recovery.',
    crop: '/assets/samples/crop_guided_filter.png',
  },
  {
    num: '03',
    name: 'Spatial computing & XR',
    spec: 'Occlusion-correct neural passthrough mesh reconstruction.',
    crop: '/assets/samples/crop_learned_cnn.png',
  },
  {
    num: '04',
    name: 'Industrial inspection',
    spec: 'Micro-crack surface depth profiling without laser distortion.',
    crop: '/assets/samples/crop_joint_bilateral.png',
  },
  {
    num: '05',
    name: 'Indoor mapping',
    spec: 'LiDAR point cloud densification for architectural CAD export.',
    crop: '/assets/samples/ch02_hybrid.png',
  },
  {
    num: '06',
    name: 'Computer vision research',
    spec: 'Rigorous benchmark for edge-preserving depth super-resolution.',
    crop: '/assets/samples/crop_bicubic.png',
  },
];

export const Chapter05Applications: React.FC = () => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  return (
    <section
      id="ch05"
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
          05 // APPLICATIONS
        </div>

        <h2 className="type-display" style={{ fontSize: 'clamp(36px, 4.8vw, 68px)', margin: '0 0 var(--space-8) 0' }}>
          Where precision depth matters.
        </h2>

        {/* Large Typographic List with Hairline Separators (01-landing.md §D) */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {APPLICATIONS.map((app, index) => {
            const isHovered = hoveredIdx === index;
            return (
              <div
                key={app.num}
                onMouseEnter={() => setHoveredIdx(index)}
                onMouseLeave={() => setHoveredIdx(null)}
                style={{
                  borderTop: '1px solid var(--hairline)',
                  padding: 'clamp(var(--space-4), 2.5vw, var(--space-6)) 0',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  position: 'relative',
                  transition: 'padding-left var(--duration-ui) var(--ease-depthlab)',
                  paddingLeft: isHovered ? '12px' : '0px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-6)' }}>
                  <span className="type-chapter-num" style={{ color: 'var(--ink-muted)' }}>
                    {app.num}
                  </span>
                  <div>
                    <h3
                      style={{
                        fontFamily: 'var(--font-display)',
                        fontSize: 'clamp(20px, 2.5vw, 32px)',
                        fontWeight: 400,
                        margin: 0,
                        color: 'var(--ink)',
                      }}
                    >
                      {app.name}
                    </h3>
                    <p
                      style={{
                        fontSize: '12.5px',
                        color: 'var(--ink-muted)',
                        margin: '4px 0 0 0',
                      }}
                    >
                      {app.spec}
                    </p>
                  </div>
                </div>

                {/* Floating Preview Crop on Hover (01-landing.md §D line 69) */}
                <div
                  className="desktop-only"
                  style={{
                    width: '84px',
                    height: '84px',
                    borderRadius: 'var(--radius-landing)',
                    overflow: 'hidden',
                    border: '1px solid var(--hairline)',
                    opacity: isHovered ? 1.0 : 0.25,
                    transform: isHovered ? 'scale(1.08)' : 'scale(1.0)',
                    transition: 'all 200ms var(--ease-depthlab)',
                  }}
                >
                  <img
                    src={app.crop}
                    alt={app.name}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </div>
              </div>
            );
          })}
          <div style={{ borderBottom: '1px solid var(--hairline)' }} />
        </div>
      </div>
    </section>
  );
};
