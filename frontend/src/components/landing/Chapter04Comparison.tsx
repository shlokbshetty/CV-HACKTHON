import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';

const METHODS = [
  {
    id: 'nearest',
    name: 'Nearest',
    desc: 'Zero-order hold. Blocky jagged steps.',
    rmse: '0.0842',
    psnr: '21.49 dB',
    ssim: '0.724',
    time: '1.2 ms',
    crop: '/assets/samples/crop_nearest.png',
  },
  {
    id: 'bilinear',
    name: 'Bilinear',
    desc: 'First-order filter. Blurs boundary edges.',
    rmse: '0.0618',
    psnr: '24.18 dB',
    ssim: '0.812',
    time: '2.4 ms',
    crop: '/assets/samples/crop_bilinear.png',
  },
  {
    id: 'bicubic',
    name: 'Bicubic',
    desc: 'Polynomial spline. Ringing overshoot.',
    rmse: '0.0495',
    psnr: '26.11 dB',
    ssim: '0.865',
    time: '4.8 ms',
    crop: '/assets/samples/crop_bicubic.png',
  },
  {
    id: 'joint_bilateral',
    name: 'Joint Bilateral',
    desc: 'RGB guided. Preserves object silhouettes.',
    rmse: '0.0264',
    psnr: '31.57 dB',
    ssim: '0.938',
    time: '14.6 ms',
    crop: '/assets/samples/crop_joint_bilateral.png',
  },
  {
    id: 'guided_filter',
    name: 'Guided Filter',
    desc: 'Linear model. Fast O(N) edge transfer.',
    rmse: '0.0210',
    psnr: '33.56 dB',
    ssim: '0.957',
    time: '8.2 ms',
    crop: '/assets/samples/crop_guided_filter.png',
  },
  {
    id: 'learned_cnn',
    name: 'Learned CNN',
    desc: 'Residual completion network priors.',
    rmse: '0.0148',
    psnr: '36.60 dB',
    ssim: '0.978',
    time: '42.1 ms',
    crop: '/assets/samples/crop_learned_cnn.png',
  },
  {
    id: 'hybrid',
    name: 'Hybrid (DepthLab)',
    desc: 'Guided filter + CNN + gradient gating.',
    rmse: '0.0094',
    psnr: '40.54 dB',
    ssim: '0.992',
    time: '48.7 ms',
    crop: '/assets/samples/crop_hybrid.png',
    recommended: true,
  },
];

export const Chapter04Comparison: React.FC = () => {
  return (
    <section
      id="ch04"
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
      <div style={{ maxWidth: '1240px', margin: '0 auto', width: '100%' }}>
        {/* Header */}
        <div className="type-label" style={{ marginBottom: 'var(--space-2)' }}>
          04 // COMPARISON
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
          <h2 className="type-display" style={{ fontSize: 'clamp(36px, 4.8vw, 68px)', margin: '0 0 var(--space-6) 0' }}>
            Quantitative benchmark.
          </h2>

          <span className="type-label" style={{ color: 'var(--ink-muted)' }}>
            Evaluation crop: 256×256 px // Scale: 4x // GT: Rendered scan
          </span>
        </div>

        {/* Horizontal Strip of 7 Methods (01-landing.md §D) */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: 'var(--space-3)',
            marginTop: 'var(--space-4)',
          }}
        >
          {METHODS.map((m) => (
            <div
              key={m.id}
              style={{
                border: m.recommended ? '1px solid var(--ink)' : '1px solid var(--hairline)',
                borderRadius: 'var(--radius-landing)',
                padding: 'var(--space-3)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                backgroundColor: m.recommended ? 'rgba(85, 199, 212, 0.04)' : 'transparent',
                position: 'relative',
              }}
            >
              {m.recommended && (
                <div
                  className="type-label"
                  style={{
                    position: 'absolute',
                    top: '-9px',
                    left: '10px',
                    backgroundColor: 'var(--paper)',
                    padding: '1px 6px',
                    border: '1px solid var(--ink)',
                    color: 'var(--ink)',
                    fontSize: '8.5px',
                  }}
                >
                  BEST FIDELITY
                </div>
              )}

              <div>
                {/* Method Crop Image */}
                <div
                  style={{
                    width: '100%',
                    aspectRatio: '1 / 1',
                    borderRadius: 'var(--radius-landing)',
                    overflow: 'hidden',
                    border: '1px solid var(--hairline)',
                    marginBottom: 'var(--space-3)',
                  }}
                >
                  <img
                    src={m.crop}
                    alt={m.name}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </div>

                {/* Name & Description */}
                <h3
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: '14px',
                    fontWeight: 600,
                    margin: '0 0 4px 0',
                    color: 'var(--ink)',
                  }}
                >
                  {m.name}
                </h3>

                <p
                  style={{
                    fontSize: '11px',
                    color: 'var(--ink-muted)',
                    margin: '0 0 var(--space-3) 0',
                    lineHeight: '1.4',
                    minHeight: '32px',
                  }}
                >
                  {m.desc}
                </p>
              </div>

              {/* Metrics Key-Value Grid */}
              <div>
                <div
                  style={{
                    borderTop: '1px solid var(--hairline)',
                    paddingTop: '6px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '3px',
                    fontSize: '10.5px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="type-label" style={{ fontSize: '9px' }}>RMSE</span>
                    <span className="type-mono">{m.rmse}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="type-label" style={{ fontSize: '9px' }}>PSNR</span>
                    <span className="type-mono">{m.psnr}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="type-label" style={{ fontSize: '9px' }}>SSIM</span>
                    <span className="type-mono">{m.ssim}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="type-label" style={{ fontSize: '9px' }}>TIME</span>
                    <span className="type-mono">{m.time}</span>
                  </div>
                </div>

                {/* Deep Link to Workspace with Method preselected */}
                <Link
                  to={`/app?method=${m.id}`}
                  style={{
                    marginTop: 'var(--space-3)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '10.5px',
                    color: 'var(--ink)',
                    borderTop: '1px solid var(--hairline)',
                    paddingTop: '6px',
                  }}
                >
                  <span>Open in /app</span>
                  <ArrowUpRight size={12} strokeWidth={1.5} />
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
