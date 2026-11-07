import React from 'react';

const PIPELINE_STAGES = [
  {
    step: '01',
    title: 'Input acquisition',
    formula: 'D_sparse(u,v) + I_rgb(u,v)',
    ops: [
      'Raw float32 numeric ingestion',
      'Mask invalid pixels (NaN / 0)',
      'Intrinsics calibration check',
    ],
  },
  {
    step: '02',
    title: 'Classical CV',
    formula: 'J(p) = (1/W) Σ G_s G_r D(q)',
    ops: [
      'Joint bilateral guidance',
      'Normalized convolution',
      'Guided filter O(N) smoothing',
    ],
  },
  {
    step: '03',
    title: 'Deep learning',
    formula: 'R = ResNet-SR(I_rgb, D_init)',
    ops: [
      'Learned structural priors',
      'Residual surface refinement',
      'Multi-scale feature fusion',
    ],
  },
  {
    step: '04',
    title: 'Hybrid refinement',
    formula: '∇D_out = Gate(∇I, ∇D)',
    ops: [
      'Texture-copy suppression',
      'Discontinuity preservation',
      'Back-project to 3D geometry',
    ],
  },
];

export const Chapter03Method: React.FC = () => {
  return (
    <section
      id="ch03"
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
          03 // METHOD
        </div>

        <h2 className="type-display" style={{ fontSize: 'clamp(36px, 4.8vw, 68px)', margin: '0 0 var(--space-8) 0' }}>
          A hybrid approach.
        </h2>

        {/* 4-Stage Horizontal Pipeline Architecture (01-landing.md §D) */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: 'var(--space-4)',
            borderTop: '1px solid var(--hairline)',
            borderBottom: '1px solid var(--hairline)',
            padding: 'var(--space-6) 0',
            position: 'relative',
          }}
        >
          {PIPELINE_STAGES.map((st, i) => (
            <div
              key={st.step}
              style={{
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                paddingRight: 'var(--space-4)',
                borderRight: i < PIPELINE_STAGES.length - 1 ? '1px solid var(--hairline)' : 'none',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-2)' }}>
                  <span className="type-chapter-num" style={{ color: 'var(--ink-muted)' }}>
                    STAGE // {st.step}
                  </span>
                  {i < PIPELINE_STAGES.length - 1 && (
                    <span className="desktop-only" style={{ color: 'var(--ink-muted)', fontSize: '12px' }}>
                      →
                    </span>
                  )}
                </div>

                <h3
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: '18px',
                    fontWeight: 500,
                    margin: '0 0 var(--space-3) 0',
                    color: 'var(--ink)',
                  }}
                >
                  {st.title}
                </h3>

                {/* Mathematical Formulation Tag */}
                <div
                  className="type-mono"
                  style={{
                    fontSize: '10.5px',
                    padding: '4px 8px',
                    borderRadius: 'var(--radius-landing)',
                    border: '1px solid var(--hairline)',
                    backgroundColor: 'transparent',
                    color: 'var(--ink)',
                    marginBottom: 'var(--space-4)',
                    display: 'inline-block',
                  }}
                >
                  {st.formula}
                </div>

                {/* Real Operations List */}
                <ul
                  style={{
                    listStyle: 'none',
                    margin: 0,
                    padding: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '6px',
                  }}
                >
                  {st.ops.map((op, j) => (
                    <li
                      key={j}
                      style={{
                        fontSize: '12px',
                        color: 'var(--ink-muted)',
                        display: 'flex',
                        alignItems: 'baseline',
                        gap: '6px',
                      }}
                    >
                      <span style={{ fontSize: '8px', color: 'var(--ink)' }}>•</span>
                      <span>{op}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        {/* Technical Sub-description */}
        <div
          style={{
            marginTop: 'var(--space-6)',
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            alignItems: 'baseline',
            gap: 'var(--space-4)',
          }}
        >
          <p style={{ maxWidth: '640px', fontSize: '13px', lineHeight: '1.6', color: 'var(--ink-muted)', margin: 0 }}>
            Classical bilateral filters prevent non-local bleeding. The lightweight neural prior restores high-frequency
            geometric relief. Gradient-gated refinement eliminates false texture copies from planar surfaces.
          </p>

          <span className="type-label" style={{ fontSize: '9.5px', color: 'var(--ink-muted)' }}>
            Pipeline latency: 48.7ms (CPU) // 4x scale
          </span>
        </div>
      </div>
    </section>
  );
};
