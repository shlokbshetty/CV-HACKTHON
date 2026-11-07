import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';

export const Chapter06Final: React.FC = () => {
  return (
    <section
      id="ch06"
      style={{
        minHeight: '100vh',
        padding: 'clamp(var(--space-8), 6vw, var(--space-16)) clamp(var(--space-4), 5vw, var(--space-12)) clamp(80px, 12vh, 120px)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        boxSizing: 'border-box',
        borderTop: '1px solid var(--hairline)',
        backgroundColor: 'var(--paper)',
      }}
    >
      <div
        style={{
          maxWidth: '1180px',
          margin: 'auto auto',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
          padding: 'var(--space-8) 0',
        }}
      >
        {/* Header */}
        <div className="type-label" style={{ marginBottom: 'var(--space-3)' }}>
          06 // WORKSPACE
        </div>

        <h2 className="type-display" style={{ margin: '0 0 var(--space-4) 0', maxWidth: '820px' }}>
          See your depth data in three dimensions.
        </h2>

        <p
          style={{
            maxWidth: '560px',
            fontSize: '14px',
            lineHeight: '1.6',
            color: 'var(--ink-muted)',
            margin: '0 0 var(--space-8) 0',
          }}
        >
          Upload your depth measurements, run reconstruction and explore the resulting geometry.
        </p>

        {/* Large Ring CTA Button (01-landing.md §D line 70) */}
        <Link
          to="/app"
          style={{
            width: '180px',
            height: '180px',
            borderRadius: '50%',
            border: '1.5px solid var(--ink)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            color: 'var(--ink)',
            textDecoration: 'none',
            transition: 'transform var(--duration-ui) var(--ease-depthlab), background-color var(--duration-ui) ease',
          }}
        >
          <span
            style={{
              fontFamily: 'var(--font-display)',
              fontSize: '14.5px',
              fontWeight: 500,
              letterSpacing: '0.04em',
            }}
          >
            Open workspace
          </span>
          <ArrowUpRight size={18} strokeWidth={1.5} />
        </Link>
      </div>

      {/* Footer Rows (01-landing.md §D) */}
      <footer
        style={{
          maxWidth: '1180px',
          margin: '0 auto',
          width: '100%',
          borderTop: '1px solid var(--hairline)',
          paddingTop: 'var(--space-6)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 'var(--space-4)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-2)' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: '13px', color: 'var(--ink)' }}>
            DEPTHLAB
          </span>
          <span className="type-label" style={{ fontSize: '9px', textTransform: 'none' }}>
            © 2026. Research build.
          </span>
        </div>

        {/* Navigation links */}
        <div style={{ display: 'flex', gap: 'var(--space-6)' }}>
          <a href="#ch01" className="type-label" style={{ color: 'var(--ink)' }}>
            Top
          </a>
          <a href="#ch02" className="type-label" style={{ color: 'var(--ink)' }}>
            Problem
          </a>
          <a href="#ch03" className="type-label" style={{ color: 'var(--ink)' }}>
            Method
          </a>
          <a href="#ch04" className="type-label" style={{ color: 'var(--ink)' }}>
            Benchmark
          </a>
          <Link to="/app" className="type-label" style={{ color: 'var(--ink)' }}>
            Workspace
          </Link>
        </div>

        <div className="type-label" style={{ fontSize: '9px', color: 'var(--ink-muted)', textTransform: 'none' }}>
          Crafted with React 18, Three.js, and General Sans.
        </div>
      </footer>
    </section>
  );
};
