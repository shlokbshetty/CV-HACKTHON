import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, ChevronUp, Sun, Moon } from 'lucide-react';
import { useThemeStore } from '../../stores/theme';

export interface ReadoutData {
  stageNum: string;
  stageName: string;
  input: string;
  scale: string;
  geometry: string;
}

interface StageChromeProps {
  currentChapter: number;
  totalChapters: number;
  onChapterSelect: (index: number) => void;
  scrollProgress: number; // 0.0 to 1.0
  readouts: ReadoutData;
}

const CHAPTERS = [
  { id: 'ch01', label: '01 HERO' },
  { id: 'ch02', label: '02 PROBLEM' },
  { id: 'ch03', label: '03 METHOD' },
  { id: 'ch04', label: '04 BENCHMARK' },
  { id: 'ch05', label: '05 APPLICATIONS' },
  { id: 'ch06', label: '06 LAUNCH' },
];

export const StageChrome: React.FC<StageChromeProps> = ({
  currentChapter,
  totalChapters,
  onChapterSelect,
  scrollProgress,
  readouts,
}) => {
  const { theme, toggleTheme } = useThemeStore();

  const isFirstChapter = currentChapter === 1;
  const isLastChapter = currentChapter === totalChapters;

  return (
    <>
      {/* ── Fixed Top Bar ── */}
      <header
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: '56px',
          padding: '0 clamp(var(--space-4), 3.5vw, var(--space-8))',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          zIndex: 40,
          pointerEvents: 'none',
        }}
      >
        {/* Top-left: Wordmark & Research build note */}
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: 'var(--space-2)',
            pointerEvents: 'auto',
          }}
        >
          <span
            style={{
              fontFamily: 'var(--font-display)',
              fontWeight: 600,
              letterSpacing: '0.04em',
              fontSize: '15px',
              color: 'var(--ink)',
            }}
          >
            DEPTHLAB<sup style={{ fontSize: '9px', marginLeft: '1px' }}>®</sup>
          </span>
          <span
            className="type-label desktop-only"
            style={{ textTransform: 'none', letterSpacing: '0.02em' }}
          >
            <span style={{ color: 'var(--ink)' }}>Edge-preserving depth reconstruction. </span>
            <span style={{ color: 'var(--ink-muted)' }}>Research build.</span>
          </span>
        </div>

        {/* Top-centre: Geometric Logomark (SVG Concentric Arcs) */}
        <div
          style={{
            position: 'absolute',
            left: '50%',
            transform: 'translateX(-50%)',
            pointerEvents: 'auto',
          }}
          className="desktop-only"
          aria-label="DepthLab concentric logomark"
        >
          <svg width="26" height="26" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="12" stroke="currentColor" strokeWidth="1" strokeOpacity="0.2" />
            <circle cx="14" cy="14" r="8" stroke="currentColor" strokeWidth="1" strokeOpacity="0.45" />
            <circle cx="14" cy="14" r="4" stroke="currentColor" strokeWidth="1" strokeOpacity="0.8" />
            <circle cx="14" cy="14" r="1.5" fill="currentColor" />
          </svg>
        </div>

        {/* Top-right: Day/Night Switch + Launch Workspace */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-4)',
            pointerEvents: 'auto',
          }}
        >
          {/* Day / Night switch pill track */}
          <button
            onClick={toggleTheme}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '999px',
              border: '1px solid var(--hairline)',
              backgroundColor: 'var(--paper)',
              color: 'var(--ink)',
              fontSize: '11px',
              letterSpacing: '0.06em',
              transition: 'background-color var(--duration-ui) var(--ease-depthlab)',
            }}
            aria-label="Toggle Day / Night theme"
          >
            {theme === 'day' ? <Sun size={12} strokeWidth={1.5} /> : <Moon size={12} strokeWidth={1.5} />}
            <span style={{ fontWeight: theme === 'day' ? 600 : 400, color: theme === 'day' ? 'var(--ink)' : 'var(--ink-muted)' }}>
              Day
            </span>
            <span style={{ color: 'var(--ink-muted)' }}>/</span>
            <span style={{ fontWeight: theme === 'night' ? 600 : 400, color: theme === 'night' ? 'var(--ink)' : 'var(--ink-muted)' }}>
              Night
            </span>
          </button>

          {/* Launch workspace button */}
          <Link
            to="/app"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              fontSize: '12px',
              fontWeight: 500,
              letterSpacing: '0.04em',
              borderBottom: '1px solid var(--ink)',
              paddingBottom: '2px',
              color: 'var(--ink)',
            }}
          >
            Launch workspace
            <ArrowUpRight size={13} strokeWidth={1.5} />
          </Link>
        </div>
      </header>

      {/* ── Left & Right 120px Circular Ring Controls (01-landing.md §A) ── */}
      {!isFirstChapter && (
        <button
          onClick={() => onChapterSelect(currentChapter - 1)}
          className="desktop-only"
          style={{
            position: 'fixed',
            left: '24px',
            top: '50%',
            transform: 'translateY(-50%)',
            width: '110px',
            height: '110px',
            borderRadius: '50%',
            border: '1px solid var(--hairline)',
            backgroundColor: 'transparent',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 35,
            color: 'var(--ink)',
            fontSize: '11.5px',
            letterSpacing: '0.08em',
            transition: 'border-color var(--duration-ui) var(--ease-depthlab), opacity var(--duration-ui) ease',
          }}
          aria-label="Jump to previous chapter"
        >
          <span className="type-label" style={{ color: 'var(--ink)' }}>Back</span>
        </button>
      )}

      {!isLastChapter && (
        <button
          onClick={() => onChapterSelect(currentChapter + 1)}
          className="desktop-only"
          style={{
            position: 'fixed',
            right: '24px',
            top: '50%',
            transform: 'translateY(-50%)',
            width: '110px',
            height: '110px',
            borderRadius: '50%',
            border: '1px solid var(--hairline)',
            backgroundColor: 'transparent',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 35,
            color: 'var(--ink)',
            fontSize: '11.5px',
            letterSpacing: '0.08em',
            transition: 'border-color var(--duration-ui) var(--ease-depthlab), opacity var(--duration-ui) ease',
          }}
          aria-label="Jump to next chapter"
        >
          <span className="type-label" style={{ color: 'var(--ink)' }}>Next</span>
        </button>
      )}

      {/* ── Bottom-Left Mono Chapter Counter: 01 / 06 ── */}
      <div
        className="desktop-only"
        style={{
          position: 'fixed',
          bottom: '24px',
          left: 'clamp(var(--space-4), 3.5vw, var(--space-8))',
          zIndex: 40,
          pointerEvents: 'none',
        }}
      >
        <span className="type-chapter-num" style={{ color: 'var(--ink)' }}>
          {String(currentChapter).padStart(2, '0')}{' '}
          <span style={{ color: 'var(--ink-muted)' }}>//</span>{' '}
          {String(totalChapters).padStart(2, '0')}
        </span>
      </div>

      {/* ── Bottom-Right Vertical Scroll Progress Meter ── */}
      <div
        className="desktop-only"
        style={{
          position: 'fixed',
          bottom: '24px',
          right: 'clamp(var(--space-4), 3.5vw, var(--space-8))',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '3px',
          zIndex: 40,
          pointerEvents: 'none',
        }}
        aria-label="Scroll progress meter"
      >
        {Array.from({ length: 8 }).map((_, i) => {
          const tickThresh = i / 7;
          const isFilled = scrollProgress >= tickThresh;
          return (
            <span
              key={i}
              style={{
                width: '12px',
                height: '1px',
                backgroundColor: isFilled ? 'var(--ink)' : 'var(--hairline)',
                transition: 'background-color 120ms ease',
              }}
            />
          );
        })}
        <span
          className="type-label"
          style={{
            fontSize: '8px',
            marginTop: '3px',
            letterSpacing: '0.15em',
            color: 'var(--ink-muted)',
          }}
        >
          SCROLL
        </span>
      </div>

      {/* ── Live Readout Row & Chapter Navigation (Bottom Center) ── */}
      <div
        style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: 'clamp(320px, 86vw, 920px)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 'var(--space-3)',
          zIndex: 40,
        }}
      >
        {/* 4-Column Live Readout Row (visible during Chapter 01 Hero sequence per 01-landing.md §A) */}
        {currentChapter === 1 && (
          <div
            style={{
              width: '100%',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
              gap: 'var(--space-4)',
              padding: '10px 18px',
              borderRadius: 'var(--radius-landing)',
              border: '1px solid var(--hairline)',
              backgroundColor: 'var(--paper)',
              backdropFilter: 'blur(8px)',
              transition: 'opacity 240ms ease',
            }}
          >
            <div>
              <div className="type-label">STAGE</div>
              <div className="type-readout-val" style={{ marginTop: '2px', color: 'var(--ink)' }}>
                {readouts.stageNum} <span style={{ fontSize: '11px', color: 'var(--ink-muted)' }}>{readouts.stageName}</span>
              </div>
            </div>
            <div>
              <div className="type-label">INPUT</div>
              <div className="type-readout-val" style={{ marginTop: '2px', color: 'var(--ink)' }}>
                {readouts.input}
              </div>
            </div>
            <div>
              <div className="type-label">SCALE</div>
              <div className="type-readout-val" style={{ marginTop: '2px', color: 'var(--ink)' }}>
                {readouts.scale}
              </div>
            </div>
            <div>
              <div className="type-label">GEOMETRY</div>
              <div className="type-readout-val" style={{ marginTop: '2px', color: 'var(--ink)' }}>
                {readouts.geometry}
              </div>
            </div>
          </div>
        )}

        {/* Chapter Navigation with Active Chevron Indicator */}
        <nav
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'clamp(6px, 1.8vw, 20px)',
            padding: '6px 16px',
            borderRadius: 'var(--radius-landing)',
            border: '1px solid var(--hairline)',
            backgroundColor: 'var(--paper)',
            backdropFilter: 'blur(12px)',
          }}
          aria-label="Chapter navigation"
        >
          {CHAPTERS.map((ch, index) => {
            const num = index + 1;
            const isActive = currentChapter === num;
            return (
              <button
                key={ch.id}
                onClick={() => onChapterSelect(num)}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  background: 'none',
                  border: 'none',
                  padding: '4px 6px',
                  color: isActive ? 'var(--ink)' : 'var(--ink-muted)',
                  cursor: 'pointer',
                  position: 'relative',
                  transition: 'color var(--duration-ui) var(--ease-depthlab)',
                }}
              >
                <span
                  className="type-label"
                  style={{
                    color: isActive ? 'var(--ink)' : 'var(--ink-muted)',
                    fontWeight: isActive ? 600 : 400,
                  }}
                >
                  <span className="desktop-only">{ch.label}</span>
                  <span
                    className="mobile-only"
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '11px',
                      letterSpacing: '0.04em',
                    }}
                  >
                    {String(num).padStart(2, '0')}
                  </span>
                </span>
                {isActive && (
                  <ChevronUp
                    size={11}
                    strokeWidth={2}
                    style={{
                      position: 'absolute',
                      bottom: '-8px',
                      color: 'var(--ink)',
                    }}
                  />
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </>
  );
};
