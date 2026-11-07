import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { CustomCursor } from '../../components/shared/CustomCursor';
import { ArrowUpRight, Box, Activity, Layers } from 'lucide-react';
import Lenis from 'lenis';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export const LandingV3: React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const butterflyVideoRef = useRef<HTMLVideoElement | null>(null);
  const heroRef = useRef<HTMLDivElement | null>(null);
  const metricsHudRef = useRef<HTMLDivElement | null>(null);

  const [videoDuration, setVideoDuration] = useState<number>(9.96);
  const [isVideoLoaded, setIsVideoLoaded] = useState<boolean>(false);
  const [videoSrc, setVideoSrc] = useState<string>('/videos/butterfly-scroll.mp4');
  const [scrollProgress, setScrollProgress] = useState<number>(0);

  const targetTimeRef = useRef<number>(0);
  const currentTimeRef = useRef<number>(0);
  const pendingTargetTimeRef = useRef<number | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const lenisRef = useRef<Lenis | null>(null);

  // ── 1. Prefetch video into RAM Blob to eliminate HTTP Range CDN latency on Vercel ──
  useEffect(() => {
    let isMounted = true;
    fetch('/videos/butterfly-scroll.mp4')
      .then(res => res.blob())
      .then(blob => {
        if (isMounted) {
          const blobUrl = URL.createObjectURL(blob);
          setVideoSrc(blobUrl);
        }
      })
      .catch(() => {
        // Fallback to static path
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // ── 2. Initialize Lenis Smooth Inertia Momentum Scroll ──
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
      touchMultiplier: 1.5,
    });
    lenisRef.current = lenis;

    lenis.on('scroll', ScrollTrigger.update);

    const updateLenis = (time: number) => {
      lenis.raf(time * 1000);
    };

    gsap.ticker.add(updateLenis);
    gsap.ticker.lagSmoothing(0);

    return () => {
      gsap.ticker.remove(updateLenis);
      lenis.destroy();
    };
  }, []);

  // ── 3. Apple Design Hero Entrance Animation ──
  useEffect(() => {
    if (!heroRef.current) return;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        '.hero-anim-item',
        {
          y: 35,
          opacity: 0,
          scale: 0.98,
        },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          duration: 1.1,
          ease: 'power3.out',
          stagger: 0.12,
          delay: 0.2,
        }
      );
    }, heroRef);

    return () => ctx.revert();
  }, []);

  // ── 4. Set video duration when metadata loads ──
  const handleLoadedMetadata = () => {
    if (butterflyVideoRef.current) {
      setVideoDuration(butterflyVideoRef.current.duration || 9.96);
      setIsVideoLoaded(true);
    }
  };

  // ── 5. Hardware Seek-Lock: only seek when decoder is ready, buffering subsequent positions ──
  const requestSeek = (time: number) => {
    const v = butterflyVideoRef.current;
    if (!v) return;

    if (v.seeking) {
      pendingTargetTimeRef.current = time;
    } else {
      v.currentTime = time;
    }
  };

  const handleSeeked = () => {
    const v = butterflyVideoRef.current;
    if (!v) return;

    if (pendingTargetTimeRef.current !== null) {
      const nextTime = pendingTargetTimeRef.current;
      pendingTargetTimeRef.current = null;
      v.currentTime = nextTime;
    }
  };

  // ── 6. Scroll listener tracking scroll progress & phase mapping ──
  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const maxScroll = (document.documentElement.scrollHeight - window.innerHeight) || 1;
      const progress = Math.min(1, Math.max(0, scrollY / maxScroll));
      setScrollProgress(progress);

      // Phase 1 (0 to 14%): Hero text visible, video pinned at 0.0s
      // Phase 2 (14% to 85%): Video scrubs through 0.0s to duration
      // Phase 3 (85% to 100%): Metrics HUD visible, video at final frame
      let target = 0;
      const scrubStart = 0.14;
      const scrubEnd = 0.86;
      const duration = butterflyVideoRef.current?.duration || videoDuration;

      if (progress < scrubStart) {
        target = 0;
      } else if (progress > scrubEnd) {
        target = duration;
      } else {
        const ratio = (progress - scrubStart) / (scrubEnd - scrubStart);
        target = ratio * duration;
      }
      targetTimeRef.current = target;
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [videoDuration]);

  // ── 7. Smooth 60fps RAF loop with Seek-Lock execution (zero decoder backlog) ──
  useEffect(() => {
    const updateScrub = () => {
      if (butterflyVideoRef.current && isVideoLoaded) {
        const delta = targetTimeRef.current - currentTimeRef.current;
        if (Math.abs(delta) > 0.005) {
          currentTimeRef.current += delta * 0.25; // Responsive spring lerp
          requestSeek(currentTimeRef.current);
        }
      }
      rafIdRef.current = requestAnimationFrame(updateScrub);
    };

    rafIdRef.current = requestAnimationFrame(updateScrub);
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [isVideoLoaded]);

  // Derived phase opacity and transform for Apple fluid transitions
  // Hero: disappears over the first scroll gesture (0% to 14%)
  const heroOpacity = Math.max(0, 1 - scrollProgress / 0.12);
  const heroTranslateY = -60 * Math.min(1, scrollProgress / 0.12);
  const heroScale = 1 - 0.04 * Math.min(1, scrollProgress / 0.12);

  // Metrics HUD: reveals gracefully in the final stage (80% to 100%)
  const metricsOpacity = Math.min(1, Math.max(0, (scrollProgress - 0.78) / 0.15));
  const metricsTranslateY = 40 * (1 - metricsOpacity);
  const metricsScale = 0.95 + 0.05 * metricsOpacity;

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        maxWidth: '100%',
        height: '340vh', // Multi-screen scroll track
        backgroundColor: '#05070A',
        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", sans-serif',
        userSelect: 'none',
      }}
    >
      <CustomCursor />

      {/* ── Fixed Living Meadow Still Frame Background (Full Time, Zero Decode Overhead) ── */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          width: '100%',
          height: '100%',
          zIndex: 0,
          overflow: 'hidden',
          pointerEvents: 'none',
        }}
      >
        <img
          src="/images/bg-meadow-frame.jpg"
          alt="Living Meadow Background"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: 'block',
          }}
        />
        {/* Subtle dark gradient overlay to ensure contrast */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: 'radial-gradient(ellipse at center, rgba(5, 7, 10, 0.12) 0%, rgba(5, 7, 10, 0.65) 100%)',
          }}
        />
      </div>

      {/* ── Sticky Viewport Container ── */}
      <div
        style={{
          position: 'sticky',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          width: '100%',
          height: '100vh',
          zIndex: 10,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'space-between',
          overflow: 'hidden',
          padding: '24px 0',
          boxSizing: 'border-box',
        }}
      >
        {/* ── Full-Bleed Edge-to-Edge Butterfly Scroll Scrub Video (RAM Resident Blob) ── */}
        <video
          ref={butterflyVideoRef}
          muted
          playsInline
          preload="auto"
          src={videoSrc}
          onLoadedMetadata={handleLoadedMetadata}
          onSeeked={handleSeeked}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: 'block',
            zIndex: 1,
          }}
        />

        {/* ── Apple Liquid Glass Floating Navigation Bar (Top) ── */}
        <header
          style={{
            position: 'relative',
            zIndex: 100,
            display: 'flex',
            justifyContent: 'center',
            padding: '0 24px',
            width: '100%',
            maxWidth: '100%',
            pointerEvents: 'none',
            boxSizing: 'border-box',
          }}
        >
          <nav
            style={{
              pointerEvents: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: 'clamp(12px, 2.5vw, 24px)',
              padding: '8px clamp(14px, 2vw, 22px)',
              borderRadius: '999px',
              backgroundColor: 'rgba(255, 255, 255, 0.45)',
              backdropFilter: 'blur(36px) saturate(210%)',
              WebkitBackdropFilter: 'blur(36px) saturate(210%)',
              border: '1px solid rgba(255, 255, 255, 0.7)',
              boxShadow: '0 20px 45px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.8)',
              fontSize: '13px',
              fontWeight: 500,
              letterSpacing: '-0.01em',
              maxWidth: 'calc(100% - 32px)',
            }}
          >
            {/* Logo */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div
                style={{
                  width: '20px',
                  height: '20px',
                  borderRadius: '50%',
                  backgroundColor: 'rgba(15, 17, 23, 0.88)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 2px 6px rgba(0, 0, 0, 0.2)',
                }}
              >
                <Box size={11} color="#FFFFFF" />
              </div>
              <span style={{ fontWeight: 600, color: '#0F1117', fontSize: '13.5px' }}>Spatial Depth</span>
            </div>

            {/* Benchmarks Studio Link */}
            <Link
              to="/studio"
              style={{
                fontSize: '12.5px',
                fontWeight: 600,
                color: '#0F1117',
                textDecoration: 'none',
                padding: '4px 8px',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              <Activity size={12} />
              <span>Studio & Benchmarks</span>
            </Link>

            {/* Direct CTA into 3D Depth Studio */}
            <Link
              to="/app"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '7px 16px',
                borderRadius: '999px',
                backgroundColor: '#0F1117',
                color: '#FFFFFF',
                textDecoration: 'none',
                fontWeight: 600,
                fontSize: '12px',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.25)',
                transition: 'transform 0.15s ease',
              }}
            >
              <span>Open 3D Studio</span>
              <ArrowUpRight size={13} />
            </Link>
          </nav>
        </header>

        {/* ── Apple Design Hero Title & Tagline (Fades out cleanly on first scroll) ── */}
        <div
          ref={heroRef}
          style={{
            position: 'relative',
            zIndex: 50,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            textAlign: 'center',
            padding: '0 24px',
            maxWidth: '920px',
            pointerEvents: heroOpacity > 0.05 ? 'auto' : 'none',
            opacity: heroOpacity,
            transform: `translateY(${heroTranslateY}px) scale(${heroScale})`,
            transition: 'opacity 0.1s linear, transform 0.1s linear',
            marginTop: 'clamp(60px, 12.5vh, 135px)',
            marginBottom: 'auto',
          }}
        >
          {/* Main Headline (Apple Typography: optical kerning, bold, refined) */}
          <h1
            className="hero-anim-item"
            style={{
              fontSize: 'clamp(32px, 5.2vw, 64px)',
              fontWeight: 700,
              letterSpacing: '-0.035em',
              lineHeight: 1.08,
              color: '#FFFFFF',
              margin: '0 0 16px 0',
            }}
          >
            Edge-Preserving Depth Super-Resolution
          </h1>

          {/* Tagline Subtitle */}
          <p
            className="hero-anim-item"
            style={{
              fontSize: 'clamp(15px, 1.8vw, 19px)',
              fontWeight: 400,
              lineHeight: 1.45,
              color: 'rgba(255, 255, 255, 0.92)',
              letterSpacing: '-0.015em',
              maxWidth: '680px',
              margin: '0',
            }}
          >
            Sub-millimeter 3D depth reconstruction from sparse monocular sensor guidance, suppressing texture copying and snapping sharp geometric boundaries.
          </p>
        </div>

        {/* ── Apple Liquid Glass End-Scroll Benchmark Metrics HUD (Fades in at 80%+) ── */}
        <div
          ref={metricsHudRef}
          style={{
            position: 'relative',
            zIndex: 100,
            display: 'flex',
            justifyContent: 'center',
            width: '100%',
            maxWidth: '100%',
            padding: '0 24px',
            pointerEvents: metricsOpacity > 0.1 ? 'auto' : 'none',
            opacity: metricsOpacity,
            transform: `translateY(${metricsTranslateY}px) scale(${metricsScale})`,
            transition: 'opacity 0.15s ease-out, transform 0.15s ease-out',
            boxSizing: 'border-box',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 'clamp(10px, 1.6vw, 24px)',
              padding: '12px clamp(16px, 2vw, 28px)',
              borderRadius: '24px',
              backgroundColor: 'rgba(15, 17, 23, 0.82)',
              backdropFilter: 'blur(40px) saturate(210%)',
              WebkitBackdropFilter: 'blur(40px) saturate(210%)',
              border: '1px solid rgba(255, 255, 255, 0.22)',
              boxShadow: '0 30px 80px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.25)',
              maxWidth: '1080px',
              width: '100%',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
            }}
          >
            {/* Metric 1: Hybrid RMSE */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '10px', color: '#8A8F98', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                Hybrid RMSE ↓
              </span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                <span style={{ fontSize: '18px', fontWeight: 700, color: '#FFFFFF', letterSpacing: '-0.02em', fontFamily: 'monospace' }}>
                  0.2354 m
                </span>
                <span style={{ fontSize: '10.5px', color: '#10B981', fontWeight: 600 }}>
                  ↓ 69.8%
                </span>
              </div>
            </div>

            {/* Metric 2: Boundary RMSE */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '10px', color: '#8A8F98', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                Boundary RMSE ↓
              </span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                <span style={{ fontSize: '18px', fontWeight: 700, color: '#FFFFFF', letterSpacing: '-0.02em', fontFamily: 'monospace' }}>
                  0.4299 m
                </span>
                <span style={{ fontSize: '10.5px', color: '#10B981', fontWeight: 600 }}>
                  ↓ 54.7%
                </span>
              </div>
            </div>

            {/* Metric 3: Hybrid PSNR */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '10px', color: '#8A8F98', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                Hybrid PSNR ↑
              </span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                <span style={{ fontSize: '18px', fontWeight: 700, color: '#FFFFFF', letterSpacing: '-0.02em', fontFamily: 'monospace' }}>
                  24.88 dB
                </span>
                <span style={{ fontSize: '10.5px', color: '#38BDF8', fontWeight: 600 }}>
                  +10.4 dB
                </span>
              </div>
            </div>

            {/* Metric 4: Edge-F1 */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '10px', color: '#8A8F98', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                Edge-F1 Score ↑
              </span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                <span style={{ fontSize: '18px', fontWeight: 700, color: '#FFFFFF', letterSpacing: '-0.02em', fontFamily: 'monospace' }}>
                  0.2796
                </span>
                <span style={{ fontSize: '10.5px', color: '#A855F7', fontWeight: 600 }}>
                  SSIM: 0.93
                </span>
              </div>
            </div>

            {/* Metric 5: Inference Latency */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
              <span style={{ fontSize: '10px', color: '#8A8F98', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                Latency (BF16)
              </span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
                <span style={{ fontSize: '18px', fontWeight: 700, color: '#FFFFFF', letterSpacing: '-0.02em', fontFamily: 'monospace' }}>
                  189.5 ms
                </span>
                <span style={{ fontSize: '10.5px', color: '#38BDF8', fontWeight: 600 }}>
                  RTX 5060
                </span>
              </div>
            </div>

            {/* Direct Action Buttons */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Link
                to="/studio"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '7px 14px',
                  borderRadius: '999px',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  color: '#FFFFFF',
                  textDecoration: 'none',
                  fontSize: '12px',
                  fontWeight: 600,
                  border: '1px solid rgba(255, 255, 255, 0.18)',
                  whiteSpace: 'nowrap',
                  transition: 'background 0.15s ease',
                }}
              >
                <Layers size={13} />
                <span>Benchmarks</span>
              </Link>

              <Link
                to="/app"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '7px 16px',
                  borderRadius: '999px',
                  backgroundColor: '#FFFFFF',
                  color: '#08090C',
                  textDecoration: 'none',
                  fontSize: '12px',
                  fontWeight: 600,
                  boxShadow: '0 4px 14px rgba(255, 255, 255, 0.25)',
                  whiteSpace: 'nowrap',
                }}
              >
                <span>3D Viewport</span>
                <ArrowUpRight size={13} />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
