import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';

export const CustomCursor: React.FC = () => {
  const cursorRef = useRef<HTMLDivElement | null>(null);
  const textRef = useRef<HTMLSpanElement | null>(null);
  const [cursorText, setCursorText] = useState<string>('');
  const [isHovered, setIsHovered] = useState<boolean>(false);
  const [isPointer, setIsPointer] = useState<boolean>(false);

  useEffect(() => {
    const cursor = cursorRef.current;
    if (!cursor) return;

    if (window.matchMedia('(pointer: coarse)').matches) return; // Touch devices

    // Respect reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const xTo = gsap.quickTo(cursor, 'x', { duration: 0.25, ease: 'power3.out' });
    const yTo = gsap.quickTo(cursor, 'y', { duration: 0.25, ease: 'power3.out' });

    const handleMouseMove = (e: MouseEvent) => {
      xTo(e.clientX);
      yTo(e.clientY);
    };

    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const interactive = target.closest('[data-cursor-text], [data-cursor-magnetic], button, a, input, select');

      if (interactive) {
        const text = interactive.getAttribute('data-cursor-text');
        if (text) {
          setCursorText(text);
          setIsHovered(true);
          setIsPointer(false);
        } else {
          setCursorText('');
          setIsHovered(false);
          setIsPointer(true);
        }
      } else {
        setCursorText('');
        setIsHovered(false);
        setIsPointer(false);
      }
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    document.addEventListener('mouseover', handleMouseOver, { passive: true });

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseover', handleMouseOver);
    };
  }, []);

  return (
    <div
      ref={cursorRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        pointerEvents: 'none',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transform: 'translate(-50%, -50%)',
        borderRadius: '50%',
        backgroundColor: isHovered
          ? 'var(--accent, #455ce9)'
          : isPointer
          ? 'rgba(255, 255, 255, 0.4)'
          : 'var(--cursor-color, #ffffff)',
        width: isHovered ? '80px' : isPointer ? '44px' : '10px',
        height: isHovered ? '80px' : isPointer ? '44px' : '10px',
        transition: 'width 0.28s cubic-bezier(0.25, 1, 0.5, 1), height 0.28s cubic-bezier(0.25, 1, 0.5, 1), background-color 0.2s ease',
        mixBlendMode: isHovered ? 'normal' : 'difference',
      }}
    >
      <span
        ref={textRef}
        style={{
          color: '#ffffff',
          fontSize: '11px',
          fontWeight: 600,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          opacity: isHovered ? 1 : 0,
          transform: isHovered ? 'scale(1)' : 'scale(0.5)',
          transition: 'opacity 0.2s ease, transform 0.2s ease',
          pointerEvents: 'none',
          userSelect: 'none',
        }}
      >
        {cursorText}
      </span>
    </div>
  );
};

