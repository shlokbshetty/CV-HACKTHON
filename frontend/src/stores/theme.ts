import { create } from 'zustand';

export type Theme = 'day' | 'night';

interface ThemeState {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const STORAGE_KEY = 'depthlab-theme';

function getInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'day';
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === 'day' || saved === 'night') return saved;
  // If user prefers dark color scheme, default to night
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'night';
  }
  return 'day';
}

function applyTheme(theme: Theme) {
  if (typeof document !== 'undefined') {
    document.documentElement.setAttribute('data-theme', theme);
  }
}

export const useThemeStore = create<ThemeState>((set) => {
  const initial = getInitialTheme();
  applyTheme(initial);

  return {
    theme: initial,
    setTheme: (theme) => {
      localStorage.setItem(STORAGE_KEY, theme);
      applyTheme(theme);
      set({ theme });
    },
    toggleTheme: () => {
      set((state) => {
        const next = state.theme === 'day' ? 'night' : 'day';
        localStorage.setItem(STORAGE_KEY, next);
        applyTheme(next);
        return { theme: next };
      });
    },
  };
});

