/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: {
          DEFAULT: 'rgb(var(--paper) / <alpha-value>)',
          raised: 'rgb(var(--paper-raised) / <alpha-value>)',
          sunk: 'rgb(var(--paper-sunk) / <alpha-value>)',
        },
        rule: 'rgb(var(--rule) / <alpha-value>)',
        ink: {
          DEFAULT: 'rgb(var(--ink) / <alpha-value>)',
          soft: 'rgb(var(--ink-soft) / <alpha-value>)',
          faint: 'rgb(var(--ink-faint) / <alpha-value>)',
        },
        persimmon: {
          DEFAULT: 'rgb(var(--persimmon) / <alpha-value>)',
          deep: 'rgb(var(--persimmon-deep) / <alpha-value>)',
        },
        moss: 'rgb(var(--moss) / <alpha-value>)',
        ochre: 'rgb(var(--ochre) / <alpha-value>)',
        plum: 'rgb(var(--plum) / <alpha-value>)',
        brick: 'rgb(var(--brick) / <alpha-value>)',
      },
      fontFamily: {
        display: ['Fraunces Variable', 'Songti SC', 'SimSun', 'Noto Serif CJK SC', 'Georgia', 'serif'],
        sans: [
          '"PingFang SC"',
          '"Microsoft YaHei"',
          '"Hiragino Sans GB"',
          '"Noto Sans CJK SC"',
          'ui-sans-serif',
          'system-ui',
          'sans-serif',
        ],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 0 0 rgb(var(--rule)), 0 2px 10px -6px rgb(var(--ink) / 0.25)',
        lift: '0 2px 0 0 rgb(var(--rule)), 0 10px 24px -14px rgb(var(--ink) / 0.4)',
        stamp: 'inset 0 0 0 1.5px currentColor',
      },
      borderRadius: {
        card: '3px',
        pill: '999px',
      },
      keyframes: {
        'rise-in': {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'ink-in': {
          '0%': { opacity: '0', transform: 'translateY(6px) scale(0.98)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        'stamp-down': {
          '0%': { opacity: '0', transform: 'rotate(-8deg) scale(1.35)' },
          '60%': { opacity: '1', transform: 'rotate(-3deg) scale(0.97)' },
          '100%': { opacity: '1', transform: 'rotate(-3deg) scale(1)' },
        },
        'draw-line': {
          '0%': { transform: 'scaleX(0)' },
          '100%': { transform: 'scaleX(1)' },
        },
        breathe: {
          '0%, 100%': { opacity: '0.55' },
          '50%': { opacity: '1' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '0.35', transform: 'scale(0.94)' },
          '50%': { opacity: '0.9', transform: 'scale(1)' },
        },
      },
      animation: {
        'rise-in': 'rise-in 420ms cubic-bezier(0.22, 1, 0.36, 1) both',
        'ink-in': 'ink-in 320ms cubic-bezier(0.22, 1, 0.36, 1) both',
        'stamp-down': 'stamp-down 460ms cubic-bezier(0.34, 1.56, 0.64, 1) both',
        'draw-line': 'draw-line 620ms cubic-bezier(0.22, 1, 0.36, 1) both',
        breathe: 'breathe 2.6s ease-in-out infinite',
        'pulse-soft': 'pulse-soft 1.5s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
