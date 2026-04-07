import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-inter)', 'Inter', 'sans-serif'],
        serif: ['var(--font-merriweather)', 'Merriweather', 'serif'],
      },
        colors: {
          graph: {
            bg:           '#0d0d0d',
            central:      '#2db7a3',
            maharashtra:  '#4e7cff',
            uttar_pradesh:'#f4b63f',
            karnataka:    '#78b94d',
            tamil_nadu:   '#ff7a59',
          },
        },
    },
  },
  plugins: [],
};

export default config;
