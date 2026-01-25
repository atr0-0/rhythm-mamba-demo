/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f5f8ff',
          100: '#e9f2ff',
          200: '#d7e6ff',
          300: '#b8d2ff',
          400: '#8cb5ff',
          500: '#2b8cff',
          600: '#1d6cd6',
          700: '#1552a3',
          800: '#143f7a',
          900: '#133665'
        },
        accent: '#5a5dff',
        surface: '#ffffff',
        ink: '#0f1b2d',
        muted: '#5b6b82'
      },
      boxShadow: {
        soft: '0 25px 80px rgba(15, 27, 45, 0.12)',
        inset: 'inset 0 1px 0 rgba(255,255,255,0.5)'
      },
      borderRadius: {
        bubble: '20px'
      },
      backgroundImage: {
        glass: 'linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,255,255,0.60))'
      }
    }
  },
  plugins: [require('@tailwindcss/forms')]
};
