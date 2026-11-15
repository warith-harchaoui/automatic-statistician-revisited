// Tailwind configuration of the autostat studio.
// Build: npx tailwindcss@3 -c src/autostat/static/tailwind.config.js \
//   -i src/autostat/static/studio-input.css -o src/autostat/static/studio.css --minify
// Colours: the deraison.ai palette (sprezzature-colors/references/palette.csv).
// Author: Warith Harchaoui <warithmetics@deraison.ai>
module.exports = {
  content: ["./src/autostat/static/studio.html", "./src/autostat/static/studio.js"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          blue: "#0088FF",
          orange: "#FF8D28",
          turquoise: "#00C3D0",
          purple: "#CB30E0",
          green: "#34C759",
          pink: "#FF2D55",
          brown: "#AC7F5E",
          red: "#FF383C",
          yellow: "#FFCC00",
        },
        surface: { DEFAULT: "#F7F7F8", dark: "#111113" },
        card: { DEFAULT: "#FFFFFF", dark: "#1C1C1F" },
        fill: { DEFAULT: "#EDEDF0", dark: "#2A2A2E" },
        line: { DEFAULT: "#E1E1E6", dark: "#34343A" },
        label: { DEFAULT: "#1A1A1D", dark: "#F2F2F5", 2: "#5E5E66", "2-dark": "#A3A3AD" },
      },
      fontFamily: {
        sans: ["Roboto", "system-ui", "sans-serif"],
        serif: ["Roboto Serif", "serif"],
        mono: ["Roboto Mono", "ui-monospace", "monospace"],
      },
    },
  },
};
