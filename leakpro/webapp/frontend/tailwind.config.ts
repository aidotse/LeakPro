import type { Config } from "tailwindcss";
import forms from "@tailwindcss/forms";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#f5a623",
        cream: "#FFFEFA",
        "background-light": "#f6f6f8",
        "background-dark": "#1d3e4a",
        // Teal-family surfaces so cards/panels harmonize with the dark teal page
        // instead of the old blue-black slate. surface = raised card (lighter than
        // page), surface-2 = chips/hover, surface-deep = darkest wells (code blocks).
        surface: "#203b45",
        "surface-2": "#2d5867",
        "surface-deep": "#152f3a",
        "surface-border": "#365d6c",
      },
      fontFamily: {
        display: ["Inter", "sans-serif"],
      },
      borderRadius: {
        DEFAULT: "0.25rem",
        lg: "0.5rem",
        xl: "0.75rem",
        full: "9999px",
      },
    },
  },
  plugins: [forms],
} satisfies Config;
