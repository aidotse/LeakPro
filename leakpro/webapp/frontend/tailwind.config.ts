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
