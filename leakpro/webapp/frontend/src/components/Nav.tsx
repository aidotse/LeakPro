import React, { useState } from "react";

interface NavProps {
  onToggleDark: () => void;
  dark: boolean;
}

export default function Nav({ onToggleDark, dark }: NavProps) {
  return (
    <header className="border-b border-slate-200 dark:border-slate-800 bg-background-light dark:bg-background-dark px-6 py-4 shrink-0">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img src="/logo.jpg" alt="LeakPro" className="h-10 w-10 rounded-lg object-contain" />
          <h1 className="text-lg font-bold tracking-tight">LeakPro</h1>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={onToggleDark}
            className="p-2 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors"
            title="Toggle dark mode"
          >
            <span className="material-symbols-outlined text-slate-600 dark:text-slate-400">
              {dark ? "light_mode" : "dark_mode"}
            </span>
          </button>
          <button className="p-2 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors">
            <span className="material-symbols-outlined text-slate-600 dark:text-slate-400">help_outline</span>
          </button>
          <button className="p-2 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors">
            <span className="material-symbols-outlined text-slate-600 dark:text-slate-400">settings</span>
          </button>
          <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold text-xs border border-primary/30">
            LP
          </div>
        </div>
      </div>
    </header>
  );
}
