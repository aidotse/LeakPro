import React from "react";

export const STEPS = [
  { label: "Dataset",  icon: "database" },
  { label: "Architecture", icon: "architecture" },
  { label: "Models",   icon: "psychology" },
  { label: "Attacks",  icon: "tune" },
  { label: "Run",      icon: "rocket_launch" },
  { label: "Results",  icon: "bar_chart" },
];

interface StepperProps {
  current: number; // 0-indexed
}

export default function Stepper({ current }: StepperProps) {
  return (
    <nav className="flex justify-between items-start w-full relative">
      {/* Progress line background */}
      <div className="absolute top-5 left-0 w-full h-0.5 bg-slate-200 dark:bg-slate-800 -z-10" />

      {STEPS.map((step, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <div
            key={step.label}
            className={`flex flex-col items-center gap-3 group ${!active && !done ? "opacity-50" : ""}`}
          >
            <div
              className={`size-10 rounded-full flex items-center justify-center ring-4 ring-background-light dark:ring-background-dark transition-all
                ${active ? "bg-primary text-white" : done ? "bg-primary/20 text-primary" : "bg-slate-200 dark:bg-slate-800 text-slate-500"}`}
            >
              {done ? (
                <span className="material-symbols-outlined text-xl">check</span>
              ) : (
                <span className="material-symbols-outlined text-xl">{step.icon}</span>
              )}
            </div>
            <div className="text-center">
              <p className={`text-xs font-bold uppercase tracking-wider ${active ? "text-primary" : "text-slate-500"}`}>
                Step {i + 1}
              </p>
              <p className="text-sm font-semibold">{step.label}</p>
            </div>
          </div>
        );
      })}
    </nav>
  );
}
