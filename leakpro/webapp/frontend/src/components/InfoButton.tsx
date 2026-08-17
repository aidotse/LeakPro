import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom";

/**
 * A small info affordance: an unobtrusive icon that opens a modal with the long explanation.
 *
 * The point is to keep prose out of the first view. Anything that would make a panel feel like
 * documentation belongs behind one of these, not inline.
 */
export default function InfoButton({
  label,
  title,
  children,
  className = "",
}: {
  /** Short name of the thing being explained. Used for the tooltip and the aria-label. */
  label: string;
  /** Modal heading. Defaults to the label. */
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <button
        type="button"
        aria-label={`About ${label}`}
        title={label}
        onClick={(e) => { e.stopPropagation(); setOpen(true); }}
        className={`material-symbols-outlined align-middle text-slate-400 hover:text-primary transition-colors leading-none ${className}`}
        style={{ fontSize: "1rem" }}
      >
        info
      </button>

      {open && ReactDOM.createPortal(
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="bg-white dark:bg-surface rounded-2xl shadow-2xl w-full max-w-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-surface-border">
              <h3 className="font-bold text-lg">{title ?? label}</h3>
              <button
                onClick={() => setOpen(false)}
                aria-label="Close"
                className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <div className="px-6 py-5 overflow-y-auto max-h-[70vh] text-sm text-slate-600 dark:text-slate-200 flex flex-col gap-3">
              {children}
            </div>
            <div className="flex justify-end px-6 py-4 border-t border-slate-200 dark:border-surface-border">
              <button
                onClick={() => setOpen(false)}
                className="px-5 py-2 rounded-lg bg-slate-700 text-cream border border-primary text-sm font-bold hover:bg-slate-600 transition-colors"
              >
                Got it
              </button>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}
