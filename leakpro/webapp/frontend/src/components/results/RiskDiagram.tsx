import React from "react";

/**
 * Schematic of the risk assessment model, in three boxes.
 *
 * The point the picture has to make is the split: the left branch is measured by this audit and
 * reproducible, the right branch is declared by the user and is a value judgement, and Risk is their
 * product.
 *
 * Factor names are spelled out in full. The abbreviations (LEF, LM, V, DTS, NR, DST, NDS) that the JSON
 * output, the summary table and the papers use are defined in the info modals instead, to keep the
 * picture to one line of text per idea.
 *
 * Structure follows Sion et al., "Privacy Risk Assessment for Data Subject-aware Threat Modeling"
 * (IWPE 2019).
 *
 * Layout rows (viewBox 720x260), kept disjoint so nothing overlaps at any width:
 *   52..114  the two products    170..228  risk
 */
export default function RiskDiagram({ className = "" }: { className?: string }) {
  const measured = "fill-amber-50 dark:fill-[#2d5867] stroke-primary";
  const declared = "fill-slate-50 dark:fill-[#1d3e4a] stroke-slate-400 dark:stroke-[#365d6c]";
  const result = "fill-white dark:fill-[#254c5b] stroke-primary";

  const headingCls = "fill-slate-500 dark:fill-slate-300";
  const body = "fill-slate-700 dark:fill-slate-100";
  const faint = "fill-slate-400 dark:fill-slate-400";
  const line = "stroke-slate-400 dark:stroke-slate-500";

  return (
    <div className={`w-full overflow-x-auto ${className}`}>
      <svg
        viewBox="0 0 720 246"
        role="img"
        aria-label="Risk is loss event frequency times loss magnitude. Loss event frequency is vulnerability times retention period times threat event frequency, measured by this audit. Loss magnitude is data type sensitivity times records per subject times data subject type times number of data subjects, declared by you."
        className="w-full min-w-[560px] h-auto font-display"
      >
        <defs>
          <marker id="rd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" className="fill-slate-400 dark:fill-slate-500" />
          </marker>
        </defs>

        {/* Column headings carry the measured / declared distinction */}
        <text x="180" y="20" textAnchor="middle" className={headingCls} fontSize="12" fontWeight="700" letterSpacing="1.2">
          MEASURED BY THIS AUDIT
        </text>
        <text x="180" y="36" textAnchor="middle" className={faint} fontSize="10.5">
          re-run it and get the same number
        </text>
        <text x="540" y="20" textAnchor="middle" className={headingCls} fontSize="12" fontWeight="700" letterSpacing="1.2">
          DECLARED BY YOU
        </text>
        <text x="540" y="36" textAnchor="middle" className={faint} fontSize="10.5">
          your use case, which LeakPro cannot observe
        </text>

        {/* Loss Event Frequency */}
        <rect x="30" y="52" width="300" height="62" rx="10" className={measured} strokeWidth="1.5" />
        <text x="180" y="74" textAnchor="middle" className={body} fontSize="13" fontWeight="700">Loss Event Frequency</text>
        <text x="180" y="93" textAnchor="middle" className={body} fontSize="10.5">Vulnerability × Retention Period</text>
        <text x="180" y="107" textAnchor="middle" className={body} fontSize="10.5">× Threat Event Frequency</text>

        {/* Loss Magnitude */}
        <rect x="390" y="52" width="300" height="62" rx="10" className={declared} strokeWidth="1.5" />
        <text x="540" y="74" textAnchor="middle" className={body} fontSize="13" fontWeight="700">Loss Magnitude</text>
        <text x="540" y="93" textAnchor="middle" className={body} fontSize="10.5">Data Type Sensitivity × Records per Subject</text>
        <text x="540" y="107" textAnchor="middle" className={body} fontSize="10.5">× Data Subject Type × Number of Data Subjects</text>

        {/* Join into risk */}
        <path d="M 180 114 L 180 142 L 360 142" fill="none" className={line} strokeWidth="1.5" />
        <path d="M 540 114 L 540 142 L 360 142" fill="none" className={line} strokeWidth="1.5" />
        <line x1="360" y1="142" x2="360" y2="168" className={line} strokeWidth="1.5" markerEnd="url(#rd-arrow)" />

        <rect x="200" y="170" width="320" height="58" rx="10" className={result} strokeWidth="2.5" />
        <text x="360" y="192" textAnchor="middle" className={body} fontSize="14" fontWeight="700">Risk</text>
        <text x="360" y="212" textAnchor="middle" className={body} fontSize="11.5">Loss Event Frequency × Loss Magnitude</text>
      </svg>
    </div>
  );
}
