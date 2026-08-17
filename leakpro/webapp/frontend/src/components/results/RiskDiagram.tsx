import React from "react";

/**
 * Schematic of the risk assessment model.
 *
 * The one thing the picture has to convey is the split: the left column is measured by this audit and
 * reproducible, the right column is declared by you and is a value judgement. Everything below is a
 * product of the two, which is why no single number can stand on its own.
 *
 * Structure follows Sion et al., "Privacy Risk Assessment for Data Subject-aware Threat Modeling"
 * (IWPE 2019). Precision follows Jayaraman et al. (PoPETs 2021), Theorem 4.2.
 *
 * Two visual encodings, spelled out in the legend at the foot of the diagram:
 *   role       dashed border = given value, solid border = computed value
 *   provenance amber = measured by the audit, grey = declared by the user
 *
 * Layout rows (viewBox 720x400), kept disjoint so nothing overlaps at any width:
 *   52..158  inputs   196..264  intermediate products   318..380  risk   400..411  legend
 *
 * Expected exposed subjects and attacker precision are deliberately NOT drawn here: neither is
 * derived from Risk. Both come off the measured vulnerability directly (with NDS, or with alpha and
 * pi), so putting them under Risk would assert a dependency the code does not have.
 */
export default function RiskDiagram({ className = "" }: { className?: string }) {
  // Two independent visual encodings, explained by the legend at the foot of the diagram:
  //   role       dashed border = a value that is given, solid border = a value that is computed
  //   provenance amber = measured by the audit, grey = declared by the user
  // Risk is the terminal calculation, so it is solid with a heavier accent stroke.
  const measuredInput = "fill-amber-50 dark:fill-[#2d5867] stroke-primary";
  const declaredInput = "fill-slate-50 dark:fill-[#1d3e4a] stroke-slate-400 dark:stroke-[#365d6c]";
  const computed = "fill-white dark:fill-[#254c5b] stroke-slate-400 dark:stroke-[#365d6c]";
  const computedFinal = "fill-white dark:fill-[#254c5b] stroke-primary";
  const DASH = "5 4";

  const heading = "fill-slate-500 dark:fill-slate-300";
  const body = "fill-slate-700 dark:fill-slate-100";
  const faint = "fill-slate-400 dark:fill-slate-400";
  const line = "stroke-slate-400 dark:stroke-slate-500";
  const tag = "fill-slate-400 dark:fill-slate-400";

  const factors: Array<[string, string, number]> = [
    ["DTS", "data type sensitivity", 88],
    ["NR", "records per subject", 106],
    ["DST", "subject type weight", 124],
    ["NDS", "number of data subjects", 142],
  ];

  return (
    <div className={`w-full overflow-x-auto ${className}`}>
      <svg
        viewBox="0 0 720 430"
        role="img"
        aria-label="Risk equals loss magnitude times loss event frequency. Loss event frequency comes from the attack success rate measured by this audit; loss magnitude comes from four factors you declare."
        className="w-full min-w-[560px] h-auto font-display"
      >
        <defs>
          <marker id="rd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" className="fill-slate-400 dark:fill-slate-500" />
          </marker>
        </defs>

        {/* Column headings */}
        <text x="170" y="20" textAnchor="middle" className={heading} fontSize="12" fontWeight="700" letterSpacing="1.2">
          MEASURED BY THIS AUDIT
        </text>
        <text x="170" y="36" textAnchor="middle" className={faint} fontSize="10.5">
          re-run it and get the same number
        </text>
        <text x="550" y="20" textAnchor="middle" className={heading} fontSize="12" fontWeight="700" letterSpacing="1.2">
          DECLARED BY YOU
        </text>
        <text x="550" y="36" textAnchor="middle" className={faint} fontSize="10.5">
          your use case, which LeakPro cannot observe
        </text>

        <line x1="360" y1="46" x2="360" y2="300" className={line} strokeDasharray="4 5" strokeWidth="1" />

        {/* INPUT, measured: vulnerability */}
        <rect x="55" y="52" width="230" height="106" rx="10" className={measuredInput} strokeWidth="1.5" strokeDasharray={DASH} />
        <text x="170" y="70" textAnchor="middle" className={tag} fontSize="9" fontWeight="700" letterSpacing="1.1">INPUT</text>
        <text x="170" y="99" textAnchor="middle" className={body} fontSize="13" fontWeight="700">Vulnerability (V)</text>
        <text x="170" y="119" textAnchor="middle" className={body} fontSize="12" fontFamily="monospace">TPR at your chosen α</text>
        <text x="170" y="137" textAnchor="middle" className={faint} fontSize="10">how often the best attack is right</text>

        {/* INPUTS, declared: the four harm factors */}
        <rect x="435" y="52" width="230" height="106" rx="10" className={declaredInput} strokeWidth="1.5" strokeDasharray={DASH} />
        <text x="550" y="68" textAnchor="middle" className={tag} fontSize="9" fontWeight="700" letterSpacing="1.1">INPUTS</text>
        {factors.map(([abbr, desc, y]) => (
          <g key={abbr}>
            <text x="455" y={y} className={body} fontSize="11.5" fontWeight="700" fontFamily="monospace">{abbr}</text>
            <text x="497" y={y} className={faint} fontSize="10.5">{desc}</text>
          </g>
        ))}

        {/* Down into the two products */}
        <line x1="170" y1="158" x2="170" y2="194" className={line} strokeWidth="1.5" markerEnd="url(#rd-arrow)" />
        <line x1="550" y1="158" x2="550" y2="194" className={line} strokeWidth="1.5" markerEnd="url(#rd-arrow)" />

        {/* CALCULATED: Loss Event Frequency */}
        <rect x="55" y="196" width="230" height="68" rx="10" className={computed} strokeWidth="1.5" />
        <text x="170" y="212" textAnchor="middle" className={tag} fontSize="9" fontWeight="700" letterSpacing="1.1">CALCULATED</text>
        <text x="170" y="230" textAnchor="middle" className={body} fontSize="13" fontWeight="700">Loss Event Frequency</text>
        <text x="170" y="247" textAnchor="middle" className={body} fontSize="11.5" fontFamily="monospace">LEF = V × (RP × TEF)</text>
        <text x="170" y="259" textAnchor="middle" className={faint} fontSize="9.5">RP × TEF assumed 1: a single attempt</text>

        {/* CALCULATED: Loss Magnitude */}
        <rect x="435" y="196" width="230" height="68" rx="10" className={computed} strokeWidth="1.5" />
        <text x="550" y="212" textAnchor="middle" className={tag} fontSize="9" fontWeight="700" letterSpacing="1.1">CALCULATED</text>
        <text x="550" y="230" textAnchor="middle" className={body} fontSize="13" fontWeight="700">Loss Magnitude</text>
        <text x="550" y="247" textAnchor="middle" className={body} fontSize="11.5" fontFamily="monospace">LM = DTS × NR × DST × NDS</text>
        <text x="550" y="259" textAnchor="middle" className={faint} fontSize="9.5">factors assumed independent</text>

        {/* Join into risk */}
        <path d="M 170 264 L 170 292 L 360 292" fill="none" className={line} strokeWidth="1.5" />
        <path d="M 550 264 L 550 292 L 360 292" fill="none" className={line} strokeWidth="1.5" />
        <line x1="360" y1="292" x2="360" y2="316" className={line} strokeWidth="1.5" markerEnd="url(#rd-arrow)" />

        <rect x="245" y="318" width="230" height="62" rx="10" className={computedFinal} strokeWidth="2.5" />
        <text x="360" y="334" textAnchor="middle" className={tag} fontSize="9" fontWeight="700" letterSpacing="1.1">CALCULATED</text>
        <text x="360" y="353" textAnchor="middle" className={body} fontSize="14" fontWeight="700">Risk</text>
        <text x="360" y="371" textAnchor="middle" className={body} fontSize="12" fontFamily="monospace">Risk = LEF × LM</text>

        {/* Legend: the two encodings, role by border and provenance by colour */}
        <rect x="118" y="400" width="16" height="11" rx="3" className={measuredInput} strokeWidth="1.5" strokeDasharray={DASH} />
        <text x="140" y="409" className={faint} fontSize="10.5">measured input</text>
        <rect x="252" y="400" width="16" height="11" rx="3" className={declaredInput} strokeWidth="1.5" strokeDasharray={DASH} />
        <text x="274" y="409" className={faint} fontSize="10.5">your input</text>
        <rect x="360" y="400" width="16" height="11" rx="3" className={computed} strokeWidth="1.5" />
        <text x="382" y="409" className={faint} fontSize="10.5">calculated</text>
        <rect x="470" y="400" width="16" height="11" rx="3" className={computedFinal} strokeWidth="2.5" />
        <text x="492" y="409" className={faint} fontSize="10.5">final result</text>
      </svg>
    </div>
  );
}
