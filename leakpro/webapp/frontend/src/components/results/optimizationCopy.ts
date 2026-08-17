/**
 * Plain-language labels for the optimization view.
 *
 * The audience includes non-technical public-sector users, so the search
 * method is never named: no "Bayesian", no "Pareto", no "acquisition". Keep
 * every visible string in this file so the vocabulary stays reviewable in one
 * place rather than scattered through JSX.
 */

export const COPY = {
  entry: "Find the best protection",
  title: "Find the best protection",
  intro:
    "We train your model many times with different protection settings and test each one, " +
    "then show you the best trade-offs between privacy and quality.",

  constraintLabel: "Maximum acceptable quality loss",
  constraintHelp: "How much accuracy you are willing to give up in exchange for protection.",
  baselineLabel: "Current quality, without protection",
  floorLabel: "Quality floor",

  advanced: "Advanced",
  advancedHelp: "Defaults come from the optimization run. Change these only if you know why.",

  start: "Start",
  running: "Testing protection settings…",
  testedOf: (done: number, total?: number) =>
    total ? `Tested ${done} of ${total} settings` : `Tested ${done} settings`,

  axisAttack: "Attack success (%) — lower is safer",
  axisQuality: "Model quality (%) — higher is better",
  fprNote:
    "Attack success is the share of training records an attacker correctly identifies, " +
    "measured at a 1% false-alarm rate.",

  resultsTitle: "Best trade-offs",
  resultsBlurb:
    "Each highlighted setting is the best available at its level of protection. " +
    "Everything tested is shown in grey behind them.",
  exceedsLimit: "Exceeds your quality limit",
  qualityLimitLine: "Your quality limit",

  panelTitle: "Protection setting",
  estimatedAttack: "Estimated attack success",
  estimatedQuality: "Estimated quality",
  confirm: "Confirm this setting",

  verifying: "Verifying with full audit…",
  verifyingHelp:
    "The search uses a faster, approximate test. This runs the full audit on the setting you picked " +
    "so the numbers can be trusted.",
  estimate: "Estimate",
  verified: "Verified",
  optimistic: "Estimate was optimistic — verified numbers shown",
  adopt: "Adopt this configuration",
  adopted: "Added to your results as an optimized setting.",

  noResults: "No settings were tested successfully.",
  failed: "The optimization run failed.",
  back: "Back to results",
} as const;

/** Tags for the highlighted settings, ordered from most protective to highest quality. */
const TAGS: Record<number, readonly string[]> = {
  1: ["Balanced"],
  2: ["Safest", "Best quality"],
  3: ["Safest", "Balanced", "Best quality"],
  4: ["Safest", "Safer", "Better quality", "Best quality"],
  5: ["Safest", "Safer", "Balanced", "Better quality", "Best quality"],
};

export function tagsFor(count: number): readonly string[] {
  return TAGS[count] ?? TAGS[5];
}
