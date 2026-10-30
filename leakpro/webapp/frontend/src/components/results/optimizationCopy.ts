/**
 * Plain-language labels for the optimization view.
 *
 * The audience includes non-technical public-sector users, so the search
 * method is never named: no "Bayesian", no "Pareto", no "acquisition". Keep
 * every visible string in this file so the vocabulary stays reviewable in one
 * place rather than scattered through JSX.
 */

/** "1%", "0.1%", "1.01%", "2.5%": as many decimals as the rate has (to three), never rounded to a different rate. */
export const fprLabel = (fpr: number) => {
  const pct = fpr * 100;
  return `${Number.isInteger(pct) ? pct : parseFloat(pct.toFixed(3))}%`;
};

const count = (n: number) => n.toLocaleString();

export const COPY = {
  entry: "Optimize protection",
  sectionBlurb: "Test protection settings per model with the buttons in the table.",
  title: "Optimize protection",
  intro:
    "We train your model many times with different protection settings and test each one, " +
    "then show you the best trade-offs between privacy and quality.",

  // The search runs on a subset of the data; every number in the chart is at that scale.
  scaleNote: (members: number) =>
    `To keep the search fast, every setting is tested on a model trained on ${count(members)} records. ` +
    "The setting you confirm is then retrained on your model's own data and measured there.",
  constraintLabel: "Maximum acceptable quality loss",
  constraintHelp:
    "Relative to the unprotected model: 5% means a model that keeps at least 95% of its quality. " +
    "Settings below the line are shown but not recommended.",
  noBaselineYet:
    "The unprotected reference model is still training; the quality limit applies once its quality is known.",
  noBaselineAtAll:
    "This run has no unprotected reference model (it was searched before one was measured), so the quality " +
    "limit cannot be applied to it. Start a new search to get one.",
  baselineLabel: "Quality without protection",
  baselineHelp: (members: number) => `an unprotected model trained on the same ${count(members)} records`,
  modelUtilityLabel: "Your model's quality",
  floorLabel: "Quality floor",

  advanced: "Advanced",
  advancedHelp: "Leave blank to search a setting; type a value to pin it.",

  start: "Start",
  running: "Testing protection settings…",
  testedOf: (done: number, total?: number) =>
    total ? `Tested ${done} of ${total} settings` : `Tested ${done} settings`,

  axisAttack: "Attack success (%) — lower is safer",
  axisQuality: "Model quality (%) — higher is better",
  fprNote: (fpr: number | undefined, realized?: [number, number] | null) =>
    "Attack success is the share of training records an attacker correctly identifies, " +
    (fpr == null ? "measured at the run's false-alarm rate." : `measured at a ${fprLabel(fpr)} false-alarm rate.`) +
    (realized == null ? "" :
      ` Where the data could not express that rate exactly, the setting shows the rate it was actually read at` +
      ` (${fprLabel(realized[0])}${realized[0] === realized[1] ? "" : ` to ${fprLabel(realized[1])}`}).`),
  intervalFloorNote:
    "Likely ranges cover only which training records were audited. The threshold itself rests on a few " +
    "false alarms, so the true uncertainty is at least this wide.",
  chanceNote: (n: number, fpr: number | undefined) =>
    `${n} setting${n === 1 ? "" : "s"} cannot be told apart from guessing: the attack's likely range reaches ` +
    `down to the false-alarm rate${fpr == null ? "" : ` (${fprLabel(fpr)})`} the attack was read at. ` +
    "An attack at or below its false-alarm rate is noise around chance, not protection. " +
    "They are shown hollow and not ranked; the audit is too small to separate them.",
  unknownNote: (n: number) =>
    `${n} setting${n === 1 ? "" : "s"} ${n === 1 ? "has" : "have"} no likely range (no interval was recorded), ` +
    "so it cannot be told whether the attack beat guessing. Shown grey and not ranked.",
  pollRetrying: "Lost contact with the backend while the run is in progress; retrying…",
  pollGaveUp: "Stopped checking the run after repeated errors. Reload the page to resume; the run itself continues.",
  stale: "These results are from a previous model under this name.",
  searchAgain: "Search again",
  enterToken: "Enter the token and retry",
  baselinePoint: "Unprotected",

  resultsTitle: "Best trade-offs",
  resultsBlurb:
    "Each highlighted setting is the best available at its level of protection. " +
    "Everything tested is shown in grey behind them. Click any point on the line, or pick from the list.",
  exceedsLimit: "Exceeds your quality limit",
  qualityLimitLine: "Your quality limit",
  unmeasured: (n: number) =>
    `${n} tested setting${n === 1 ? "" : "s"} could not be measured and ${n === 1 ? "is" : "are"} not shown.`,
  frontierList: "Settings on the line",
  frontierNone: "No setting stays within your quality limit. Move the slider to see what the search found.",
  chanceTag: "≈ guessing",
  unknownTag: "range unknown",

  panelTitle: "Protection setting",
  estimatedAttack: "Estimated attack success",
  estimatedQuality: "Estimated quality",
  estimatedScale: (members: number) => `On a model trained on ${count(members)} records.`,
  confirm: "Confirm this setting",
  confirmHelp:
    "Retrains your model with this setting on its own data, then attacks it with more reference models. " +
    "The numbers you adopt come from that model, not from the estimate.",
  retryConfirm: "Try again",

  verifying: "Measuring on your model…",
  verifyingHelp:
    "Training your model with this setting at full size, then attacking it. " +
    "This takes longer than one search step.",
  estimate: "Estimate",
  verified: "Measured",
  verifiedOnYourModel: "Measured on your model",
  verifiedOnSubset: "Measured on the search model",
  subsetScopeHelp:
    "Your model's own training split is unknown, so this re-attacks the search-scale model with more " +
    "reference models. It describes that model, not yours, and quality is not compared.",
  qualityLoss: "Quality loss",
  qualityGain: "Quality gain",
  qualityLossHelp: (yours: string) => `vs your model's ${yours}`,
  qualityGainHelp:
    "The protected model scored higher than your model on its own test split: the noise acted as " +
    "regularisation, or your model was itself trained privately.",
  exceedsMeasuredLimit: (loss: string, limit: string) =>
    `Measured on your model, the quality loss is ${loss}, above the ${limit} limit you set. ` +
    "The estimate was made on the search model; this is the number that counts.",
  estimateQualitySubset: "quality (search model)",
  measuredFprLine: (fpr: string) => `False-alarm rate ${fpr}.`,
  range: "likely range",
  optimistic: "The estimate was optimistic: the measured attack is clearly above it.",
  epsilonLabel: "Privacy budget (ε)",
  epsilonNone: "∞ (no DP guarantee)",
  epsilonNote: "ε is for one training run; choosing this setting from many tested ones is not accounted.",
  remeasure: "Measure again (retrains from scratch)",
  adopt: "Adopt this configuration",
  adoptHelp: "Adds the measured model to your results as a normal row, with its own attack result.",
  adopted: "Added to your results as an optimized setting.",

  lowResolution: "Read these numbers with care.",
  notMeasurable: "This result could not be measured directly.",
  notMeasurableHelp:
    "Too few records separated the protected model's answers, so no attack figure could be read " +
    "at this false-alarm rate. Treat it as unknown, not as proof of protection.",
  noResults: "No settings were tested successfully.",
  allUnmeasured: (n: number) =>
    `All ${n} tested setting${n === 1 ? "" : "s"} could not be measured, so there is nothing to rank.`,
  allUnmeasuredReasons: "Why:",
  allUnmeasuredHelp:
    "A setting whose model does not learn is never attacked: its low attack number would look like protection. " +
    "This search is finished; pinning a lower noise multiplier or a higher learning rate under Advanced needs a " +
    "new search, on a copy of the model, so the two searches never share one frontier.",
  failed: "The optimization run failed.",
  retry: "Try again",
  retryHint: "You can start it again — finished settings are kept.",
  back: "Back to results",
  authLost: "Your session's API token was rejected. Reload the page and enter the token again.",
} as const;

/** Tags for the highlighted settings, ordered from most protective to highest quality. */
// "Safest measurable", not "Safest": settings at chance level are excluded
// before the tags are handed out, so the top tag names the most protective
// setting whose attack could still be measured.
const TAGS: Record<number, readonly string[]> = {
  1: ["Balanced"],
  2: ["Safest measurable", "Best quality"],
  3: ["Safest measurable", "Balanced", "Best quality"],
  4: ["Safest measurable", "Safer", "Better quality", "Best quality"],
  5: ["Safest measurable", "Safer", "Balanced", "Better quality", "Best quality"],
};

export function tagsFor(count: number): readonly string[] {
  return TAGS[count] ?? TAGS[5];
}
