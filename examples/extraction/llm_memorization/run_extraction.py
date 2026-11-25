"""Command-line version of extraction_main.ipynb: run a memorization extraction audit.

Runs an audit config, by default audit.yaml (Llama-3.1-8B, Elasticsearch index, full
budgets, Phases 1-2); the options below override single settings without editing the file:

    python run_extraction.py                                  # audit.yaml as it is, ~/es_env.sh
    python run_extraction.py --prefixes PATH --es-env PATH    # inputs elsewhere
    python run_extraction.py --finetune                       # also the optional Phases 3-4
    python run_extraction.py --n 200                          # small trial run
    python run_extraction.py --check                          # check the inputs only, then stop
    python run_extraction.py --phase 1                        # Phase 1 only; then --phase 2
    python run_extraction.py --finetune --phase 3             # optional Phases 3, then 4

--phase N runs the phases up to N and saves the result so far. Earlier phases are read back from
the cache in seconds, so running --phase 1, 2, 3, 4 one after the other costs no more than one run.

Elasticsearch settings (ES_HOST, ES_USER, ES_PASSWORD, ES_CA_CERT) are taken from the environment,
or from the `export NAME=value` lines of --es-env for any that are not set; they are never printed.
Re-running the same command resumes from <output-dir>/artifacts. The log is also written to
<output-dir>/leakpro.log.
"""

import argparse
import os
import re
from pathlib import Path

import yaml
from extraction_model_handler import ExtractionModelHandler

from leakpro import LeakPro
from leakpro.attacks.extraction_attacks.memorization_extraction import MemorizationExtractionConfig
from leakpro.attacks.utils.extraction.elasticsearch_verifier import connect
from leakpro.utils.seed import seed_everything

ES_KEYS = ("ES_HOST", "ES_USER", "ES_PASSWORD", "ES_CA_CERT")


def load_es_env(path: Path) -> None:
    """Set ES_* variables from `export NAME=value` lines, without overriding the environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        m = re.match(r"""\s*export\s+(\w+)=["']?(.*?)["']?\s*$""", line)
        if m and m.group(1) in ES_KEYS:
            os.environ.setdefault(m.group(1), os.path.expanduser(m.group(2)))


def build_config(args: argparse.Namespace) -> dict:
    """The config of --audit (audit.yaml), with the command-line options applied."""
    config = yaml.safe_load(args.audit.read_text())
    audit, target = config["audit"], config["target"]
    attack = audit["attack_list"][0]
    here = args.audit.resolve().parent
    if args.n is not None:
        for strategy in attack["initial"]:
            strategy["n"] = args.n
    if args.seed is not None:
        audit["random_seed"] = args.seed
    if args.index is not None:
        attack["attacker_verifier"]["index"] = args.index
    if args.finetune:
        attack.setdefault("finetune", {})
    if args.no_finetune:
        attack.pop("finetune", None)
    audit["output_dir"] = str(args.output_dir or here / audit["output_dir"])
    for strategy in attack["initial"] + (attack.get("finetune") or {}).get("strategies", []):
        if strategy.get("type") == "conditional":
            strategy["prefix_path"] = str(args.prefixes or here / strategy["prefix_path"])
    target["module_path"] = str(here / target["module_path"])
    target["target_folder"] = str(here / target["target_folder"])
    return config


def check_inputs(config: dict, args: argparse.Namespace) -> list:
    """Fail fast, before the model is loaded, on a missing input; returns the configured phases."""
    attack = config["audit"]["attack_list"][0]
    verifier = attack.get("attacker_verifier") or {}
    pools = {Path(s["prefix_path"]) for s in attack["initial"] if s.get("type") == "conditional"}
    for prefixes in sorted(pools):
        if not prefixes.is_file():
            raise SystemExit(f"Prefix pool {prefixes} not found: build it with prepare_data.py (README.md) "
                             "or pass --prefixes")
        with prefixes.open() as f:
            print(f"prefix pool: {sum(1 for _ in f)} prefixes ({prefixes})")
    if verifier.get("type") == "elasticsearch":
        missing = [k for k in ("ES_HOST", "ES_USER", "ES_PASSWORD") if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"Elasticsearch settings {missing} not set: export them or pass --es-env (README.md)")
        index = verifier["index"]
        print(f"Elasticsearch index {index}: {connect().count(index=index)['count']} chunks")
    # The attack's own config decides which phases run (validated here, before the model is loaded).
    configured = MemorizationExtractionConfig(**{k: v for k, v in attack.items() if k != "attack"}).phases()
    if args.phase is not None and args.phase not in configured:
        hint = "add --finetune for the optional Phases 3-4" if args.phase > 2 else f"see `expansion` in {args.audit}"
        raise SystemExit(f"Phase {args.phase} is not configured (phases: {configured}); {hint}")
    return configured


def prepare_attack(audit_yaml: Path) -> object:
    """Load the model and return the prepared attack of ``audit_yaml`` (as LeakPro's scheduler does)."""
    leakpro = LeakPro(ExtractionModelHandler, str(audit_yaml))
    scheduler = leakpro.attack_scheduler
    if not scheduler.attacks:
        raise SystemExit(f"The attack could not be created; see the log in {Path(audit_yaml).parent}/leakpro.log")
    attack = scheduler.attacks[0]
    seed_everything(scheduler.random_seed)
    attack.prepare_attack()
    return attack


def run_phases(audit_yaml: Path, numbers: list) -> object:
    """Load the model, run the given phases in order, and save and return the result so far."""
    attack = prepare_attack(audit_yaml)
    phases = {1: attack.run_phase1, 2: attack.run_phase2, 3: attack.run_phase3, 4: attack.run_phase4}
    for number in numbers:
        phases[number]()
    result = attack.finish()
    result.save(attack_obj=attack, output_dir=attack.output_dir)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path, default=Path(__file__).with_name("audit.yaml"),
                    help="audit config (default audit.yaml); the options below override it")
    ap.add_argument("--prefixes", type=Path, help="prefix pool for the conditional strategy (its prefix_path)")
    ap.add_argument("--es-env", type=Path, default=Path.home() / "es_env.sh",
                    help="optional shell file with `export ES_...=` lines")
    ap.add_argument("--index", help="Elasticsearch index of the reference corpus")
    ap.add_argument("--output-dir", type=Path, help="cache and results (audit.output_dir)")
    ap.add_argument("--n", type=int, help="generations per Phase 1/4 strategy")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--finetune", action="store_true", help="include the optional Phases 3-4 (extensive attack)")
    ap.add_argument("--no-finetune", action="store_true", help="leave out Phases 3-4 even if the config has them")
    ap.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                    help="run the phases up to this one and save the result so far (default: all)")
    ap.add_argument("--check", action="store_true", help="check the prefix pool and Elasticsearch, then stop")
    args = ap.parse_args()

    load_es_env(args.es_env.expanduser())
    config = build_config(args)
    configured = check_inputs(config, args)
    if args.check:
        print(f"Inputs OK. Configured phases: {configured}")
        return
    output_dir = Path(config["audit"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_yaml = output_dir / "audit.yaml"   # the config as run, options applied
    audit_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"config: {args.audit} -> {audit_yaml}")
    last = args.phase or configured[-1]
    result = run_phases(audit_yaml, [n for n in configured if n <= last])
    # finish() has already logged the per-strategy table; only the totals follow here.
    def total(phase: str, key: str) -> object:
        value = result.metrics.get(phase, {}).get("total", {}).get(key)
        return "not run yet" if value is None else value

    def counts(phase: str, key: str, unique_key: str) -> str:
        value = total(phase, key)
        return value if isinstance(value, str) else f"{value} sequences, {total(phase, unique_key)} unique passages"

    print(f"S0 (Phase 1 verified):    {counts('initial', 'verified', 'unique')}")
    if 2 in configured:
        print(f"S1-novel (Phase 2):       {counts('expansion', 'novel', 'novel_unique')}")
    if 4 in configured:
        print(f"S_new (Phase 4 verified): {counts('finetuned', 'verified', 'unique')}")
        if "finetuned" in result.metrics:
            print(f"  already in Phase 1 pool: {total('finetuned', 'in_phase1_window')} (a verified window), "
                  f"{total('finetuned', 'in_phase1_text')} (same text)")
        if "finetuned_control" in result.metrics:
            print(f"Adapter-off control:      {counts('finetuned_control', 'verified', 'unique')}")
    if last < configured[-1]:
        done = f"Phase {last}" if last == 1 else f"Phases 1-{last}"
        print(f"{done} done. Next: python run_extraction.py --phase {configured[configured.index(last) + 1]}")
    print(f"results: {output_dir}/results/{result.id}/")


if __name__ == "__main__":
    main()
