"""Tabulate every finished experiment in logs/ into one markdown table.

Reads the `[name] ... DONE [lpips=...] ssim=... psnr=... best_loss=...` line the runner prints, plus
the config echoed on the first line, so the table stays in sync with whatever sweeps have been run.
LPIPS is the default objective and is a distance, so the table sorts by it ascending; runs from
before the switch have no lpips field and fall back to SSIM ordering.
"""
import argparse
import ast
import glob
import re

p = argparse.ArgumentParser()
p.add_argument("--glob", default="logs/*.log")
args = p.parse_args()

DONE = re.compile(r"\[(?P<name>[^\]]+)\] DONE (?:lpips=(?P<lpips>[\d.]+) )?ssim=(?P<ssim>[\d.]+) "
                  r"psnr=(?P<psnr>[-\d.]+) best_loss=(?P<loss>[\d.]+) time=(?P<time>\d+)s")
HEAD = re.compile(r"\[(?P<name>[^\]]+)\] (?:model=(?P<model>\S+) )?objects=(?P<objects>\[[^\]]*\]) "
                  r"cfg=(?P<cfg>\{.*\})")
BEST = re.compile(r"Best optimized value: (?P<value>[\d.eE+-]+)")
PARAMS = re.compile(r"Best hyperparameters: (?P<params>\{.*\})")

rows = []
for path in sorted(glob.glob(args.glob)):
    text = open(path, errors="ignore").read()
    done, head = DONE.search(text), HEAD.search(text)
    if done:
        cfg = ast.literal_eval(head.group("cfg")) if head else {}
        rows.append({
            "run": done.group("name"),
            "model": (head.group("model") if head and head.group("model") else "detr50") ,
            "objects": len(ast.literal_eval(head.group("objects"))) if head else None,
            "img": cfg.get("img_size"), "iters": cfg.get("iters"), "attack": cfg.get("attack"),
            "tv": cfg.get("tv"), "lr": cfg.get("lr"),
            "loss": float(done.group("loss")), "ssim": float(done.group("ssim")),
            "lpips": float(done.group("lpips")) if done.group("lpips") else None,
            "time": int(done.group("time")),
        })
        continue
    best = BEST.search(text)
    if best:
        params = PARAMS.search(text)
        # The study always maximizes, so an LPIPS study reports -LPIPS. Nothing else in these logs is
        # negative, which is what tells the two objectives apart after the fact.
        value = float(best.group("value"))
        rows.append({"run": path.split("/")[-1].removesuffix(".log"), "model": "optuna search",
                     "objects": None, "img": None, "iters": None, "attack": None, "tv": None,
                     "lr": None, "loss": None, "time": None,
                     "ssim": value if value >= 0 else None,
                     "lpips": -value if value < 0 else None,
                     "params": params.group("params") if params else ""})

if not rows:
    print(f"No finished runs matched {args.glob}.")
    raise SystemExit

cols = ["run", "model", "objects", "img", "iters", "attack", "tv", "lr", "loss", "lpips", "ssim"]


def rank(row: dict) -> tuple:
    """Best first: LPIPS ascending, and SSIM descending for runs that predate the LPIPS objective."""
    return (row["lpips"] if row["lpips"] is not None else float("inf"),
            -(row["ssim"] if row["ssim"] is not None else 0.0))


print("| " + " | ".join(cols) + " |")
print("|" + "|".join("---" for _ in cols) + "|")
for row in sorted(rows, key=rank):
    cells = ["" if row.get(c) is None else
             (f"{row[c]:.4f}" if c in {"loss", "ssim", "lpips"} else str(row.get(c))) for c in cols]
    print("| " + " | ".join(cells) + " |")

for row in rows:
    if row.get("params"):
        print(f"\n{row['run']} best hyperparameters: {row['params']}")
