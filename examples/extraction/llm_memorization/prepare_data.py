"""Download the attacker's reference corpus and build the prefix pool.

Builds the data for the Llama-3.1-8B setup, without the deprecated ``datasets`` loading script:

1. **Reference corpus** (``download``): streams RedPajama-Data-1T's Wikipedia, StackExchange, C4 and
   Common Crawl files straight from the URLs the dataset lists, taking one document from each source
   in turn (round robin) until ``--target-gb`` is written. Output: one ``{"doc_id", "text", "source"}``
   per line.
2. **Prefix pool** (``prefixes``): reservoir-samples ``--num-prefixes`` documents and keeps a random
   5-10 token prefix of each document with at least 60 tokens.
3. **Elasticsearch index** (``index``): the attacker verifier for the 20 GB corpus, which is too large
   for the in-memory verifier. Uses ``ES_HOST`` / ``ES_USER`` / ``ES_PASSWORD`` / ``ES_CA_CERT``.

The defaults are the settings of ``audit.yaml`` (Llama-3.1-8B):

    python prepare_data.py download     # 20 GB corpus -> data/aux_corpus.jsonl (hours)
    python prepare_data.py prefixes     # Llama tokenizer -> data/prefixes_conditional.jsonl
    python prepare_data.py index        # -> Elasticsearch index redpajama_chunks (hours)

"""

import argparse
import io
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Iterator

URL_LISTS = "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/raw/main/urls/{subset}.txt"
SUBSETS = ["wikipedia", "stackexchange", "c4", "common_crawl"]


def _open(url: str):  # noqa: ANN202
    # The data host rejects urllib's default User-Agent.
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "leakpro-example"}))  # noqa: S310


def _stream_subset(subset: str) -> Iterator[str]:
    """Texts of one RedPajama subset, file by file in the order the dataset lists them."""
    with _open(URL_LISTS.format(subset=subset)) as f:
        urls = [line.decode().strip() for line in f if line.strip()]
    for url in urls:
        with _open(url) as raw:
            if url.endswith(".zst"):
                import zstandard

                lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
            else:
                lines = io.TextIOWrapper(raw, encoding="utf-8")
            for line in lines:
                text = json.loads(line).get("text", "")
                if text:
                    yield text


def download(output: Path, target_gb: float) -> None:
    """Round-robin sample of the four sources until ``target_gb`` gigabytes are written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"{output} exists; delete it to download again (a partial file cannot be resumed)")
    target = target_gb * 1024**3
    streams = {s: _stream_subset(s) for s in SUBSETS}
    written, doc = 0, 0
    with output.open("w", encoding="utf-8") as f:
        while streams and written < target:
            for subset in list(streams):
                try:
                    text = next(streams[subset])
                except StopIteration:
                    del streams[subset]
                    continue
                line = json.dumps({"doc_id": str(doc), "text": text, "source": f"redpajama_{subset}"},
                                  ensure_ascii=False) + "\n"
                f.write(line)
                written += len(line.encode("utf-8"))
                doc += 1
                if doc % 10_000 == 0:
                    print(f"{doc} documents, {written / 1024**3:.2f} GB")
                if written >= target:
                    break
    print(f"Wrote {doc} documents ({written / 1024**3:.2f} GB) to {output}")


def prefixes(corpus: Path, output: Path, model: str, num: int, seed: int, min_tokens: int = 60) -> None:
    """Reservoir-sample ``num`` documents and keep a 5-10 token prefix of each long-enough one."""
    from transformers import AutoTokenizer

    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model)
    reservoir = []
    with corpus.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(reservoir) < num:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < num:
                    reservoir[j] = line
    kept = 0
    with output.open("w", encoding="utf-8") as f:
        for line in reservoir:
            record = json.loads(line)
            ids = tok(record["text"], add_special_tokens=False)["input_ids"]
            if len(ids) < min_tokens:
                continue
            n = rng.randint(5, 10)
            prefix = tok.decode(ids[:n], skip_special_tokens=True)
            f.write(json.dumps({"prefix": prefix, "prefix_len": n, "source": record.get("source")},
                               ensure_ascii=False) + "\n")
            kept += 1
    print(f"Wrote {kept} prefixes to {output}")


def index(corpus: Path, name: str) -> None:
    """Index the corpus into Elasticsearch as overlapping 100-word chunks."""
    from leakpro.attacks.utils.extraction.elasticsearch_verifier import connect, index_corpus, iter_jsonl_documents

    client = connect(os.environ.get("ES_HOST"), os.environ.get("ES_USER"), "ES_PASSWORD", os.environ.get("ES_CA_CERT"))
    count = index_corpus(client, name, iter_jsonl_documents(str(corpus)))
    print(f"Indexed {count} chunks into {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--output", type=Path, default=Path("data/aux_corpus.jsonl"))
    d.add_argument("--target-gb", type=float, default=20)
    p = sub.add_parser("prefixes")
    p.add_argument("--corpus", type=Path, default=Path("data/aux_corpus.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/prefixes_conditional.jsonl"))
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Tokenizer of the audited model")
    p.add_argument("--num-prefixes", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    i = sub.add_parser("index")
    i.add_argument("--corpus", type=Path, default=Path("data/aux_corpus.jsonl"))
    i.add_argument("--index", default="redpajama_chunks")
    args = ap.parse_args()
    if args.cmd == "download":
        download(args.output, args.target_gb)
    elif args.cmd == "prefixes":
        prefixes(args.corpus, args.output, args.model, args.num_prefixes, args.seed)
    else:
        index(args.corpus, args.index)


if __name__ == "__main__":
    main()
