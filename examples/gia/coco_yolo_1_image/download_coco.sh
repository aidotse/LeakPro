#!/usr/bin/env bash
# Download and verify the COCO 2017 archives used by this example.
#
# images.cocodataset.org is an S3 bucket whose name contains dots, so its
# virtual-host HTTPS certificate does not match the host name. The path-style
# S3 URL serves the same objects over a valid certificate. Every archive is
# checked against a pinned SHA-256 before anything is extracted; a mismatch
# stops the script.
set -euo pipefail

BASE_URL="${COCO_BASE_URL:-https://s3.amazonaws.com/images.cocodataset.org}"

# path on the server  SHA-256 of the archive
ARCHIVES=(
  "zips/train2017.zip                        69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929"
  "zips/val2017.zip                          4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
  "annotations/annotations_trainval2017.zip  113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
)

mkdir -p COCO/images/train2017 COCO/images/val2017 COCO/annotations
cd COCO

fetch_and_verify() {
  local remote_path="$1" expected="$2" name
  name="$(basename "$remote_path")"
  if [ ! -f "$name" ]; then
    echo "Downloading $name"
    curl -fL --retry 3 --retry-delay 5 -o "$name.part" "$BASE_URL/$remote_path"
    mv "$name.part" "$name"
  fi
  echo "Verifying $name"
  if ! echo "$expected  $name" | sha256sum -c --quiet -; then
    echo "ERROR: $name does not match its pinned SHA-256; refusing to extract it." >&2
    echo "       Delete the file and rerun, or check whether the upstream archive changed." >&2
    exit 1
  fi
}

for entry in "${ARCHIVES[@]}"; do
  # shellcheck disable=SC2086  # intentional word split: "path sha256"
  fetch_and_verify $entry
done

# Unzip only after every archive verified
unzip -q train2017.zip -d images/
unzip -q val2017.zip -d images/
unzip -q annotations_trainval2017.zip
cd ..
python3 prepare_coco.py
