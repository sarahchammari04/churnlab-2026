"""Fetch the commit-pinned Telco CSV if absent and record its source and hash."""

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churnlab.data import EXPECTED_SHA256, RAW_CSV, SOURCE_URL, sha256_of  # noqa: E402

if not RAW_CSV.exists():
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, RAW_CSV)
else:
    print(f"{RAW_CSV} already present, not downloading.")

digest = sha256_of(RAW_CSV)
if digest != EXPECTED_SHA256:
    raise ValueError(f"SHA-256 mismatch: got {digest}, expected {EXPECTED_SHA256}")

meta = {
    "source_url": SOURCE_URL,
    "file": RAW_CSV.as_posix(),
    "sha256": digest,
    "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
meta_path = RAW_CSV.with_name("telco.source.json")
meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(json.dumps(meta, indent=2))
