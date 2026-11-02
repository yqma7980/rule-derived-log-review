"""Download pinned organizer data; resume and record content fingerprints."""
import hashlib
import argparse
import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c8d01ee92c1c8e1ecba36f96cca6ea7b689338a1"
EXPECTED_SHA256 = "e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9"
URL = f"https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/{COMMIT}/lithology_competition/data/train.zip"

def download(url, path):
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 1200
    for attempt in range(5):
        start = path.stat().st_size if path.exists() else 0
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-"} if start else {}, stream=True, timeout=(30,90)) as response:
                response.raise_for_status()
                if start and response.status_code != 206:
                    raise RuntimeError("Server did not honor resume; preserved partial file")
                with path.open("ab" if start else "wb") as handle:
                    for chunk in response.iter_content(1024*1024):
                        handle.write(chunk)
                        if time.monotonic() > deadline:
                            raise TimeoutError("Total download deadline exceeded")
            return
        except (requests.RequestException, TimeoutError) as exc:
            print(f"Attempt {attempt+1}: {exc}", flush=True)
            if time.monotonic() > deadline or attempt == 4:
                raise
            time.sleep(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--download', action='store_true', help='Explicitly enable network acquisition')
    args = parser.parse_args()
    if not args.download:
        parser.error('Use --download to explicitly enable network acquisition')
    ROOT = args.workspace.resolve()
    target = ROOT / "data/external/FORCE_train.zip"
    if not zipfile.is_zipfile(target):
        download(URL, target)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        contents = [{"name": x.filename, "size": x.file_size, "crc32": x.CRC} for x in archive.infolist()]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_SHA256, "Pinned input hash mismatch"
    (ROOT / "qa").mkdir(exist_ok=True)
    record = {"url": URL, "commit": COMMIT, "dataset_doi": "10.5281/zenodo.4351156",
              "scope": "Organizer train.zip only; not the 118-well LAS release",
              "bytes": target.stat().st_size,
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "contents": contents}
    (ROOT / "qa/SOURCE_DOWNLOAD.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
