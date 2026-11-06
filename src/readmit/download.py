"""Download the public UCI 'Diabetes 130-US Hospitals for Years 1999-2008' dataset.

Source: https://archive.ics.uci.edu/dataset/296 (CC BY 4.0). De-identified,
101,766 encounters from 130 US hospitals. If the URL below changes, download the
zip by hand from that page and put diabetic_data.csv in data/raw/.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://archive.ics.uci.edu/static/public/296/"
       "diabetes+130-us+hospitals+for+years+1999-2008.zip")


def main(out_dir: str = "data/raw") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "diabetic_data.csv"
    if target.exists():
        print(f"Already present: {target}")
        return target
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith("diabetic_data.csv"))
        target.write_bytes(z.read(name))
    print(f"Saved {target}")
    return target


if __name__ == "__main__":
    main(*sys.argv[1:])
