"""Ön işleme hattının adımlarını yan yana gösteren demo.

Kullanım:
    python scripts/on_isleme_demo.py yol_goruntusu.jpg cikti.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolhasar import preprocessing as pp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("girdi", type=Path)
    ap.add_argument("cikti", type=Path)
    ap.add_argument("--uzun-kenar", type=int, default=640)
    ap.add_argument("--sigma", type=float, default=1.5)
    a = ap.parse_args()

    img = pp.boyutlandir(pp.yukle(a.girdi), a.uzun_kenar)
    gri = pp.gri_ton(img)
    clahe = pp.kontrast_dengele(gri)
    son = pp.gauss_bulaniklastir(clahe, a.sigma)

    adimlar = [("1) Gri ton", gri), ("2) CLAHE", clahe), (f"3) Gauss s={a.sigma}", son)]
    paneller = []
    for baslik, p in adimlar:
        p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        cv2.putText(p, baslik, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        paneller.append(p)
    pp.kaydet(np.hstack([img, *paneller]), a.cikti)
    print(f"kaydedildi: {a.cikti}")


if __name__ == "__main__":
    main()
