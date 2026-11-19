"""Görüntü ön işleme adımları.

Tespit algoritmaları (klasik ve derin öğrenme) girdiyi tutarlı biçimde bekler:
makul boyut, gri ton, gürültüsü azaltılmış ve kontrastı dengelenmiş görüntü.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def yukle(yol: str | Path) -> np.ndarray:
    """Görüntüyü BGR formatında yükler.

    ``cv2.imread`` Windows'ta Türkçe karakter içeren yolları okuyamadığı için
    dosya önce bayt olarak okunup ``cv2.imdecode`` ile çözülür.
    """
    veri = np.fromfile(str(yol), dtype=np.uint8)
    goruntu = cv2.imdecode(veri, cv2.IMREAD_COLOR)
    if goruntu is None:
        raise ValueError(f"görüntü okunamadı: {yol}")
    return goruntu


def kaydet(goruntu: np.ndarray, yol: str | Path) -> None:
    """Görüntüyü diske yazar; format uzantıdan belirlenir."""
    yol = Path(yol)
    basarili, tampon = cv2.imencode(yol.suffix, goruntu)
    if not basarili:
        raise ValueError(f"görüntü kodlanamadı: {yol}")
    tampon.tofile(str(yol))


def gri_ton(goruntu: np.ndarray) -> np.ndarray:
    """BGR görüntüyü gri tona çevirir (BT.601: Y = 0.299R + 0.587G + 0.114B).

    Çatlak ve çukurlar renkten çok parlaklık farkıyla ayırt edildiği için
    sonraki adımlar tek kanallı görüntüyle çalışır. Zaten gri olan görüntü
    olduğu gibi döner.
    """
    if goruntu.ndim == 2:
        return goruntu
    return cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)


def boyutlandir(goruntu: np.ndarray, uzun_kenar: int = 640) -> np.ndarray:
    """Uzun kenar ``uzun_kenar`` piksel olacak şekilde en-boy oranını koruyarak küçültür.

    Telefon kameralarından gelen 4000×3000 gibi görüntüleri işlemek hem yavaş
    hem de gereksizdir; YOLO modeli de 640 piksel girdiyle eğitilecektir.
    Görüntü zaten küçükse büyütülmez.
    """
    if uzun_kenar <= 0:
        raise ValueError("uzun_kenar pozitif olmalı")
    y, g = goruntu.shape[:2]
    mevcut = max(g, y)
    if mevcut <= uzun_kenar:
        return goruntu
    olcek = uzun_kenar / mevcut
    yeni = (max(1, round(g * olcek)), max(1, round(y * olcek)))
    # INTER_AREA küçültmede örtüşme (aliasing) bozulmasını en aza indirir
    return cv2.resize(goruntu, yeni, interpolation=cv2.INTER_AREA)
