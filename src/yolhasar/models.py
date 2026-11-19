"""Sistemin çekirdek veri modelleri."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HasarTuru(str, Enum):
    """Tespit edilebilen yol hasarı türleri.

    Kodlar RDD2022 (Road Damage Dataset) ile uyumludur; böylece ileride
    eğitilecek YOLO modelinin sınıf indeksleri doğrudan eşlenebilir.
    """

    BOYUNA_CATLAK = "D00"
    ENINE_CATLAK = "D10"
    TIMSAH_SIRTI = "D20"
    CUKUR = "D40"

    @property
    def ad(self) -> str:
        return _ADLAR[self]

    @property
    def risk_agirligi(self) -> float:
        """Yol güvenliğine etki ağırlığı (0-1). Çukurlar araçlar için en tehlikelisidir."""
        return _RISK[self]

    @property
    def sinif_indeksi(self) -> int:
        """YOLO modelindeki sınıf indeksi."""
        return list(HasarTuru).index(self)

    @classmethod
    def indeksten(cls, indeks: int) -> HasarTuru:
        return list(cls)[indeks]

    @classmethod
    def koddan(cls, kod: str) -> HasarTuru:
        try:
            return cls(kod.strip().upper())
        except ValueError:
            raise ValueError(f"bilinmeyen hasar kodu: {kod!r}") from None


_ADLAR = {
    HasarTuru.BOYUNA_CATLAK: "Boyuna çatlak",
    HasarTuru.ENINE_CATLAK: "Enine çatlak",
    HasarTuru.TIMSAH_SIRTI: "Timsah sırtı çatlak",
    HasarTuru.CUKUR: "Çukur",
}

_RISK = {
    HasarTuru.BOYUNA_CATLAK: 0.4,
    HasarTuru.ENINE_CATLAK: 0.5,
    HasarTuru.TIMSAH_SIRTI: 0.7,
    HasarTuru.CUKUR: 1.0,
}


@dataclass(frozen=True)
class Kutu:
    """Piksel koordinatlarında sol üst köşe + boyut ile tanımlı sınırlayıcı kutu."""

    x: float
    y: float
    genislik: float
    yukseklik: float

    def __post_init__(self) -> None:
        if not (self.genislik > 0 and self.yukseklik > 0):
            raise ValueError(
                f"geçersiz kutu boyutu: genişlik={self.genislik}, yükseklik={self.yukseklik}"
            )

    @classmethod
    def koselerden(cls, x1: float, y1: float, x2: float, y2: float) -> Kutu:
        return cls(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    @classmethod
    def yolodan(
        cls, cx: float, cy: float, w: float, h: float, resim_g: int, resim_y: int
    ) -> Kutu:
        """YOLO formatından (normalize merkez x, merkez y, genişlik, yükseklik) dönüşüm."""
        return cls((cx - w / 2) * resim_g, (cy - h / 2) * resim_y, w * resim_g, h * resim_y)

    def yoloya(self, resim_g: int, resim_y: int) -> tuple[float, float, float, float]:
        """YOLO formatına dönüşüm: (cx, cy, w, h), hepsi 0-1 aralığında."""
        return (
            (self.x + self.genislik / 2) / resim_g,
            (self.y + self.yukseklik / 2) / resim_y,
            self.genislik / resim_g,
            self.yukseklik / resim_y,
        )

    @property
    def x2(self) -> float:
        return self.x + self.genislik

    @property
    def y2(self) -> float:
        return self.y + self.yukseklik

    @property
    def alan(self) -> float:
        return self.genislik * self.yukseklik

    @property
    def en_boy_orani(self) -> float:
        """> 1 yatay uzanan, < 1 dikey uzanan yapı. Çatlak yönü tahmininde kullanılır."""
        return self.genislik / self.yukseklik

    def kesisim(self, diger: Kutu) -> float:
        g = max(0.0, min(self.x2, diger.x2) - max(self.x, diger.x))
        y = max(0.0, min(self.y2, diger.y2) - max(self.y, diger.y))
        return g * y

    def iou(self, diger: Kutu) -> float:
        """Kesişim / Birleşim oranı. NMS ve mAP hesabında kullanılır."""
        kesisim = self.kesisim(diger)
        birlesim = self.alan + diger.alan - kesisim
        return kesisim / birlesim if birlesim > 0 else 0.0
