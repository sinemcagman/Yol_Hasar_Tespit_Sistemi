"""Sistemin çekirdek veri modelleri."""

from __future__ import annotations

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
