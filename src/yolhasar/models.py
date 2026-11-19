"""Sistemin çekirdek veri modelleri."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum

DUNYA_YARICAPI_M = 6_371_000.0


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
    def yolodan(cls, cx: float, cy: float, w: float, h: float, resim_g: int, resim_y: int) -> Kutu:
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


@dataclass(frozen=True)
class Konum:
    """WGS84 enlem/boylam çifti."""

    enlem: float
    boylam: float

    def __post_init__(self) -> None:
        if not (-90 <= self.enlem <= 90 and -180 <= self.boylam <= 180):
            raise ValueError(f"geçersiz koordinat: enlem={self.enlem}, boylam={self.boylam}")

    def mesafe_m(self, diger: Konum) -> float:
        """Haversine formülü ile iki nokta arası büyük daire mesafesi (metre).

        Aynı hasarın farklı görüntülerde tekrar raporlanmasını önlemek için
        kullanılacaktır (ör. birkaç metre içindeki aynı tür tespitleri birleştirme).
        """
        e1, e2 = math.radians(self.enlem), math.radians(diger.enlem)
        de = e2 - e1
        db = math.radians(diger.boylam - self.boylam)
        a = math.sin(de / 2) ** 2 + math.cos(e1) * math.cos(e2) * math.sin(db / 2) ** 2
        return 2 * DUNYA_YARICAPI_M * math.asin(math.sqrt(a))


class Siddet(str, Enum):
    """Hasarın şiddet seviyesi; onarım önceliklendirmesinde kullanılır."""

    DUSUK = "dusuk"
    ORTA = "orta"
    YUKSEK = "yuksek"
    KRITIK = "kritik"

    @classmethod
    def puandan(cls, puan: float) -> Siddet:
        if puan < 25:
            return cls.DUSUK
        if puan < 50:
            return cls.ORTA
        if puan < 75:
            return cls.YUKSEK
        return cls.KRITIK

    @property
    def derece(self) -> int:
        return list(Siddet).index(self)


def siddet_puani(tur: HasarTuru, kutu: Kutu, guven: float, resim_alani: float) -> float:
    """0-100 aralığında şiddet puanı.

    ``puan = 100 · risk_ağırlığı · (0.6 · alan_etkisi + 0.4 · güven)``

    ``alan_etkisi``, kutunun görüntü alanına oranının kareköküyle ölçeklenir;
    görüntünün %10'unu kaplayan hasar tam etki (1.0) kabul edilir.
    """
    oran = kutu.alan / resim_alani if resim_alani > 0 else 0.0
    alan_etkisi = min(1.0, math.sqrt(oran / 0.10))
    puan = 100 * tur.risk_agirligi * (0.6 * alan_etkisi + 0.4 * guven)
    return max(0.0, min(100.0, puan))


@dataclass(frozen=True)
class Tespit:
    """Tek bir hasar tespiti."""

    tur: HasarTuru
    kutu: Kutu
    guven: float
    puan: float
    siddet: Siddet

    @classmethod
    def olustur(cls, tur: HasarTuru, kutu: Kutu, guven: float, resim_alani: float) -> Tespit:
        if not 0 <= guven <= 1:
            raise ValueError(f"güven skoru 0 ile 1 arasında olmalı: {guven}")
        puan = siddet_puani(tur, kutu, guven, resim_alani)
        return cls(tur, kutu, guven, round(puan, 2), Siddet.puandan(puan))


@dataclass
class Rapor:
    """Bir görüntü için üretilen tespit raporu."""

    goruntu: str
    genislik: int
    yukseklik: int
    konum: Konum | None = None
    tespitler: list[Tespit] = field(default_factory=list)

    @property
    def resim_alani(self) -> float:
        return float(self.genislik * self.yukseklik)

    def ekle(self, tur: HasarTuru, kutu: Kutu, guven: float) -> Tespit:
        t = Tespit.olustur(tur, kutu, guven, self.resim_alani)
        self.tespitler.append(t)
        return t

    def en_yuksek_siddet(self) -> Siddet | None:
        if not self.tespitler:
            return None
        return max((t.siddet for t in self.tespitler), key=lambda s: s.derece)

    def tur_sayilari(self) -> dict[str, int]:
        return dict(Counter(t.tur.value for t in self.tespitler))

    def puana_gore_sirala(self) -> None:
        self.tespitler.sort(key=lambda t: t.puan, reverse=True)

    def sozluk(self) -> dict:
        d = asdict(self)
        if d["konum"] is None:
            del d["konum"]
        for t in d["tespitler"]:
            t["tur"] = t["tur"].value
            t["siddet"] = t["siddet"].value
        return d

    def json(self) -> str:
        return json.dumps(self.sozluk(), ensure_ascii=False, indent=2)

    @classmethod
    def jsondan(cls, metin: str) -> Rapor:
        d = json.loads(metin)
        konum = Konum(**d["konum"]) if d.get("konum") else None
        tespitler = [
            Tespit(
                tur=HasarTuru(t["tur"]),
                kutu=Kutu(**t["kutu"]),
                guven=t["guven"],
                puan=t["puan"],
                siddet=Siddet(t["siddet"]),
            )
            for t in d.get("tespitler", [])
        ]
        return cls(d["goruntu"], d["genislik"], d["yukseklik"], konum, tespitler)
