import numpy as np
import pytest

from yolhasar import preprocessing as pp


def test_saf_renkler_gri_degeri():
    img = np.zeros((1, 3, 3), dtype=np.uint8)
    img[0, 0] = (0, 0, 255)  # BGR: kırmızı
    img[0, 1] = (0, 255, 0)  # yeşil
    img[0, 2] = (255, 255, 255)  # beyaz
    g = pp.gri_ton(img)
    assert g.shape == (1, 3)
    assert list(g[0]) == [76, 150, 255]


def test_gri_goruntu_degismez():
    g = np.full((4, 4), 7, dtype=np.uint8)
    assert pp.gri_ton(g) is g


def test_kaydet_ve_yukle_turkce_yol(tmp_path):
    yol = tmp_path / "çukur_örneği.png"
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    pp.kaydet(img, yol)
    assert np.array_equal(pp.yukle(yol), img)


def test_bozuk_dosya(tmp_path):
    yol = tmp_path / "bozuk.jpg"
    yol.write_bytes(b"resim degil")
    with pytest.raises(ValueError):
        pp.yukle(yol)
