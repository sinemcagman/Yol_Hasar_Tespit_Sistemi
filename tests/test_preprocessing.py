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


@pytest.mark.parametrize(
    "boyut,uzun,beklenen",
    [((3000, 4000), 640, (480, 640)), ((1200, 300), 600, (600, 150)), ((50, 100), 640, (50, 100))],
)
def test_boyutlandir_oran_korunur(boyut, uzun, beklenen):
    img = np.zeros(boyut, dtype=np.uint8)
    assert pp.boyutlandir(img, uzun).shape == beklenen


def test_boyutlandir_renkli():
    img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    assert pp.boyutlandir(img, 500).shape == (250, 500, 3)


def test_boyutlandir_gecersiz():
    with pytest.raises(ValueError):
        pp.boyutlandir(np.zeros((5, 5), dtype=np.uint8), 0)


def test_duz_goruntu_bulaniklastirmayla_degismez():
    img = np.full((20, 20), 128, dtype=np.uint8)
    assert np.array_equal(pp.gauss_bulaniklastir(img, 2.0), img)


def test_parlak_nokta_yayilir():
    img = np.zeros((21, 21), dtype=np.uint8)
    img[10, 10] = 255
    b = pp.gauss_bulaniklastir(img, 1.0)
    assert 0 < b[10, 11] < b[10, 10] < 255


def test_gauss_gecersiz_sigma():
    with pytest.raises(ValueError):
        pp.gauss_bulaniklastir(np.zeros((5, 5), dtype=np.uint8), 0)


def test_clahe_dusuk_kontrasti_artirir():
    # 100-130 aralığına sıkışmış düşük kontrastlı görüntü
    rng = np.random.default_rng(1)
    img = rng.integers(100, 131, (64, 64), dtype=np.uint8)
    cikti = pp.kontrast_dengele(img)
    assert cikti.std() > img.std()


def test_clahe_renkli_reddedilir():
    with pytest.raises(ValueError):
        pp.kontrast_dengele(np.zeros((8, 8, 3), dtype=np.uint8))
