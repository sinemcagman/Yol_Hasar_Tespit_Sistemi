import pytest

from yolhasar.models import HasarTuru, Konum, Kutu


def test_koddan_donusum():
    assert HasarTuru.koddan("D40") is HasarTuru.CUKUR
    assert HasarTuru.koddan(" d00 ") is HasarTuru.BOYUNA_CATLAK
    with pytest.raises(ValueError):
        HasarTuru.koddan("D99")


def test_indeks_gidis_donus():
    for t in HasarTuru:
        assert HasarTuru.indeksten(t.sinif_indeksi) is t
    assert HasarTuru.CUKUR.sinif_indeksi == 3


def test_cukur_en_riskli():
    assert max(HasarTuru, key=lambda t: t.risk_agirligi) is HasarTuru.CUKUR


@pytest.mark.parametrize("g,y", [(0, 5), (5, -1), (float("nan"), 1)])
def test_gecersiz_kutu(g, y):
    with pytest.raises(ValueError):
        Kutu(0, 0, g, y)


def test_ayni_kutu_iou_bir():
    k = Kutu(10, 10, 20, 20)
    assert k.iou(k) == pytest.approx(1.0)


def test_ayrik_kutular_iou_sifir():
    assert Kutu(0, 0, 10, 10).iou(Kutu(20, 20, 10, 10)) == 0.0


def test_yari_ortusen_kutular():
    # kesişim 50, birleşim 150
    assert Kutu(0, 0, 10, 10).iou(Kutu(5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_yolo_gidis_donus():
    k = Kutu(100, 50, 200, 100)
    geri = Kutu.yolodan(*k.yoloya(640, 480), 640, 480)
    assert geri.x == pytest.approx(k.x) and geri.genislik == pytest.approx(k.genislik)


def test_koselerden_sira_bagimsiz():
    assert Kutu.koselerden(10, 10, 0, 0) == Kutu(0, 0, 10, 10)


@pytest.mark.parametrize("e,b", [(91, 0), (0, -181), (float("nan"), 0)])
def test_gecersiz_koordinat(e, b):
    with pytest.raises(ValueError):
        Konum(e, b)


def test_ayni_nokta_mesafe_sifir():
    k = Konum(41.0082, 28.9784)
    assert k.mesafe_m(k) == pytest.approx(0.0, abs=1e-6)


def test_istanbul_ankara_mesafesi():
    # Kuş uçuşu yaklaşık 350 km
    km = Konum(41.0082, 28.9784).mesafe_m(Konum(39.9334, 32.8597)) / 1000
    assert 340 < km < 360
