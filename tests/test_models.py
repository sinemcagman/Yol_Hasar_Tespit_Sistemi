import pytest

from yolhasar.models import HasarTuru


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
