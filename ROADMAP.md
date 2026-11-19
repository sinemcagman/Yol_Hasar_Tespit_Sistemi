# Yol Haritası

Proje 5 faz ve 20 PR'a bölünmüştür. Her PR kendi başına çalışabilen, test
edilebilen ve `main` dalına birleştirilebilen küçük bir iş paketidir.

## Mimari

```
Yol_Hasar_Tespit_Sistemi/
├── src/yolhasar/
│   ├── models.py          # Veri modelleri: hasar türü, kutu, konum, tespit, rapor
│   ├── preprocessing.py   # Görüntü ön işleme (OpenCV)
│   ├── classical/         # Klasik görüntü işleme ile tespit      (Faz 2)
│   ├── ml/                # YOLOv8 eğitim / çıkarım               (Faz 3)
│   ├── storage/           # SQLite kayıt katmanı                  (Faz 4)
│   ├── api/               # FastAPI REST servisi                  (Faz 4)
│   └── cli.py             # Komut satırı aracı                    (Faz 2)
├── web/                   # Harita tabanlı izleme paneli          (Faz 5)
├── tests/                 # pytest testleri
└── data/                  # Veri seti (git'e eklenmez)
```

**Teknolojiler:** Python 3.10+, OpenCV, NumPy, Ultralytics YOLOv8, FastAPI,
SQLite, Leaflet.js, pytest.

**Yaklaşım:** Önce klasik görüntü işleme (kenar tespiti, eşikleme, kontur
analizi) ile çalışan bir temel (baseline) sistem kurulur. Ardından derin
öğrenme modeli (YOLOv8) eğitilir ve iki yöntem doğruluk/hız açısından
karşılaştırılır.

## Hasar sınıfları (RDD2022 standardı)

| Kod  | Sınıf                     |
|------|---------------------------|
| D00  | Boyuna çatlak             |
| D10  | Enine çatlak              |
| D20  | Timsah sırtı (ağ) çatlak  |
| D40  | Çukur                     |

## Fazlar ve PR'lar

### Faz 1 — Temel altyapı
- [x] **PR 1** — Proje iskeleti: paket yapısı, `pyproject.toml`, pytest, ruff
- [x] **PR 2** — Veri modelleri: hasar türü, sınırlayıcı kutu, GPS konumu, şiddet puanlama, JSON rapor
- [ ] **PR 3** — Görüntü ön işleme: gri tonlama, yeniden boyutlandırma, Gauss bulanıklaştırma, CLAHE

### Faz 2 — Klasik görüntü işleme ile temel tespit
- [ ] **PR 4** — Kenar çıkarımı (Canny/Sobel) ve adaptif eşikleme
- [ ] **PR 5** — Morfolojik işlemlerle gürültü temizleme
- [ ] **PR 6** — Kontur analizi ve kural tabanlı sınıflandırma (çatlak / çukur)
- [ ] **PR 7** — Komut satırı aracı: JSON rapor ve işaretlenmiş görüntü çıktısı

### Faz 3 — Derin öğrenme
- [ ] **PR 8** — RDD2022 veri setinin YOLO formatına dönüştürülmesi
- [ ] **PR 9** — Veri artırma ve eğitim/doğrulama/test ayrımı
- [ ] **PR 10** — YOLOv8 eğitimi
- [ ] **PR 11** — Değerlendirme: mAP@0.5, sınıf bazlı precision/recall
- [ ] **PR 12** — Çıkarım (inference) modülü ve tespit hattına entegrasyon
- [ ] **PR 13** — Klasik yöntem ile YOLO karşılaştırması (doğruluk ve hız)

### Faz 4 — Konum ve veri katmanı
- [ ] **PR 14** — EXIF'ten GPS okuma, videodan kare örnekleme
- [ ] **PR 15** — SQLite kayıt katmanı
- [ ] **PR 16** — REST API (FastAPI): görüntü yükleme, sonuç sorgulama

### Faz 5 — Arayüz ve dağıtım
- [ ] **PR 17** — Web paneli: hasarların harita üzerinde gösterimi (Leaflet)
- [ ] **PR 18** — Onarım önceliklendirme raporu (CSV/PDF)
- [ ] **PR 19** — Docker ile dağıtım
- [ ] **PR 20** — Uçtan uca testler ve son dokümantasyon
