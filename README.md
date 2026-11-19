# Yol Hasar Tespit Sistemi

Yol yüzeyindeki hasarları (çukur, boyuna/enine çatlak, timsah sırtı çatlak)
görüntülerden otomatik olarak tespit eden, şiddetini puanlayan ve konum bazlı
raporlayan bir sistem.

> Proje geliştirme aşamasındadır. Ayrıntılı plan için [ROADMAP.md](ROADMAP.md) dosyasına bakınız.

## Kurulum

Gereksinimler: Python 3.10+

```bash
git clone https://github.com/sinemcagman/Yol_Hasar_Tespit_Sistemi.git
cd Yol_Hasar_Tespit_Sistemi
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Geliştirme

```bash
pytest            # testler
ruff check .      # kod stili / statik analiz
```
