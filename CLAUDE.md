# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Ortam

- Python 3.12, WSL2 (Ubuntu)
- Sanal ortam: `venv/` — her komut öncesi `source venv/bin/activate`
- `.env` dosyası proje kökünde; `config.py` mutlak yol ile bulur (çalışma dizini bağımsız)
- Servis olarak çalışır: `sudo systemctl status bist-robot`

## Sık Kullanılan Komutlar

```bash
# Sunucu
uvicorn app.main:app --reload --port 8000
python -m app.main

# Testler
pytest tests/ -q                          # tümü (scheduler hariç hızlı)
pytest tests/ --ignore=tests/test_scheduler.py -q   # event loop izolasyonu için
pytest tests/test_scheduler.py -q         # yavaş — APScheduler gerçek timer bekler
pytest tests/test_indicators.py::TestAddRSI -v      # tek sınıf
pytest tests/test_integration.py -v       # 5 temel işlev

# Servis (WSL)
sudo systemctl start|stop|restart|status bist-robot
sudo journalctl -u bist-robot -f

# Masaüstü senkronizasyon
rsync -av --exclude='venv/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='bist_robot.db' \
  /home/hyayan/bist-sinyal-robotu/ \
  "/mnt/c/Users/hasan/Desktop/claude projelerim/bist-sinyal-robotu/"
```

## BIST100 Tarama

| Fonksiyon | Sembol Sayısı | Kaynak |
|-----------|--------------|--------|
| `scan_bist30()` | 30 | `app/data/bist100.py` → BIST30_SYMBOLS |
| `scan_bist50()` | 50 | `app/data/bist100.py` → BIST50_SYMBOLS |
| `scan_bist100()` | 100 | `app/data/bist100.py` → BIST100_SYMBOLS |

Üçü de `_scan_parallel()` üzerinden ThreadPoolExecutor (10 worker) kullanır.
Sonuçlar `strength_score` (0-100) ile sıralanır; likidite filtresi ≥ 50M TL.
1 saatlik module-level önbellek: `clear_scan_cache()` ile temizlenir.

## Mimari

Sinyal üretim pipeline'ı doğrusal akar:

```
fetch_symbol_data()      → app/data/fetcher.py        FetchResult + to_cache_records()
        ↓
add_indicators()         → app/indicators/technical.py  DataFrame + yeni kolonlar
        ↓
generate_signal()        → app/strategies/trend_breakout.py  BreakoutSignal dict
        ↓
is_market_favorable()    → app/risk/market_filter.py   XU100 > EMA50 kontrolü (1s TTL)
        ↓
scan_market()            → app/signals/scanner.py      EARLY_WATCH / BUY / LATE_BREAKOUT listesi
        ↓
send_telegram_message()  → app/notifications/telegram.py
```

### Kolon Standardı

Tüm iç DataFrame'ler **küçük harf** kullanır: `open, high, low, close, volume`.  
`backtesting.py` için `_to_bt_df()` büyük harfe çevirir.  
`_clean_ohlcv()` timezone kaldırır, sıfır/negatif fiyat ve ters H/L satırlarını atar.

### Strateji Sistemi

`BaseStrategy` → `generate_signal(df, symbol) → Optional[StrategySignal]`

Yeni strateji eklemek için:
1. `app/strategies/` altında dosya oluştur, `BaseStrategy`'yi kalıt al
2. `app/strategies/__init__.py`'deki `STRATEGIES` listesine ekle — scanner ve generator otomatik alır

`trend_breakout.py`'de iki ayrı arayüz var:
- `generate_signal(df)` → `BreakoutSignal` dict (scanner doğrudan kullanır, `details` içerir)
- `TrendBreakoutStrategy.generate_signal(df, symbol)` → `StrategySignal` (motor entegrasyonu)

### Risk Katmanları

İki bağımsız filtre sırayla çalışır:

1. **`market_filter.py`** — XU100 endeks filtresi. Fail-open: veri yoksa BUY'ları engellemez.
2. **`risk/manager.py`** — R/R oranı (min 1.5), sinyal gücü, açık sinyal limiti.

### Grafik-Bazlı Stop/Hedef Seviyeleri (ATR'ye ek)

BUY/LATE sinyallerinde ATR seviyelerinin (`trend_breakout._atr_sl_tp`) yanına iki
**fail-open** katman eklenir; `scanner._graphic_levels()` çağırır:

1. **`app/indicators/structure.py`** — swing dip/tepe + `resistance_20` bazlı
   stop/hedef (`structure_sl_tp`). Deterministik, hep çalışır; yapı yoksa `None`.
2. **`app/ai/vision_levels.py`** — `app/charts/render.py` ile üretilen işaretli mum
   grafiğini Claude'a (`claude-opus-4-8`, vision + structured outputs) gönderip
   seviyeleri teyit/düzeltir. **Varsayılan kapalı**: `enable_ai_levels` + dolu
   `anthropic_api_key` gerekir; eksikse sessizce atlanır.

Seviyeler `ScanResult` alanları olarak taşınır (`struct_*`, `ai_*`, `chart_b64`).
Telegram'da `_levels_block` / `_best_level` **tek temiz stop/hedef** gösterir:
öncelik AI (🤖 etiketli), yoksa sessizce yapı, o da yoksa ATR seviyesine düşer.
`enable_signal_charts` açıksa scheduler `send_telegram_photo` ile grafiği yollar.
Config bayrakları: `app/config.py` → `anthropic_api_key`, `enable_ai_levels`,
`enable_signal_charts`.

### Backtest

`runner.py` → `TrendBreakoutBT(Strategy)` backtesting.py sınıfı  
Pozisyon büyüklüğü: `risk_per_trade (1%) / stop_loss_pct (3%) ≈ %33 sermaye`  
`BacktestResult.to_dict()` / `.to_json()` doğrudan API yanıtı olarak kullanılır.

### Zamanlayıcı

`app/scheduler.py` — `AsyncIOScheduler`, her gün 4 kez TR saati (`Europe/Istanbul`):
10:30 · 12:30 · 15:30 · 18:10 (Pazartesi–Cuma)
Her tarama `scan_bist100()` çağırır, tüm sinyal tiplerini tek mesajda `format_bist100_full_report()` ile Telegram'a gönderir.
Lifespan'da başlar/durur. Test ortamında `conftest.py`'deki session-scope mock ile devre dışı bırakılır.

## Test Mimarisi

- Tüm testler ağdan bağımsız — yfinance `patch("app.data.fetcher.yf.Ticker")` ile mock'lanır
- `conftest.py` session-scope `_mock_scheduler_global` fixture'ı ile `app.main.scheduler` mock'lanır; `TestBISTScheduler` ve `TestRunDailyScan` kendi `BISTScheduler()` instance'ları oluşturduğu için etkilenmez
- `test_scheduler.py` gerçek APScheduler timer'ı beklediğinden (~80s) diğer testlerden ayrı çalıştırılır
- `make_ohlcv()` / `make_fetch_result()` fabrika fonksiyonları `conftest.py`'de tanımlı

## Önemli Kısıtlamalar

- **Gerçek emir yok.** Sistem yalnızca sinyal üretir.
- `.env` dosyasındaki `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID` eksikse bildirimler sessizce atlanır, hata fırlatılmaz.
- `DEFAULT_SYMBOLS` `.env`'de artık kullanılmaz; `app/data/symbols.py`'deki `SymbolRegistry` (singleton `registry`) çalışma zamanında yönetilir.

## Dosya Konumları

| Amaç | Dosya |
|------|-------|
| Tüm ayarlar | `app/config.py` → `settings` singleton |
| Sembol listesi | `app/data/symbols.py` → `registry` singleton |
| BIST endeks listeleri | `app/data/bist100.py` → BIST30/50/100_SYMBOLS |
| İndikatör kolonları | `app/indicators/technical.py` → `_INDICATOR_COLS` |
| Strateji sabitleri | `app/strategies/trend_breakout.py` → `_STOP_LOSS_PCT`, `_TAKE_PROFIT_PCT` vb. |
| Strength score sabitleri | `app/signals/scanner.py` → `_SCORE_*` sabitleri |
| Yapı (swing/destek-direnç) seviyeleri | `app/indicators/structure.py` |
| Sinyal grafiği render | `app/charts/render.py` |
| Claude vision seviye teyidi | `app/ai/vision_levels.py` |
| API endpoint'leri | `app/main.py` |
| Systemd servis | `/etc/systemd/system/bist-robot.service` |
