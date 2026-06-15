# BIST Sinyal Robotu

Python tabanlı Borsa İstanbul teknik analiz ve sinyal üretim sistemi.

---

> **⚠️ ÖNEMLİ UYARI**
>
> Bu yazılım **yatırım tavsiyesi değildir.** Üretilen sinyaller yalnızca teknik analiz
> algoritmalarına dayanır; finansal danışmanlık, alım-satım önerisi veya kâr garantisi
> içermez. Borsa yatırımları zarar riski taşır. Tüm yatırım kararlarınızı kendi
> araştırmanıza ve lisanslı bir finansal danışmana danışarak alınız.

---

## İçindekiler

1. [Proje Amacı](#proje-amacı)
2. [Özellikler](#özellikler)
3. [BIST100 Tam Tarama Sistemi](#bist100-tam-tarama-sistemi)
4. [Kurulum](#kurulum)
5. [.env Ayarları](#env-ayarları)
6. [Çalıştırma](#çalıştırma)
7. [API Endpoint'leri](#api-endpointleri)
8. [Örnek API İstekleri](#örnek-api-i̇stekleri)
9. [Backtest Nasıl Yapılır](#backtest-nasıl-yapılır)
10. [Telegram Nasıl Bağlanır](#telegram-nasıl-bağlanır)
11. [Mimari](#mimari)
12. [Testler](#testler)
13. [Yeni Strateji Eklemek](#yeni-strateji-eklemek)
14. [Risk Uyarıları](#risk-uyarıları)

---

## Proje Amacı

BIST Sinyal Robotu; Borsa İstanbul'da işlem gören hisse senetlerini teknik analiz
yöntemleriyle izleyen, kırılım sinyalleri üreten ve bu sinyalleri Telegram üzerinden
ileten bir araştırma aracıdır.

**Sistem şunları yapar:**
- yfinance üzerinden günlük OHLCV verisi çeker
- EMA, RSI, MACD, ATR, Bollinger Bands gibi indikatörleri hesaplar
- Trend + hacimli kırılım stratejisiyle EARLY_WATCH/BUY/LATE_BREAKOUT sinyali üretir
- XU100 endeks filtresiyle düşüş trendinde AL sinyallerini engeller
- Geçmiş veri üzerinde backtest çalıştırır ve performans raporlar
- Sinyalleri Telegram'a gönderir
- FastAPI ile JSON API sunar

**Sistem şunları yapmaz:**
- Gerçek emir göndermez
- Yatırım tavsiyesi vermez
- Kâr garantisi sunmaz

---

## Özellikler

| Modül | Açıklama |
|-------|----------|
| `data/fetcher.py` | yfinance ile BIST hisse verisi, hata yönetimi, SQLite cache hazırlığı |
| `data/symbols.py` | Sembol kayıt defteri, CSV yükleme/kaydetme, `.IS` normalleştirme |
| `indicators/technical.py` | EMA20/50/200, RSI14, MACD, ATR14, Bollinger, 20 günlük direnç, volatilite daralması |
| `strategies/trend_breakout.py` | Trend + hacimli kırılım — 5 koşul, güç skoru, risk seviyesi |
| `strategies/ema_crossover.py` | EMA20/50 golden cross + RSI filtresi |
| `strategies/rsi_bb.py` | RSI aşırı bölge + Bollinger Band dokunuşu |
| `backtest/runner.py` | backtesting.py motoru, pozisyon büyüklüğü yönetimi, JSON çıktı |
| `signals/scanner.py` | Tüm piyasayı tarar, EARLY_WATCH / BUY / LATE_BREAKOUT sinyallerini listeler |
| `risk/manager.py` | R/R filtresi, pozisyon büyüklüğü, açık sinyal limiti |
| `risk/market_filter.py` | XU100 Close > EMA50 kontrolü, 1 saatlik önbellek |
| `notifications/telegram.py` | Mesaj formatlama, retry mantığı, günlük özet |

---

## BIST100 Tam Tarama Sistemi

### Genel Bakış

`scan_bist100()`, BIST100 endeksindeki tüm hisseleri **paralel olarak** tarar ve
en güçlü BUY sinyallerini Telegram'a gönderir.

| Özellik | Detay |
|---------|-------|
| Evren | BIST100 — 100 hisse (XU100) |
| Tarama | ThreadPoolExecutor, 10 paralel bağlantı |
| Süre | Hedef < 90 saniye |
| Önbellek | 1 saatlik TTL — aynı hisse 1 saat içinde tekrar çekilmez |
| Likidite Filtresi | Ortalama günlük TL hacmi ≥ 50 milyon TL |
| Tarama Saatleri | 10:30 · 12:30 · 15:30 · 18:10 (Pazartesi–Cuma, TR saati) |

### strength\_score Puanlama Sistemi

Her sinyal, 5 kritere göre 0–100 arası bir **güç skoru** alır:

| Kriter | Puan | Koşul |
|--------|------|-------|
| EMA trend güçlü | +20 | EMA20 > EMA50 × 1.01 |
| Hacim patlaması | +25 | Günlük hacim ≥ 20 günlük ortalama × 2.0 |
| RSI ideal aralık | +15 | 55 ≤ RSI14 ≤ 70 |
| Güçlü breakout | +25 | Kapanış > 20 günlük direnç × 1.01 |
| MACD pozitif | +15 | MACD > 0 ve MACD > MACD signal |
| **Toplam** | **100** | |

Sinyaller `strength_score`'a göre sıralanır; en güçlü 5 tanesi Telegram'a gönderilir.

### Telegram Mesaj Formatı

```
🚨 BIST100 SİNYALLERİ
🕐 17.05.2026 10:32
──────────────────────

1️⃣ ASELS
Skor: 85
Fiyat: 123.45 TL
Sebep:
  - EMA trend güçlü
  - Hacim patlaması (x2.3)
  - RSI ideal (62.4)
  - Güçlü breakout (%1.8)

2️⃣ THYAO
Skor: 75
Fiyat: 245.80 TL
Sebep:
  - Güçlü breakout (%2.1)
  - MACD pozitif
  - RSI ideal (58.7)

──────────────────────
📊 BIST100 Tarama Raporu
🔍 Taranan: 100 hisse
🟠 EARLY: 5 | 🟢 BUY: 3 | 🔴 GEÇ: 1
❌ Hata: 2 | ⏱ Süre: 78.4s
🟢 Endeks: favorable
```

### API ile BIST100 Tarama

```bash
# BIST100 tam tarama
curl "http://localhost:8000/scan/bist100"

# BIST50 tarama
curl "http://localhost:8000/scan/bist50"

# BIST30 tarama
curl "http://localhost:8000/scan/bist30"
```

### Sembol Listesi

`app/data/bist100.py` üç liste tanımlar:

```python
from app.data.bist100 import BIST30_SYMBOLS   # 30 hisse
from app.data.bist100 import BIST50_SYMBOLS   # 50 hisse
from app.data.bist100 import BIST100_SYMBOLS  # 100 hisse
```

> BİST endeks bileşenleri her üç ayda bir güncellenir.
> Güncel listeyi [borsaistanbul.com](https://www.borsaistanbul.com/tr/sayfa/418/endeks-verileri)
> adresinden teyit edin ve `app/data/bist100.py`'yi güncelleyin.

---

## Kurulum

### Gereksinimler

- Python 3.11 veya üzeri
- İnternet bağlantısı (yfinance veri çekimi için)

### Adımlar

```bash
# 1. Depoyu klonlayın veya proje klasörüne gidin
cd bist-sinyal-robotu

# 2. Sanal ortam oluşturun
python -m venv venv

# 3. Sanal ortamı etkinleştirin
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 4. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 5. Ortam dosyasını oluşturun
cp .env.example .env
```

---

## .env Ayarları

`.env` dosyasını bir metin editörüyle açın ve değerleri doldurun:

```env
# ── Telegram ──────────────────────────────────────────────────────────
# @BotFather'dan /newbot komutuyla alınan token
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ

# Botun mesaj göndereceği sohbet veya kanal ID'si
# Kendi chat ID'nizi öğrenmek için @userinfobot'a /start yazın
TELEGRAM_CHAT_ID=-1001234567890

# ── Uygulama ──────────────────────────────────────────────────────────
APP_ENV=development
LOG_LEVEL=INFO

# ── Veritabanı ────────────────────────────────────────────────────────
DATABASE_URL=sqlite+aiosqlite:///./bist_robot.db

# ── Veri ──────────────────────────────────────────────────────────────
DATA_INTERVAL=1d
DATA_PERIOD=2y

# ── Sinyal ────────────────────────────────────────────────────────────
SIGNAL_CHECK_INTERVAL=300
MIN_SIGNAL_STRENGTH=0.6

# ── Risk ──────────────────────────────────────────────────────────────
MAX_POSITION_SIZE=0.10
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.10
MAX_OPEN_SIGNALS=5
```

Telegram için bilgileriniz yoksa sistem hata vermez; bildirimler atlanır.

---

## Çalıştırma

```bash
# Sanal ortamı etkinleştirin
source venv/bin/activate

# API'yi başlatın (geliştirme modu — otomatik yeniden yükleme)
uvicorn app.main:app --reload --port 8000

# Veya doğrudan Python ile
python -m app.main
```

Başarılı çıktı:

```
INFO  | BIST Sinyal Robotu v0.1.0 başlatıldı
INFO  | Uvicorn running on http://0.0.0.0:8000
```

| Adres | İçerik |
|-------|--------|
| `http://localhost:8000/docs` | Swagger UI — tüm endpoint'leri tarayıcıdan deneyin |
| `http://localhost:8000/redoc` | ReDoc API dokümantasyonu |
| `http://localhost:8000/health` | Hızlı sağlık kontrolü |

---

## API Endpoint'leri

### Temel

| Method | URL | Açıklama |
|--------|-----|----------|
| `GET` | `/health` | Versiyon, Telegram durumu, sembol sayısı |
| `GET` | `/symbols` | Aktif izleme listesi |
| `GET` | `/scan` | Tüm piyasa taraması — EARLY_WATCH + BUY + LATE_BREAKOUT sinyalleri |
| `GET` | `/signal/{symbol}` | Tek hisse anlık sinyal ve indikatörler |
| `GET` | `/backtest/{symbol}` | Geçmiş performans analizi |

### Sembol Yönetimi

| Method | URL | Açıklama |
|--------|-----|----------|
| `POST` | `/symbols/{symbol}` | Sembol ekle |
| `DELETE` | `/symbols/{symbol}` | Sembol çıkar |
| `POST` | `/symbols/reset` | Varsayılan listeye dön |
| `POST` | `/symbols/upload/csv` | CSV dosyasından toplu yükleme |
| `GET` | `/symbols/{symbol}/ohlcv` | Ham fiyat verisi |

### Sistem

| Method | URL | Açıklama |
|--------|-----|----------|
| `GET` | `/market-filter` | XU100 endeks filtresi durumu |
| `POST` | `/market-filter/invalidate` | Önbelleği temizle |
| `GET` | `/backtest` | Tüm semboller toplu backtest |
| `POST` | `/signals/trigger` | Arka planda sinyal tarama + DB kayıt |
| `GET` | `/signals/history` | Veritabanındaki geçmiş sinyaller |

---

## Örnek API İstekleri

### Sağlık kontrolü

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "telegram_configured": true,
  "symbol_count": 15
}
```

### İzleme listesini görüntüle

```bash
curl http://localhost:8000/symbols
```

### Tüm piyasayı tara

```bash
curl "http://localhost:8000/scan"
```

```json
{
  "scanned": 15,
  "buy_count": 2,
  "early_watch_count": 4,
  "late_breakout_count": 1,
  "results": [
    {
      "symbol": "ASELS.IS",
      "signal": "BUY",
      "price": 123.45,
      "reason": "Kırılım: 123.45 > direnç 121.00 | EMA20/50 trend yukarı | Hacim x2.3 | RSI14: 61.2",
      "risk_level": "MEDIUM",
      "strength": 0.72,
      "stop_loss": 119.75,
      "take_profit": 130.86,
      "market_filter": "favorable",
      "conditions_met": 5
    }
  ]
}
```

Sadece BUY sinyalleri:

```bash
curl "http://localhost:8000/scan?include_watch=false"
```

Belirli semboller:

```bash
curl "http://localhost:8000/scan?symbols=THYAO.IS&symbols=ASELS.IS"
```

### Tek hisse sinyali

```bash
curl "http://localhost:8000/signal/THYAO"
```

```json
{
  "symbol": "THYAO.IS",
  "signal": "HOLD",
  "price": 245.80,
  "reason": "Hacim oranı 1.42 < 1.8",
  "risk_level": "LOW",
  "strength": 0.0,
  "stop_loss": null,
  "take_profit": null,
  "conditions": {
    "c1_above_ema20": true,
    "c2_ema_uptrend": true,
    "c3_breakout": true,
    "c4_volume_surge": false,
    "c5_rsi_range": true
  },
  "market_filter": {
    "status": "favorable",
    "favorable": true
  },
  "indicators": {
    "ema_20": 242.10,
    "ema_50": 235.40,
    "rsi_14": 63.5,
    "atr_14": 4.20
  }
}
```

Endeks filtresi olmadan:

```bash
curl "http://localhost:8000/signal/ASELS?apply_filter=false"
```

### Sembol ekle / çıkar

```bash
# Ekle
curl -X POST http://localhost:8000/symbols/PGSUS

# Çıkar
curl -X DELETE http://localhost:8000/symbols/PGSUS

# Varsayılana dön
curl -X POST http://localhost:8000/symbols/reset
```

### CSV ile toplu sembol yükleme

```bash
# symbols.csv içeriği:
# symbol
# PGSUS
# TAVHL
# MAVI

curl -X POST "http://localhost:8000/symbols/upload/csv" \
     -F "file=@symbols.csv"

# Mevcut listeyi tamamen değiştirmek için:
curl -X POST "http://localhost:8000/symbols/upload/csv?replace=true" \
     -F "file=@symbols.csv"
```

---

## Backtest Nasıl Yapılır

Backtest, **Trend + Hacimli Kırılım** stratejisini geçmiş veriler üzerinde çalıştırır.

### Tek hisse

```bash
curl "http://localhost:8000/backtest/THYAO"
```

```json
{
  "symbol": "THYAO.IS",
  "strategy": "trend_breakout",
  "period": "2y",
  "initial_cash": 100000.0,
  "final_equity": 118450.0,
  "total_return_pct": 18.45,
  "buy_hold_return_pct": 42.10,
  "total_trades": 9,
  "win_rate_pct": 66.67,
  "max_drawdown_pct": -8.32,
  "best_trade_pct": 6.0,
  "worst_trade_pct": -3.0,
  "avg_trade_pct": 2.05,
  "sharpe_ratio": 1.24,
  "profit_factor": 2.31,
  "commission_pct": 0.1,
  "risk_per_trade_pct": 1.0
}
```

Farklı periyot ve sermaye:

```bash
curl "http://localhost:8000/backtest/ASELS?period=5y&cash=50000"
```

### Tüm sembollere toplu backtest

```bash
curl "http://localhost:8000/backtest"
```

```json
{
  "summary": {
    "total_symbols": 15,
    "successful": 14,
    "failed": 1,
    "avg_return_pct": 12.3,
    "best_symbol": "ASELS.IS",
    "worst_symbol": "PETKM.IS",
    "profitable_count": 10
  },
  "results": [ ... ]
}
```

### Backtest parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `period` | `2y` | Veri periyodu: `1y` `2y` `5y` `max` |
| `cash` | `100000` | Başlangıç sermayesi (TL) |
| `commission` | `0.001` | Komisyon oranı (%0.1) |

**Pozisyon büyüklüğü:** İşlem başına maksimum risk sermayenin **%1**'i,
stop-loss mesafesi **%3** → pozisyon büyüklüğü sermayenin **~%33**'ü.

---

## Telegram Nasıl Bağlanır

### Adım 1 — Bot oluşturun

1. Telegram'da **@BotFather**'a gidin
2. `/newbot` yazın ve yönergeleri izleyin
3. Size verilen token'ı kopyalayın:
   ```
   1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
   ```

### Adım 2 — Chat ID öğrenin

**Kişisel sohbet için:**
1. **@userinfobot**'a `/start` yazın — ID'nizi gösterir
2. Bota `/start` yazarak onu başlatın

**Grup veya kanal için:**
1. Botu gruba/kanala ekleyin
2. Kanala/gruba bir mesaj gönderin
3. Şu URL'yi ziyaret edin (tarayıcıda):
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
4. `chat.id` değerini bulun (genellikle `-100` ile başlar)

### Adım 3 — .env'ye ekleyin

```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_CHAT_ID=-1001234567890
```

### Adım 4 — Test edin

```bash
curl -X POST http://localhost:8000/market-filter/invalidate
curl "http://localhost:8000/scan"
```

Bir BUY sinyali varsa Telegram'a şu formatta mesaj gelir:

```
🚨 BIST SİNYALİ
──────────────────────
Hisse: ASELS
Sinyal: BUY 🟢
Fiyat: 123.45 TL
Güç: 72%
Sebep: EMA20/50 trend yukarı | Hacim x2.3 | RSI14: 61.2
Risk: MEDIUM
🛑 Stop Loss: 119.75 TL
🎯 Take Profit: 130.86 TL
──────────────────────
⚠️ Bu yatırım tavsiyesi değildir.
```

---

## Mimari

```
bist-sinyal-robotu/
├── app/
│   ├── main.py               # FastAPI — tüm endpoint'ler
│   ├── config.py             # .env ayarları (pydantic-settings)
│   ├── data/
│   │   ├── fetcher.py        # yfinance veri çekimi, FetchResult
│   │   └── symbols.py        # SymbolRegistry, CSV yükleme
│   ├── indicators/
│   │   └── technical.py      # add_indicators(), get_latest()
│   ├── strategies/
│   │   ├── base.py           # BaseStrategy, StrategySignal
│   │   ├── trend_breakout.py # Ana strateji (5 koşul)
│   │   ├── ema_crossover.py  # EMA20/50 golden cross
│   │   └── rsi_bb.py         # RSI + Bollinger
│   ├── backtest/
│   │   ├── engine.py         # EMACrossoverBT (eski)
│   │   └── runner.py         # TrendBreakoutBT, run_single, run_multiple
│   ├── signals/
│   │   ├── generator.py      # SignalGenerator (DB kayıt)
│   │   └── scanner.py        # scan_market() — EARLY_WATCH + BUY + LATE_BREAKOUT
│   ├── risk/
│   │   ├── manager.py        # RiskManager — R/R, pozisyon büyüklüğü
│   │   └── market_filter.py  # XU100 endeks filtresi, TTL cache
│   ├── notifications/
│   │   └── telegram.py       # send_telegram_message(), format_signal_message()
│   ├── database/
│   │   ├── models.py         # Signal, BacktestResult, PriceCache
│   │   └── crud.py           # Async SQLAlchemy CRUD
│   └── utils/
│       └── helpers.py        # Logger, yardımcı fonksiyonlar
├── tests/                    # 300+ birim test (ağ bağlantısı gerektirmez)
├── .env.example
├── requirements.txt
└── README.md
```

### Sinyal üretim akışı

```
yfinance
   ↓
fetch_symbol_data()          # ham OHLCV
   ↓
add_indicators()             # EMA, RSI, MACD, ATR, BB, direnç, squeeze
   ↓
generate_signal()            # trend_breakout — 5 koşul değerlendirme
   ↓
is_market_favorable()        # XU100 > EMA50?
   ↓
BUY → risk_manager.assess()  # R/R, güç filtresi
   ↓
send_telegram_message()      # 🚨 Bildirim
```

---

## Testler

```bash
# Tüm testler
pytest tests/ -v

# Belirli modül
pytest tests/test_trend_breakout.py -v
pytest tests/test_api.py -v
pytest tests/test_backtest_runner.py -v

# Kısa çıktı
pytest tests/ -q
```

Testler ağ bağlantısı gerektirmez — yfinance mock'lanır.

---

## Yeni Strateji Eklemek

1. `app/strategies/` altında yeni dosya oluşturun:

```python
# app/strategies/macd_cross.py
from app.strategies.base import BaseStrategy, StrategySignal, SignalType
import pandas as pd
from typing import Optional

class MACDCrossStrategy(BaseStrategy):
    name = "macd_cross"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        required = ["macd", "macd_signal", "atr_14", "close"]
        if not self._validate_df(df, required):
            return None

        curr = df.dropna(subset=required).iloc[-1]
        prev = df.dropna(subset=required).iloc[-2]

        # MACD, sinyal çizgisini yukarı kesiyor mu?
        cross_up = prev["macd"] <= prev["macd_signal"] and curr["macd"] > curr["macd_signal"]
        if not cross_up:
            return None

        close = curr["close"]
        atr   = curr["atr_14"]
        return StrategySignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            strategy=self.name,
            strength=0.65,
            entry_price=close,
            stop_loss=round(close - 1.5 * atr, 2),
            take_profit=round(close + 2.5 * atr, 2),
            notes=f"MACD yukarı kesişim",
        )
```

2. `app/strategies/__init__.py`'deki `STRATEGIES` listesine ekleyin:

```python
from app.strategies.macd_cross import MACDCrossStrategy

STRATEGIES = [
    EMACrossoverStrategy(),
    RSIBollingerStrategy(),
    TrendBreakoutStrategy(),
    MACDCrossStrategy(),   # ← yeni
]
```

---

## Risk Uyarıları

Sistemi kullanmadan önce aşağıdakileri mutlaka okuyun:

**Genel Uyarılar**
- Bu sistem **yatırım tavsiyesi, alım-satım önerisi veya kâr garantisi değildir.**
- Geçmiş performans gelecekteki sonuçları garanti etmez.
- Teknik analiz her zaman doğru sinyal üretmez; yanlış sinyaller (false positive) olabilir.

**Teknik Sınırlamalar**
- Veriler yfinance üzerinden çekilir; veri gecikmesi veya eksikliği olabilir.
- Backtest sonuçları gerçek piyasa koşullarından farklılık gösterebilir (slipaj, likidite vb.).
- XU100 filtresi yalnızca kapanış fiyatı bazlıdır; gün içi hareketleri yansıtmaz.
- Sistem sadece günlük (1d) veride test edilmiştir.

**Kullanım Uyarıları**
- Sistemi **canlı para** ile kullanmadan önce kağıt trading (simüle işlem) yapın.
- Risk yönetimi parametrelerini (`STOP_LOSS_PCT`, `MAX_POSITION_SIZE`) kendi risk toleransınıza göre ayarlayın.
- İşlem başına maksimum risk önerisi: **%1-2** (varsayılan %1).
- Gerçek emir göndermek için bu sisteme herhangi bir broker API'si **bağlı değildir** ve **bağlanmamalıdır** — sistem yalnızca sinyal üretmek için tasarlanmıştır.

**Yasal**
- Bu yazılım MIT lisansı altında "olduğu gibi" sunulmaktadır.
- Yazılımın kullanımından doğacak herhangi bir mali kayıptan geliştirici sorumlu tutulamaz.
