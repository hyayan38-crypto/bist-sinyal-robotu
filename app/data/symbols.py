import csv
from pathlib import Path
from typing import List
from loguru import logger


DEFAULT_SYMBOLS: List[str] = [
    # Savunma & Teknoloji
    "ASELS.IS",  # Aselsan
    # Havacılık
    "THYAO.IS",  # Türk Hava Yolları
    # Holding
    "KCHOL.IS",  # Koç Holding
    "SAHOL.IS",  # Sabancı Holding
    # Cam & Kimya
    "SISE.IS",   # Şişe Cam
    "PETKM.IS",  # Petkim
    # Demir & Çelik
    "EREGL.IS",  # Ereğli Demir Çelik
    # Enerji & Rafineri
    "TUPRS.IS",  # Tüpraş
    # Bankacılık
    "YKBNK.IS",  # Yapı Kredi Bankası
    "AKBNK.IS",  # Akbank
    "GARAN.IS",  # Garanti BBVA
    # Perakende
    "BIMAS.IS",  # BİM Birleşik Mağazalar
    # Otomotiv
    "FROTO.IS",  # Ford Otosan
    "TOASO.IS",  # Tofaş Oto
    # Madencilik
    "KOZAL.IS",  # Koza Altın
]


def normalize(symbol: str) -> str:
    """Sembolü büyük harfe çevirir, .IS uzantısı yoksa ekler."""
    symbol = symbol.strip().upper()
    if not symbol.endswith(".IS"):
        symbol = f"{symbol}.IS"
    return symbol


def load_from_csv(path: str | Path) -> List[str]:
    """
    CSV dosyasından sembol listesi yükler.

    Desteklenen formatlar:
      - Tek sütun: her satır bir sembol
      - Çok sütun: 'symbol' başlıklı sütun kullanılır, yoksa ilk sütun
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sembol dosyası bulunamadı: {path}")

    symbols: List[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return []

    # Başlık tespiti: ilk satır bilinen başlık kelimelerinden biriyse atla
    _HEADER_WORDS = {"symbol", "symbols", "ticker", "tickers", "hisse", "kod"}
    first = lines[0].split(",")[0].strip().lower()
    has_header = first in _HEADER_WORDS

    if has_header:
        lines = lines[1:]

    for line in lines:
        # Virgüllü satırda ilk sütunu al
        value = line.split(",")[0].strip()
        if value:
            symbols.append(normalize(value))

    symbols = list(dict.fromkeys(symbols))  # tekrarları koru sırayı bozmadan
    logger.info(f"{path.name} dosyasından {len(symbols)} sembol yüklendi")
    return symbols


class SymbolRegistry:
    """Aktif sembol listesini yönetir. Çalışma zamanında güncellenebilir."""

    def __init__(self, symbols: List[str] | None = None):
        self._symbols: List[str] = [normalize(s) for s in (symbols or DEFAULT_SYMBOLS)]

    # ── okuma ─────────────────────────────────────────────────────────────

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    def __len__(self) -> int:
        return len(self._symbols)

    def __contains__(self, symbol: str) -> bool:
        return normalize(symbol) in self._symbols

    # ── değiştirme ─────────────────────────────────────────────────────────

    def add(self, symbol: str) -> bool:
        sym = normalize(symbol)
        if sym in self._symbols:
            logger.warning(f"{sym} zaten listede")
            return False
        self._symbols.append(sym)
        logger.info(f"{sym} listeye eklendi")
        return True

    def remove(self, symbol: str) -> bool:
        sym = normalize(symbol)
        if sym not in self._symbols:
            logger.warning(f"{sym} listede bulunamadı")
            return False
        self._symbols.remove(sym)
        logger.info(f"{sym} listeden çıkarıldı")
        return True

    def load_csv(self, path: str | Path) -> List[str]:
        """CSV'den yüklenen sembolleri mevcut listeye ekler, tekrarları atlar."""
        loaded = load_from_csv(path)
        added = [s for s in loaded if self.add(s)]
        logger.info(f"CSV'den {len(added)} yeni sembol eklendi")
        return added

    def replace_from_csv(self, path: str | Path) -> List[str]:
        """Mevcut listeyi CSV içeriğiyle tamamen değiştirir."""
        self._symbols = load_from_csv(path)
        return self._symbols

    def reset(self):
        """Varsayılan listeye döner."""
        self._symbols = [normalize(s) for s in DEFAULT_SYMBOLS]
        logger.info("Sembol listesi varsayılana sıfırlandı")

    # ── yardımcı ───────────────────────────────────────────────────────────

    def save_csv(self, path: str | Path):
        """Aktif listeyi CSV'ye kaydeder."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol"])
            writer.writerows([[s] for s in self._symbols])
        logger.info(f"{len(self._symbols)} sembol {path} dosyasına kaydedildi")


registry = SymbolRegistry()
