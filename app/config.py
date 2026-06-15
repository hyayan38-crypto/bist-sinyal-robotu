from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

# Projenin kök dizini — config.py'nin konumundan iki üst (app/ → proje kökü)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    # Uygulama
    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change_this_secret_key"

    # Veritabanı
    database_url: str = "sqlite+aiosqlite:///./bist_robot.db"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Veri
    data_interval: str = "1d"
    data_period: str = "2y"

    # Sinyal
    signal_check_interval: int = 300  # saniye
    min_signal_strength: float = 0.6

    # Risk
    max_position_size: float = 0.10
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_open_signals: int = 5

    # Fiyat önbelleği (SQLite) — paralel taramada yfinance yükünü azaltır
    price_cache_enabled: bool = True

    # Sinyal takibi
    signal_expiry_days: int = 10       # bu süre sonunda açık sinyal EXPIRED olur
    performance_window_days: int = 30  # isabet raporu pencere genişliği

    # Tarama akışı risk kalite kapısı — min risk/ödül oranı (R/R)
    # Zamanlanmış tarama risk/manager.py'den geçmediği için bu eşik scanner'da
    # uygulanır; altında kalan BUY/LATE sinyalleri yayınlanmaz.
    min_risk_reward: float = 1.5

    model_config = SettingsConfigDict(
        # Mutlak yol — uvicorn hangi dizinden çalıştırılırsa çalıştırılsın .env bulunur
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        # .env yoksa sessizce devam et (ortam değişkenleri yine de okunur)
        extra="ignore",
    )


settings = Settings()
