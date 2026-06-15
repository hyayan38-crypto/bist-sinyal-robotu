"""
Sinyal Grafiği Görseli
======================
Son ~60 barlık mum grafiğini EMA20/50, direnç ve stop/hedef seviyeleri işaretli
olarak PNG byte olarak üretir. Hem Telegram'a gönderim hem de Claude vision
teyidi için aynı görsel kullanılır.

`matplotlib` başsız (Agg) modda; içe aktarım sırasında backend ayarlanır.
mplfinance kurulu değilse `render_signal_chart` None döner (fail-open).
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")  # başsız — sunucu/servis ortamı
    import mplfinance as mpf
    _MPF_AVAILABLE = True
except Exception as exc:  # ImportError veya backend hatası
    _MPF_AVAILABLE = False
    logger.warning(f"mplfinance kullanılamıyor, grafik üretimi devre dışı: {exc}")

_CHART_BARS = 60


def _to_mpf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Küçük harf OHLCV → mplfinance büyük harf format, DatetimeIndex."""
    mpf_df = df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low",  "close": "Close", "volume": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]].copy()

    if hasattr(mpf_df.index, "tz") and mpf_df.index.tz is not None:
        mpf_df.index = mpf_df.index.tz_convert("UTC").tz_localize(None)
    mpf_df.index = pd.to_datetime(mpf_df.index)
    return mpf_df


def render_signal_chart(
    symbol: str,
    df: pd.DataFrame,
    atr_sl: Optional[float] = None,
    atr_tp: Optional[float] = None,
    struct_sl: Optional[float] = None,
    struct_tp: Optional[float] = None,
    bars: int = _CHART_BARS,
) -> Optional[bytes]:
    """
    İşaretli mum grafiğini PNG byte döner. Üretilemezse None (fail-open).

    Stop seviyeleri kırmızı, hedefler yeşil yatay çizgi; ATR kesik, yapı düz.
    EMA20/50 ve resistance_20 mevcutsa ek panel çizgileri olarak eklenir.
    """
    if not _MPF_AVAILABLE:
        return None
    try:
        plot_df = _to_mpf_df(df).tail(bars)
        if plot_df.empty:
            return None

        addplots = []
        for col, color in (("ema_20", "#1f77b4"), ("ema_50", "#ff7f0e")):
            if col in df.columns:
                series = df[col].rename(None)
                series.index = pd.to_datetime(
                    df.index.tz_convert("UTC").tz_localize(None)
                    if getattr(df.index, "tz", None) is not None else df.index
                )
                addplots.append(mpf.make_addplot(series.tail(bars), color=color, width=0.8))

        hlines_vals:   list[float] = []
        hlines_colors: list[str]   = []
        hlines_styles: list[str]   = []
        for val, color, style in (
            (atr_sl,    "#d62728", "--"),   # ATR stop  — kırmızı kesik
            (struct_sl, "#d62728", "-"),    # yapı stop  — kırmızı düz
            (atr_tp,    "#2ca02c", "--"),   # ATR hedef — yeşil kesik
            (struct_tp, "#2ca02c", "-"),    # yapı hedef — yeşil düz
        ):
            if val is not None and val == val:  # NaN değil
                hlines_vals.append(float(val))
                hlines_colors.append(color)
                hlines_styles.append(style)

        title = symbol.removesuffix(".IS")
        buf = io.BytesIO()
        kwargs = dict(
            type="candle",
            style="yahoo",
            title=title,
            ylabel="Fiyat (TL)",
            volume=True,
            figsize=(10, 6),
            tight_layout=True,
            savefig=dict(fname=buf, format="png", dpi=110),
        )
        # mplfinance None'ı reddeder — yalnız doluysa ekle.
        if addplots:
            kwargs["addplot"] = addplots
        if hlines_vals:
            kwargs["hlines"] = dict(
                hlines=hlines_vals, colors=hlines_colors,
                linestyle=hlines_styles, linewidths=1.0,
            )

        mpf.plot(plot_df, **kwargs)
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        logger.warning(f"{symbol} grafik üretimi başarısız: {exc}")
        return None
