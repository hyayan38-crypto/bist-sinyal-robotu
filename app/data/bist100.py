"""
BIST Endeks Sembol Listeleri
============================
BIST30, BIST50 ve BIST100 endekslerine dahil hisselerin .IS formatındaki listesi.

NOT: BİST endeks bileşenleri her üç ayda bir güncellenir.
Güncel listeyi https://www.borsaistanbul.com/tr/sayfa/418/endeks-verileri
adresinden teyit edin.
"""

# ── BIST30 (XU030) ────────────────────────────────────────────────────────────

BIST30_SYMBOLS: list[str] = [
    "AKBNK.IS",  # Akbank
    "ARCLK.IS",  # Arçelik
    "ASELS.IS",  # Aselsan
    "BIMAS.IS",  # BİM Birleşik Mağazalar
    "EKGYO.IS",  # Emlak Konut GYO
    "ENKAI.IS",  # Enka İnşaat
    "EREGL.IS",  # Ereğli Demir Çelik
    "FROTO.IS",  # Ford Otosan
    "GARAN.IS",  # Garanti BBVA
    "HEKTS.IS",  # Hektaş Ticaret
    "ISCTR.IS",  # İş Bankası (C)
    "KCHOL.IS",  # Koç Holding
    "BRSAN.IS",  # Borusan Boru (KOZAL yerine — yfinance veri vermiyor)
    "AKSEN.IS",  # Aksa Enerji (KOZAA yerine — yfinance veri vermiyor)
    "KRDMD.IS",  # Kardemir (D)
    "PETKM.IS",  # Petkim Petrokimya
    "PGSUS.IS",  # Pegasus Hava Yolları
    "SAHOL.IS",  # Sabancı Holding
    "SASA.IS",   # SASA Polyester
    "SISE.IS",   # Şişe Cam
    "TAVHL.IS",  # TAV Havalimanları
    "TCELL.IS",  # Turkcell
    "THYAO.IS",  # Türk Hava Yolları
    "TKFEN.IS",  # Tekfen Holding
    "TOASO.IS",  # Tofaş Otomobil Fabrikaları
    "TTKOM.IS",  # Türk Telekom
    "TUPRS.IS",  # Tüpraş
    "VAKBN.IS",  # Vakıfbank
    "YKBNK.IS",  # Yapı Kredi Bankası
    "ZOREN.IS",  # Zorlu Enerji
]

# ── BIST50 (XU050) — BIST30 + 20 ─────────────────────────────────────────────

_BIST50_EXTRA: list[str] = [
    "AKSA.IS",   # Aksa Akrilik Kimya
    "AYGAZ.IS",  # Aygaz
    "BRISA.IS",  # Brisa Bridgestone Sabancı
    "CCOLA.IS",  # Coca-Cola İçecek
    "CIMSA.IS",  # Çimsa Çimento
    "DOHOL.IS",  # Doğan Holding
    "EGEEN.IS",  # Ege Endüstri
    "ENJSA.IS",  # Enerjisa Enerji
    "GUBRF.IS",  # Gübre Fabrikaları
    "HALKB.IS",  # Halkbank
    "INDES.IS",  # İndeks Bilgisayar
    "LOGO.IS",   # Logo Yazılım
    "MAVI.IS",   # Mavi Giyim
    "MGROS.IS",  # Migros Ticaret
    "NUHCM.IS",  # Nuh Çimento
    "OTKAR.IS",  # Otokar
    "SKBNK.IS",  # Şekerbank
    "SOKM.IS",   # Şok Marketler
    "TSKB.IS",   # Türkiye Sınai Kalkınma Bankası
    "AKENR.IS",  # Akenerji
]

BIST50_SYMBOLS: list[str] = BIST30_SYMBOLS + _BIST50_EXTRA

# ── BIST100 (XU100) — BIST50 + 50 ────────────────────────────────────────────

_BIST100_EXTRA: list[str] = [
    "AEFES.IS",  # Anadolu Efes Biracılık
    "AGHOL.IS",  # AG Anadolu Grubu Holding
    "ALARK.IS",  # Alarko Holding
    "ALBRK.IS",  # Albaraka Türk Katılım Bankası
    "KONTR.IS",  # Kontrolmatik Teknoloji (ANACM yerine — Şişecam çatısına geçti)
    "ANSGR.IS",  # Anadolu Sigorta
    "ARSAN.IS",  # Arsan Tekstil Ticaret
    "ASTOR.IS",  # Astor Enerji
    "AVGYO.IS",  # Avrasya GYO
    "AYEN.IS",   # Ayen Enerji
    "BANVT.IS",  # Banvit Bandırma Vitaminli
    "BSOKE.IS",  # Batısöke Söke Çimento
    "BUCIM.IS",  # Bursa Çimento
    "BURCE.IS",  # Burçelik Vana
    "CONSE.IS",  # Consus Enerji
    "CRFSA.IS",  # CarrefourSA
    "DARDL.IS",  # Dardanel Önentaş Gıda
    "DOAS.IS",   # Doğuş Otomotiv
    "DYOBY.IS",  # DYO Boya Fabrikaları
    "ECILC.IS",  # Eczacıbaşı İlaç
    "FENER.IS",  # Fenerbahçe Futbol
    "GLYHO.IS",  # Global Yatırım Holding
    "GOODY.IS",  # Goodyear Lastikleri
    "HURGZ.IS",  # Hürriyet Gazetecilik
    "SMRTG.IS",  # Smart Güneş Enerjisi (IPEKE yerine — yfinance veri vermiyor)
    "ISDMR.IS",  # İskenderun Demir Çelik
    "ISGYO.IS",  # İş GYO
    "ISYAT.IS",  # İş Yatırım Menkul Değerler
    "KAREL.IS",  # Karel Elektronik
    "KARSN.IS",  # Karsan Otomotiv
    "KATMR.IS",  # Katmerciler
    "KENT.IS",   # Kent Gıda Maddeleri
    "MIATK.IS",  # Mia Teknoloji (KERVT yerine — kottan çıktı)
    "NETAS.IS",  # Netaş Telekomünikasyon
    "CWENE.IS",  # CW Enerji (QNBFL yerine — kottan çıktı)
    "SELEC.IS",  # Selçuk Ecza Deposu
    "ULKER.IS",  # Ülker Bisküvi
    "VESTL.IS",  # Vestel Elektronik
    "GOLTS.IS",  # Göltaş Çimento
    "PRKME.IS",  # Park Elektrik Madencilik
    "ORGE.IS",   # Orge Enerji Elektrik
    "YEOTK.IS",  # Yeo Teknoloji Enerji (DENTA yerine — kottan çıktı)
    "MPARK.IS",  # MLP Sağlık Hizmetleri
    "CLEBI.IS",  # Çelebi Hava Servisi
    "TURSG.IS",  # Türkiye Sigorta
    "EUPWR.IS",  # Europower Enerji
    "AKFGY.IS",  # Akfen GYO
    "GESAN.IS",  # Gesan Enerji
    "ALFAS.IS",  # Alfa Solar Enerji (MIPAZ yerine — kottan çıktı)
    "GRSEL.IS",  # GR Sigorta
]

BIST100_SYMBOLS: list[str] = BIST50_SYMBOLS + _BIST100_EXTRA
