import csv
import pytest
from pathlib import Path
from app.data.symbols import SymbolRegistry, DEFAULT_SYMBOLS, normalize, load_from_csv


class TestNormalize:
    def test_adds_is_suffix(self):
        assert normalize("thyao") == "THYAO.IS"

    def test_uppercase(self):
        assert normalize("garan.is") == "GARAN.IS"

    def test_no_double_suffix(self):
        assert normalize("AKBNK.IS") == "AKBNK.IS"

    def test_strips_whitespace(self):
        assert normalize("  SISE  ") == "SISE.IS"


class TestSymbolRegistry:
    def setup_method(self):
        self.reg = SymbolRegistry(["THYAO.IS", "GARAN.IS"])

    def test_initial_symbols(self):
        assert len(self.reg) == 2
        assert "THYAO.IS" in self.reg

    def test_add_new(self):
        assert self.reg.add("AKBNK") is True
        assert "AKBNK.IS" in self.reg
        assert len(self.reg) == 3

    def test_add_duplicate(self):
        assert self.reg.add("THYAO.IS") is False
        assert len(self.reg) == 2

    def test_remove_existing(self):
        assert self.reg.remove("GARAN") is True
        assert "GARAN.IS" not in self.reg

    def test_remove_missing(self):
        assert self.reg.remove("XXXXXX") is False

    def test_reset(self):
        self.reg.add("EREGL")
        self.reg.reset()
        assert self.reg.symbols == [normalize(s) for s in DEFAULT_SYMBOLS]

    def test_default_symbols_count(self):
        reg = SymbolRegistry()
        assert len(reg) == len(DEFAULT_SYMBOLS)


class TestLoadFromCSV:
    def test_single_column_no_header(self, tmp_path):
        f = tmp_path / "symbols.csv"
        f.write_text("thyao\ngaran\nakbnk\n")
        result = load_from_csv(f)
        assert result == ["THYAO.IS", "GARAN.IS", "AKBNK.IS"]

    def test_with_symbol_header(self, tmp_path):
        f = tmp_path / "symbols.csv"
        f.write_text("symbol\nEREGL\nTUPRS\n")
        result = load_from_csv(f)
        assert "TUPRS.IS" in result
        assert "EREGL.IS" in result

    def test_deduplication(self, tmp_path):
        f = tmp_path / "symbols.csv"
        f.write_text("THYAO\nthyao\nTHYAO.IS\n")
        result = load_from_csv(f)
        assert result.count("THYAO.IS") == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_from_csv("/nonexistent/path.csv")

    def test_registry_load_csv(self, tmp_path):
        f = tmp_path / "extra.csv"
        f.write_text("BIMAS\nFROTO\n")
        reg = SymbolRegistry(["THYAO.IS"])
        added = reg.load_csv(f)
        assert "BIMAS.IS" in added
        assert len(reg) == 3

    def test_registry_replace_from_csv(self, tmp_path):
        f = tmp_path / "new.csv"
        f.write_text("symbol\nKOZAL\nPETKM\n")
        reg = SymbolRegistry(["THYAO.IS", "GARAN.IS"])
        reg.replace_from_csv(f)
        assert len(reg) == 2
        assert "THYAO.IS" not in reg

    def test_save_and_reload(self, tmp_path):
        reg = SymbolRegistry(["THYAO.IS", "AKBNK.IS"])
        out = tmp_path / "out.csv"
        reg.save_csv(out)
        reloaded = load_from_csv(out)
        assert reloaded == ["THYAO.IS", "AKBNK.IS"]
