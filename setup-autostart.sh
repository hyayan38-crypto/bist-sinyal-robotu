#!/bin/bash
# BIST Sinyal Robotu — Otomatik Başlatma Kurulum Scripti
# WSL terminalinde çalıştırın: bash setup-autostart.sh

set -e

SERVICE_FILE="/home/hyayan/bist-sinyal-robotu/bist-robot.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "==> Servis dosyası kopyalanıyor..."
sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/bist-robot.service"

echo "==> Systemd yeniden yükleniyor..."
sudo systemctl daemon-reload

echo "==> Servis etkinleştiriliyor (açılışta otomatik başlasın)..."
sudo systemctl enable bist-robot.service

echo "==> Servis başlatılıyor..."
sudo systemctl start bist-robot.service

echo ""
echo "✅ Kurulum tamamlandı!"
echo ""
echo "Durum kontrolü için:"
echo "  sudo systemctl status bist-robot"
echo ""
echo "Log görüntüleme için:"
echo "  sudo journalctl -u bist-robot -f"
