#!/usr/bin/env bash
set -e

echo "[INFO] Adding CRAN GPG key..."
sudo mkdir -p /etc/apt/keyrings
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
    sudo tee /etc/apt/keyrings/cran.gpg > /dev/null

echo "[INFO] Adding CRAN repo for Ubuntu noble (24.04)..."
echo "deb [signed-by=/etc/apt/keyrings/cran.gpg] https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" | \
    sudo tee /etc/apt/sources.list.d/cran-r.list

echo "[INFO] Updating package lists..."
sudo apt update

echo "[INFO] Installing latest R (>=4.4) and development tools..."
sudo apt install --no-install-recommends -y r-base r-base-dev

echo "[INFO] Removing old R 4.3 user libraries to avoid conflicts..."
rm -rf ~/R/x86_64-pc-linux-gnu-library/4.3 || true

echo "[INFO] Upgrade complete. Installed R version:"
R --version

echo
echo "[NEXT STEP] Start R and reinstall your packages:"
echo '    install.packages(c("tram", "MASS", "readr"))'
