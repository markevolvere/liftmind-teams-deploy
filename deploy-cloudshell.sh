#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy-cloudshell.sh — Quick deploy from Azure Cloud Shell
# Usage: cd ~/bot-build && bash deploy-cloudshell.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RG="MjeanesResourceGroup"
APP="liftshop-teams-bot"
ACR="liftshopbotacr"
IMAGE="${ACR}.azurecr.io/${APP}:latest"

echo "═══════════════════════════════════════════════════════════════"
echo " LiftShop Teams Bot — Cloud Shell Deploy"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Build container image via ACR ─────────────────────────────────────
echo ""
echo "[1/4] Building container image in ACR (this takes 2-3 minutes)..."
az acr build --registry "$ACR" --image "${APP}:latest" .

# ── Step 2: Add new environment variables ─────────────────────────────────────
echo ""
echo "[2/4] Setting environment variables..."
az webapp config appsettings set \
    --name "$APP" \
    --resource-group "$RG" \
    --settings SQL_CONNECT_HOST=127.0.0.1 \
    --output none
echo "  ✓ SQL_CONNECT_HOST=127.0.0.1"

# ── Step 3: Force App Service to pull new image ──────────────────────────────
echo ""
echo "[3/4] Updating container image reference..."
az webapp config container set \
    --name "$APP" \
    --resource-group "$RG" \
    --container-image-name "$IMAGE" \
    --output none

# ── Step 4: Clean restart ────────────────────────────────────────────────────
echo ""
echo "[4/4] Restarting App Service (clean stop + start)..."
az webapp stop --name "$APP" --resource-group "$RG"
sleep 3
az webapp start --name "$APP" --resource-group "$RG"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Deploy complete!"
echo " Wait ~60s then check: https://${APP}.azurewebsites.net/health"
echo ""
echo " Stream logs: az webapp log tail --name $APP --resource-group $RG"
echo "═══════════════════════════════════════════════════════════════"
