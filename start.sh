#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Establish SSH tunnel to Azure SQL, then start the bot
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SSH_HOST="${SSH_TUNNEL_HOST:-20.70.160.245}"
SSH_USER="${SSH_TUNNEL_USER:-liftadmin}"
SSH_KEY_DATA="${SSH_BRIDGE_KEY:-}"
SSH_PASS="${SSH_BRIDGE_PASSWORD:-}"
SQL_HOST="${SQL_TUNNEL_TARGET:-lssql04.database.windows.net}"
LOCAL_PORT="${SQL_TUNNEL_LOCAL_PORT:-1433}"

echo "[start.sh] Bot starting up at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Skip tunnel if no SSH credentials configured ─────────────────────────────
if [[ -z "$SSH_KEY_DATA" && -z "$SSH_PASS" ]]; then
    echo "[start.sh] No SSH credentials set — running without tunnel"
    exec python app.py
fi

# ── Write SSH private key to file ────────────────────────────────────────────
SSH_KEY_FILE="$HOME/.ssh/tunnel_key"
mkdir -p "$HOME/.ssh"

if [[ -n "$SSH_KEY_DATA" ]]; then
    # The key is stored with literal \n — convert to real newlines
    echo -e "$SSH_KEY_DATA" | sed 's/ *$//' > "$SSH_KEY_FILE"
    chmod 600 "$SSH_KEY_FILE"
    echo "[start.sh] SSH private key written to $SSH_KEY_FILE"
    AUTH_METHOD="key"
else
    AUTH_METHOD="password"
fi

# ── SSH config for unattended use ────────────────────────────────────────────
cat > "$HOME/.ssh/config" <<EOF
Host tunnel-target
    HostName ${SSH_HOST}
    User ${SSH_USER}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 30
    ServerAliveCountMax 3
    LogLevel ERROR
EOF

if [[ "$AUTH_METHOD" == "key" ]]; then
    echo "    IdentityFile ${SSH_KEY_FILE}" >> "$HOME/.ssh/config"
fi

chmod 600 "$HOME/.ssh/config"

# ── Start SSH tunnel in background ───────────────────────────────────────────
echo "[start.sh] Establishing SSH tunnel: localhost:${LOCAL_PORT} -> ${SQL_HOST}:1433 via ${SSH_USER}@${SSH_HOST} (${AUTH_METHOD})"

if [[ "$AUTH_METHOD" == "key" ]]; then
    ssh -f -N \
        -L "${LOCAL_PORT}:${SQL_HOST}:1433" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -i "$SSH_KEY_FILE" \
        "${SSH_USER}@${SSH_HOST}" 2>&1 || {
        echo "[start.sh] Key auth failed, trying password fallback..."
        AUTH_METHOD="password"
    }
fi

if [[ "$AUTH_METHOD" == "password" && -n "$SSH_PASS" ]]; then
    sshpass -p "${SSH_PASS}" ssh -f -N \
        -L "${LOCAL_PORT}:${SQL_HOST}:1433" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        "${SSH_USER}@${SSH_HOST}" 2>&1
fi

echo "[start.sh] SSH tunnel command issued"

# ── Wait for tunnel to be ready ──────────────────────────────────────────────
MAX_WAIT=20
READY=0
for i in $(seq 1 $MAX_WAIT); do
    if python3 -c "
import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('127.0.0.1', ${LOCAL_PORT}))
    s.close()
    print('port open')
except:
    exit(1)
" 2>/dev/null; then
        echo "[start.sh] Tunnel is ready (port ${LOCAL_PORT} open) after ${i}s"
        READY=1
        break
    fi
    sleep 1
done

if [[ $READY -eq 0 ]]; then
    echo "[start.sh] WARNING: Tunnel port check timed out after ${MAX_WAIT}s — starting bot anyway"
fi

# ── Write FreeTDS config to /tmp (writable by non-root) ─────────────────────
cat > /tmp/freetds.conf <<FREETDS
[global]
tds version = 7.4
client charset = UTF-8

[lssql04]
host = 127.0.0.1
port = ${LOCAL_PORT}
tds version = 7.4
FREETDS

export FREETDSCONF=/tmp/freetds.conf
echo "[start.sh] FreeTDS config written to /tmp/freetds.conf"

# ── Start the bot ────────────────────────────────────────────────────────────
echo "[start.sh] Starting bot application..."
exec python app.py
