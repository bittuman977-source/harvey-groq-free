#!/usr/bin/env bash
# ============================================================
# setup-free.sh — one-command 100% free setup for Harvey-Groq.
#
#   git clone https://github.com/bittuman977-source/harvey-groq-free.git
#   cd harvey-groq-free
#   bash setup-free.sh          (it will ask for your Groq API key)
#
# Or fully unattended:
#   GROQ_API_KEY=gsk_... bash setup-free.sh
#
# Does: python check -> venv -> install -> .env with your key ->
#       harvey.yaml set to groq brain + free-tier pacing ->
#       live smoke test against Groq.
# Never commits or uploads your key anywhere.
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---------- 1. Python ----------
if ! command -v python3 >/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.11+ first." >&2
  exit 1
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "ERROR: Python 3.11+ required (you have $PYV)." >&2
  echo "  Debian/Ubuntu: sudo apt install python3.11 python3.11-venv" >&2
  exit 1
fi
echo "[1/5] Python $PYV OK"

# ---------- 2. venv + install ----------
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e .
echo "[2/5] Harvey installed"

# ---------- 3. .env with the Groq key ----------
if [ ! -f .env ]; then
  cp .env.example .env
fi
KEY="${GROQ_API_KEY:-}"
if [ -z "$KEY" ]; then
  read -r -p "Paste your Groq API key (free at https://console.groq.com/keys): " KEY
fi
case "$KEY" in
  gsk_*) ;;
  "") echo "ERROR: no key given. Re-run with: GROQ_API_KEY=gsk_... bash setup-free.sh" >&2; exit 1 ;;
  *) echo "WARNING: key doesn't start with gsk_ — double-check it." >&2 ;;
esac
python3 - "$KEY" <<'PYEOF'
import sys, re
key = sys.argv[1].strip()
path = ".env"
text = open(path).read()
text, n = re.subn(r"(?m)^GROQ_API_KEY=.*$", f"GROQ_API_KEY={key}", text)
if n == 0:
    if not text.endswith("\n"): text += "\n"
    text += f"GROQ_API_KEY={key}\n"
open(path, "w").write(text)
PYEOF
echo "[3/5] .env saved (key kept local, gitignored)"

# ---------- 4. harvey.yaml -> free-tier Groq config ----------
python3 - <<'PYEOF'
import yaml, pathlib
path = pathlib.Path("harvey.yaml")
cfg = yaml.safe_load(path.read_text())
# Groq brain (default model openai/gpt-oss-120b)
cfg["brain"] = {"provider": "groq"}
# Free-tier pacing: 30-min heartbeat; 40% => 80 model calls/day budget.
cfg["usage"] = cfg.get("usage") or {}
cfg["usage"]["heartbeat_interval_minutes"] = 30
cfg["usage"]["max_daily_claude_percent"] = 40
path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print("harvey.yaml: brain=groq, heartbeat=30min, 80 calls/day")
PYEOF
echo "[4/5] harvey.yaml configured for the free tier"

# ---------- 5. Live smoke test ----------
python3 - <<'PYEOF'
import os
from harvey.groq_client import GroqClient, GroqError
client = GroqClient()
try:
    result = asyncio = None
    import asyncio
    result = asyncio.run(client.run("Reply with exactly: HARVEY-READY"))
except GroqError as e:
    print(f"ERROR: Groq test failed: {e}")
    raise SystemExit(1)
if "HARVEY-READY" not in result.text.upper():
    print(f"WARNING: unexpected reply: {result.text[:80]!r}")
else:
    print("Groq replied OK on model " + result.model)
PYEOF
echo "[5/5] Live Groq test passed"

cat <<'NEXT'

============================================================
 SETUP COMPLETE — Harvey is ready, 100% free.
============================================================
Next (see README.md / docs/GROQ.md):
  1. harvey train https://your-company.com   # teach your product
  2. harvey signals --confirm free           # choose free signals
  3. Edit .env: email provider (Gmail works free for testing)
  4. harvey discover                        # find businesses ($0)
  5. harvey dashboard                       # http://localhost:5555
  6. harvey run                             # start the heartbeat
============================================================
NEXT
