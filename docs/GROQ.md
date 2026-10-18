# Harvey-Groq: 100% free mode

This fork adds a third brain provider, `groq`, so Harvey can think on
Groq's **free API tier** instead of requiring a Claude Pro/Max or
ChatGPT subscription. Free tier, free key, no card. Total cost: $0.

## Setup

1. **Get a free Groq API key** — https://console.groq.com/keys (sign in,
   create key, no billing needed).

2. **Install Harvey** (Python 3.11+):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```

3. **Configure**:
   Easiest: run the one-command setup (also does all yaml tuning and a
   live key test):
   ```bash
   bash setup-free.sh
   ```
   Or by hand: `cp .env.example .env`, put your Groq key in `GROQ_API_KEY=`,
   and set in `harvey.yaml`:
   ```yaml
   brain:
     provider: groq
     # optional; empty = openai/gpt-oss-120b
     model: openai/gpt-oss-120b
   ```
   Also configure an email provider in `.env` (Gmail works free for
   testing; a cheap secondary-domain mailbox for real campaigns).

4. **Run** exactly as the normal README describes:
   ```bash
   harvey train https://your-company.com
   harvey signals --confirm free
   harvey discover
   harvey dashboard   # http://localhost:5555
   harvey run
   ```

## Free-tier tuning (recommended)

Groq's free tier has requests-per-minute and tokens-per-day caps. Edit
`harvey.yaml` so Harvey stays polite and never burns through the caps:

```yaml
usage:
  heartbeat_interval_minutes: 30   # default 15; halve the pace
  max_daily_claude_percent: 40      # groq has no quota endpoint, so this
                                    # sets the call budget: 40% => 80 calls/day
```

When Groq returns a rate-limit (HTTP 429), Harvey pauses model calls for
an hour and resumes automatically — no crash, no lost pipeline state.
A bad or missing key pauses Harvey with a clear reason in
`harvey status` / the dashboard instead of retrying forever.

## Hosting without your own machine

Harvey must run continuously to work its heartbeat. Free 24/7 options:

- **Oracle Cloud Always Free** VM (ARM, 4 cores, free forever) — best fit.
- **GitHub Codespaces** (60 free hours/month) — good for testing, not all-day.
- Any always-on box you already have. Harvey is small: a few hundred MB of
  disk including its SQLite database.

## What's different from the Claude path

- Model quality: free-tier Llama models write noticeably simpler emails
  than Claude. Expect more editing of drafts before approving sends.
- Quota: Groq exposes no utilization endpoint, so Harvey budgets by its
  own daily call count (`max_daily_calls`) instead of plan percentages.
- Everything else — discovery (OpenStreetMap, $0), profiling (3 free HTTP
  reads per site), signals approval, the outbox review flow, reply
  handling, the dashboard — is identical and still costs nothing.
