# portfolio

The shubhlabh app

- Full architecture and features: see `docs/ARCHITECTURE.md`.

## Reversal Email Alerts

You can send periodic email summaries of Bullish/Bearish reversals.

1) Configure SMTP via environment variables:

```
export MAIL_ENABLED=1
export MAIL_SERVER="smtp.gmail.com"
export MAIL_PORT=587
export MAIL_USE_TLS=1
export MAIL_USERNAME="your@gmail.com"
export MAIL_PASSWORD="your_app_password"   # use an app password for Gmail
export MAIL_FROM="your@gmail.com"
export MAIL_TO="you@example.com,team@example.com"
```

2) Run the sender (dry run to preview):

```
python scripts/send_reversals_email.py --universe n500 --min-score 70 --top-n 10 --dry-run
```

3) Schedule (cron example, weekdays 4pm):

```
0 16 * * 1-5 /usr/bin/env bash -lc "cd /path/to/repo && . .venv/bin/activate && \
  python scripts/send_reversals_email.py --universe n500 --min-score 70 --top-n 10 --only-if-changed"
```

Flags:
- `--universe n50|n500|watch` (default: `n500`; for `watch`, set `WATCHLIST=A.NS,B.NS`)
- `--min-score` (reversal score threshold)
- `--top-n` (max items per direction)
- `--only-if-changed` (suppress duplicate emails)
