# Dashboard themes

Pick one, paste its block into `.streamlit/config.toml` (replacing the existing
`[theme]` block), and **restart** Streamlit (`Ctrl+C`, then `streamlit run
dashboard/app.py`). Charts auto-match light/dark.

## 1. Graphite (default) — dark, warm amber
Executive terminal. Neutral, works for any company.
```toml
[theme]
base = "dark"
primaryColor = "#F2A900"
backgroundColor = "#0F1218"
secondaryBackgroundColor = "#191E27"
textColor = "#E6E9EF"
font = "sans serif"
```

## 2. Boardroom — light, deep indigo
Clean consulting-deck feel; best for printing / projecting in a bright room.
```toml
[theme]
base = "light"
primaryColor = "#2F4B7C"
backgroundColor = "#F7F8FA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#1B1F24"
font = "sans serif"
```

## 3. Signal — dark, NVIDIA green
Grounded in the tracked company: NVIDIA's own brand green as the accent.
```toml
[theme]
base = "dark"
primaryColor = "#76B900"
backgroundColor = "#101410"
secondaryBackgroundColor = "#1A201A"
textColor = "#E8EFE3"
font = "sans serif"
```

## 4. Midnight — dark navy, electric cyan
Modern fintech look; high contrast, good for screen demos.
```toml
[theme]
base = "dark"
primaryColor = "#22D3EE"
backgroundColor = "#0B1020"
secondaryBackgroundColor = "#151B2E"
textColor = "#E5E9F0"
font = "sans serif"
```
