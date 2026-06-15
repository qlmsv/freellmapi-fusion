# FreeLLMAPI Fusion

Poor-man's model fusion on top of a self-hosted [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) gateway:
fan a prompt out to several free-tier models, then have a judge model synthesize one best answer.
Exposed as an OpenAI-compatible endpoint and a Telegram bot.

## Run locally
```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.fusion.example .env.fusion   # fill UPSTREAM_API_KEY + FUSION_API_KEY
./venv/bin/python fusion_bot.py      # OpenAI API on :8000 (model="fusion") + Telegram bot
```
Point clients at `http://localhost:8000/v1`, model `fusion`, key `FUSION_API_KEY`.

## Deploy (Render)
`render.yaml` is a Blueprint for the `fusion-api` web service. Set the `sync:false`
secrets (`UPSTREAM_API_KEY`, `FUSION_API_KEY`, `TELEGRAM_TOKEN`, `ALLOWED_USER_IDS`)
in the Render dashboard.

## Test
```bash
./venv/bin/python test_fusion_bot.py   # offline, upstream mocked
```

## Benchmark
```bash
./venv/bin/pip install datasets
./venv/bin/python benchmark.py --smoke --benchmarks gsm8k,mmlu,humaneval
```
