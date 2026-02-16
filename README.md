# SR&ED Automation PoC

Local-first automation tool for SR&ED tax credit evidence extraction and processing.

## Setup

This project uses `uv` for dependency management.

1. **Install uv** (if not installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies**:
   ```bash
   uv sync
   ```
   This will create a `.venv` directory containing the virtual environment.

3. **Activate Virtual Environment** (optional but recommended):
   ```bash
   source .venv/bin/activate
   ```

## Configuration

Create a `.env` file in the root directory (or set environment variables):

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL_AGENT=gpt-5
OPENAI_MODEL_VISION=gpt-5-mini
# ... see src/sred/config.py for all options
```

## Usage

Run the CLI:

```bash
uv run sred doctor
```

## Development

Run tests:

```bash
uv run pytest
```
