import sys
import typer
from pathlib import Path
from sred.config import settings
from sred.logging import logger, get_run_id

app = typer.Typer(no_args_is_help=True)

@app.callback()
def main():
    """
    SR&ED Automation CLI.
    """
    pass

@app.command(name="doctor")
def doctor():
    """
    Check system configuration and environment health.
    """
    logger.info("Running doctor check...")
    
    print("\n🩺 SRED Automation Doctor\n")
    
    # Check 1: Environment / Interpreter
    print(f"Python: {sys.version.split()[0]}")
    print(f"Prefix: {sys.prefix}")
    print(f"Run ID: {get_run_id()}")
    
    # Check 2: Configuration
    print("\n[Configuration]")
    print(f"OPENAI_MODEL_AGENT:       {settings.OPENAI_MODEL_AGENT}")
    print(f"OPENAI_MODEL_VISION:      {settings.OPENAI_MODEL_VISION}")
    print(f"OPENAI_MODEL_STRUCTURED:  {settings.OPENAI_MODEL_STRUCTURED}")
    print(f"PAYROLL_MISMATCH_THRESHOLD: {settings.PAYROLL_MISMATCH_THRESHOLD}")
    
    # Mask API Key
    api_key_status = "✅ Set" if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.get_secret_value() else "❌ Missing"
    print(f"OPENAI_API_KEY:           {api_key_status}")

    # Check 3: Data Directory
    data_dir = Path("data")
    if data_dir.exists() and data_dir.is_dir():
        print(f"\n[Data Directory]          ✅ Found: {data_dir.absolute()}")
    else:
        print(f"\n[Data Directory]          ❌ Missing: {data_dir.absolute()} (Create this directory if needed)")

    print("\nDoctor check complete.")


db_app = typer.Typer(help="Database management commands.")
app.add_typer(db_app, name="db")

@db_app.command("init")
def init():
    """Initialize the database tables."""
    from sred.db import init_db
    from sred.search import setup_fts
    try:
        init_db()
        setup_fts()
        logger.info("Database initialized successfully.")
        print("✅ Database initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        print(f"❌ Failed: {e}")
        raise typer.Exit(code=1)

@db_app.command("reindex")
def reindex():
    """Rebuild FTS5 search index."""
    from sred.search import reindex_all
    try:
        reindex_all()
        print("✅ Search index rebuilt.")
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        print(f"❌ Failed: {e}")
        raise typer.Exit(code=1)

@db_app.command("search")
def search(query: str):
    """Search segments using FTS5."""
    from sred.search import search_segments
    results = search_segments(query)
    if not results:
        print("No results found.")
        return
        
    print(f"Found {len(results)} results:")
    for i, (id, snippet) in enumerate(results, 1):
        print(f"{i}. [ID {id}] {snippet}")

if __name__ == "__main__":
    app()
