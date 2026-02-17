from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path
from sred.logging import logger

# Import all models at module level so mappers are registered once.
# This prevents "Multiple classes found" and "Table already defined"
# errors when Streamlit hot-reloads page scripts.
from sred.models import core, vector, memory, artifact, finance, agent_log, world, alias, hypothesis  # noqa: F401

DATA_DIR = Path("data")
DB_NAME = "sred.db"
DB_URL = f"sqlite:///{DATA_DIR / DB_NAME}"

engine = create_engine(DB_URL, echo=False)

def init_db():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(exist_ok=True)
    
    logger.info(f"Initializing database at {DB_URL}")
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
