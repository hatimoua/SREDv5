from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path
from sred.logging import logger

DATA_DIR = Path("data")
DB_NAME = "sred.db"
DB_URL = f"sqlite:///{DATA_DIR / DB_NAME}"

engine = create_engine(DB_URL, echo=False)

def init_db():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(exist_ok=True)
    
    # Import all models here so SQLModel knows about them
    # This is critical for create_all to work
    from sred.models import core, vector, memory, artifact, finance, agent_log  # noqa: F401
    
    logger.info(f"Initializing database at {DB_URL}")
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
