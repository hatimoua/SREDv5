from typing import List
from pathlib import Path
from sqlmodel import Session, select
from sred.db import engine, DATA_DIR
from sred.models.core import Run, Person, File
from sred.logging import logger

def validate_schema() -> List[str]:
    """Validate that required models and fields exist in the runtime schema."""
    errors = []
    
    # This is a runtime check of our code definitions, mostly useful if we had dynamic loading or migrations pending.
    # Since we are using code-first models, if imports work, definitions are there.
    # But we can check for critical fields on classes.
    
    required_models = [Run, Person, File]
    for model in required_models:
        if not hasattr(model, "__table__"):
             errors.append(f"Model {model.__name__} is missing table definition.")
    
    return errors

def validate_data_dir() -> List[str]:
    """Validate data directory structure and permissions."""
    errors = []
    
    if not DATA_DIR.exists():
        errors.append(f"Data directory missing: {DATA_DIR}")
        return errors
        
    # Check runs upload dir creation capability
    runs_dir = DATA_DIR / "runs"
    try:
        if not runs_dir.exists():
            runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write
        test_file = runs_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        errors.append(f"Cannot write to runs directory {runs_dir}: {e}")
        
    return errors

def validate_db_connection() -> List[str]:
    """Validate database connection and basic query capability."""
    errors = []
    try:
        with Session(engine) as session:
            # Trivial query
            session.exec(select(Run).limit(1)).first()
    except Exception as e:
        errors.append(f"Database connection failed: {e}")
        
    return errors

def run_all_checks() -> List[str]:
    """Run all validation checks."""
    errors = []
    errors.extend(validate_schema())
    errors.extend(validate_data_dir())
    errors.extend(validate_db_connection())
    return errors
