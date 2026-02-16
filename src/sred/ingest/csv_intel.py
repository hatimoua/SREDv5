import duckdb
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from sqlmodel import Session, select
from sred.db import engine, DATA_DIR
from sred.models.core import File, Run
from sred.models.hypothesis import Hypothesis, HypothesisType, HypothesisStatus, StagingMappingProposal
from sred.llm.openai_client import get_chat_completion
from sred.logging import logger

def get_duckdb_conn():
    return duckdb.connect(database=':memory:')

def csv_profile(file_path: str) -> Dict[str, Any]:
    """
    Profile a CSV using DuckDB.
    Returns: {columns: {name, type}, sample_rows: [...]}
    """
    con = get_duckdb_conn()
    try:
        # Load CSV into temp table 'raw_csv'
        con.execute(f"CREATE TABLE raw_csv AS SELECT * FROM read_csv_auto('{file_path}', ignore_errors=true)")
        
        # Get schema
        schema_info = con.execute("DESCRIBE raw_csv").fetchall()
        columns = [{"name": row[0], "type": row[1]} for row in schema_info]
        
        # Get sample rows
        sample = con.execute("SELECT * FROM raw_csv LIMIT 5").fetchall()
        # Convert to list of dicts
        col_names = [c["name"] for c in columns]
        sample_rows = [dict(zip(col_names, row)) for row in sample]
        
        # Count
        count = con.execute("SELECT count(*) FROM raw_csv").fetchone()[0]
        
        return {
            "columns": columns,
            "row_count": count,
            "sample_rows": sample_rows
        }
    except Exception as e:
        logger.error(f"DuckDB profiling failed: {e}")
        raise
    finally:
        con.close()

def csv_query(file_path: str, sql_query: str) -> List[Dict[str, Any]]:
    """
    Execute read-only SQL on CSV.
    Query should reference table as 'raw_csv' after we create it, OR we simply replace table name?
    DuckDB can query file directly: SELECT * FROM 'file.csv'.
    User provides SQL like "SELECT * FROM data WHERE x > 5".
    We need to inject filename.
    Assumption: User knows to write `FROM 'filename'` or we replace a placeholder.
    Let's enforce a simple pattern: user writes SQL clause, we wrap it?
    Or better: User writes full SQL, but refers to table as `df`.
    We create view `df` pointing to file.
    """
    con = get_duckdb_conn()
    try:
        con.execute(f"CREATE VIEW df AS SELECT * FROM read_csv_auto('{file_path}')")
        
        # Safety check: prevent file read/write to other paths?
        # DuckDB local is powerful. Assuming trusted local user for PoC.
        
        result = con.execute(sql_query).fetchall()
        # columns?
        # description is cursor.description
        # DuckDB cursor description: [(name, type_code, ...)]
        col_names = [desc[0] for desc in con.description]
        
        rows = [dict(zip(col_names, row)) for row in result]
        return rows
    except Exception as e:
        logger.error(f"DuckDB query failed: {e}")
        return [{"error": str(e)}]
    finally:
        con.close()

MAPPING_PROMPT = """
Analyze the following CSV columns and sample data. 
Propose a mapping to the target schema:
Targets:
- date (Date of work)
- hours (Numeric duration)
- person (Name or ID, or "Employee")
- description (Task details)

Return JSON with format:
{{
    "mappings": [
        {{
            "heuristic_name": "Standard Layout",
            "column_map": {{"date": "Date Column", "hours": "Hours Col", ...}},
            "confidence": 0.9,
            "reasoning": "Columns match standard names exactly."
        }}
    ]
}}

If ambiguous (e.g. multiple date columns or hour columns), perform conservative branching by proposing max 2 alternatives.
If not confident, set confidence low.

CSV Profile:
Columns: {columns}
Sample: {sample}
"""

def propose_schema_mapping(session: Session, file: File):
    """
    Generate schema mapping hypotheses for a CSV file.
    """
    file_path = str(DATA_DIR / file.path)
    
    # 1. Profile
    try:
        profile = csv_profile(file_path)
    except Exception:
        return # Skip if failed
        
    # 2. LLM Call
    prompt = MAPPING_PROMPT.format(
        columns=json.dumps(profile["columns"]),
        sample=json.dumps(profile["sample_rows"][:2], default=str)
    )
    
    response = get_chat_completion(prompt, json_mode=True)
    try:
        data = json.loads(response)
        mappings = data.get("mappings", [])
    except json.JSONDecodeError:
        logger.error("Failed to decode LLM mapping response")
        return

    # 3. Store Proposals
    if not mappings:
        return

    # Create parent Hypothesis
    hyp = Hypothesis(
        run_id=file.run_id,
        type=HypothesisType.CSV_SCHEMA,
        description=f"Schema mapping for {file.original_filename}",
        status=HypothesisStatus.ACTIVE
    )
    session.add(hyp)
    session.commit()
    
    for m in mappings:
        prop = StagingMappingProposal(
            hypothesis_id=hyp.id,
            file_id=file.id,
            mapping_json=json.dumps(m["column_map"]),
            confidence=m["confidence"],
            reasoning=m["reasoning"]
        )
        session.add(prop)
        
    session.commit()
