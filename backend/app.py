# app.py - FastAPI backend for the records_management_system_template

import os
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError
import openai
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from typing import List, Dict
import json

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Database configuration from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "records_db")
ENGINE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(ENGINE_URL)

app = FastAPI()

# CORS for local development (allow React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FilePath(BaseModel):
    file_path: str

class SchemaInference(BaseModel):
    schema: Dict[str, str]

class NaturalLanguageQuery(BaseModel):
    query: str

# Helper function to read file (XLSX or CSV)
def read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use XLSX or CSV.")

# Helper function to infer schema using LLM (OpenAI via LlamaIndex)
def infer_schema_with_llm(df: pd.DataFrame) -> Dict[str, str]:
    # Sample data for prompting
    sample_data = df.head(5).to_json(orient="records")
    
    # Create a document with sample data
    document = Document(text=f"Sample data: {sample_data}")
    
    # Use LlamaIndex with OpenAI LLM
    llm = OpenAI(model="gpt-4o-mini")  # Use a suitable model
    index = VectorStoreIndex.from_documents([document])
    
    # Query the LLM to infer schema
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(
        "Infer the database schema from the sample data. Provide a JSON object where keys are column names and values are SQL data types (e.g., VARCHAR, INTEGER, DATE)."
    )
    
    # Parse the response
    try:
        schema = json.loads(str(response))
        return schema
    except json.JSONDecodeError:
        raise ValueError("Failed to parse schema from LLM response.")

# Helper function to create table from schema
def create_table_from_schema(table_name: str, schema: Dict[str, str]):
    metadata = MetaData()
    columns = [f"{col} {typ}" for col, typ in schema.items()]
    create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
    with engine.connect() as conn:
        conn.execute(text(create_query))
        conn.commit()

# Helper function to insert data into table
def insert_data_to_table(table_name: str, df: pd.DataFrame):
    try:
        df.to_sql(table_name, engine, if_exists='append', index=False)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function for natural language to SQL using LLM
def nl_to_sql(nl_query: str, table_name: str) -> str:
    # Assume table schema is known; for simplicity, prompt with table name
    llm = OpenAI(model="gpt-4o-mini")
    response = llm.complete(
        f"Convert this natural language query to SQL for table '{table_name}': {nl_query}. Assume standard SQL syntax for Postgres."
    )
    sql = str(response).strip()
    return sql

# API Route 2: Infer schema
@app.post("/infer-schema")
def infer_schema(file: FilePath):
    try:
        df = read_file(file.file_path)
        schema = infer_schema_with_llm(df)
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# API Route 3: Write to DB (create table and insert data)
@app.post("/write-to-db")
def write_to_db(file: FilePath = Body(...), schema: SchemaInference = Body(...), table_name: str = Body(...)):
    try:
        df = read_file(file.file_path)
        create_table_from_schema(table_name, schema.schema)
        insert_data_to_table(table_name, df)
        return {"message": f"Data written to table '{table_name}' successfully."}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Route 4: Query DB in natural language
@app.post("/query-db")
def query_db(query: NaturalLanguageQuery, table_name: str = Body(...)):
    try:
        sql = nl_to_sql(query.query, table_name)
        with engine.connect() as conn:
            result = conn.execute(text(sql)).fetchall()
            return {"sql": sql, "results": [dict(row) for row in result]}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"SQL execution error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Route 5: Placeholder for metrics/features
@app.get("/metrics")
def get_metrics(table_name: str):
    # Placeholder: Return dummy metrics
    return {
        "numerical": {"mean": 0, "median": 0},  # Extend as needed
        "categorical": {"unique_count": 0}     # Extend as needed
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)