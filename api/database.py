import yaml
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{config['database']['user']}:"
    f"{config['database']['password']}@db:5432/"
    f"{config['database']['name']}"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()