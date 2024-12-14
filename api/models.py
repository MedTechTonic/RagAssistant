import yaml
from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector
from database import Base
import numpy as np

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, index=True)
    embedding = Column(Vector(config['embedding_model']['dimension']))

    def __init__(self, content: str, embedding: np.ndarray):
        self.content = content
        self.embedding = embedding