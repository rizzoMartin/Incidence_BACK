from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./tasks.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    solicitud = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    suggested_response = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    urgency = Column(String, nullable=False)
    tags = Column(String, nullable=False)  # Se almacenarán como string separado por comas

# Modelo para guardar mensajes de chat asociados a una incidencia
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    incidencia_id = Column(Integer, nullable=False)  # Relaciona con Task.id
    role = Column(String, nullable=False)  # 'user' o 'assistant'
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Crear las tablas si no existen
Base.metadata.create_all(bind=engine)
