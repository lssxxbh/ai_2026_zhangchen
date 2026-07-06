from sqlalchemy import Column, String, Integer, SmallInteger, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from database import Base


class MedicalNode(Base):
    __tablename__ = "medical_node"

    id = Column(String(20), primary_key=True, index=True)
    name = Column(String(500), nullable=False, index=True)
    parent_id = Column(String(20), ForeignKey("medical_node.id"), nullable=True)
    level = Column(SmallInteger, nullable=False, index=True)
    node_type = Column(String(30), nullable=False, index=True)
    category = Column(String(20), nullable=False)
    english = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    create_time = Column(DateTime, server_default=func.now())
    update_time = Column(DateTime, server_default=func.now(), onupdate=func.now())
