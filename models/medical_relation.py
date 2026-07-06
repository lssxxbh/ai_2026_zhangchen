from sqlalchemy import Column, String, BigInteger, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from database import Base


class MedicalRelation(Base):
    __tablename__ = "medical_relation"

    rid = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id = Column(String(20), ForeignKey("medical_node.id"), nullable=False, index=True)
    target_id = Column(String(20), ForeignKey("medical_node.id"), nullable=False, index=True)
    rel_type = Column(String(50), nullable=False, index=True)
    rel_desc = Column(String(200), nullable=True)
    create_time = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("source_id", "target_id", name="uk_source_target"),
    )
