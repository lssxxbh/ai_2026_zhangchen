from typing import List, Dict, Optional, Set
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from models import MedicalNode, MedicalRelation


class GraphService:
    @staticmethod
    async def get_root_nodes(db: AsyncSession) -> List[MedicalNode]:
        """获取所有系统节点（root节点）"""
        result = await db.execute(
            select(MedicalNode).where(
                MedicalNode.node_type == "system"
            )
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_children_nodes(db: AsyncSession, parent_id: str) -> List[MedicalNode]:
        """获取指定节点的直接子节点"""
        child_result = await db.execute(
            select(MedicalNode).join(
                MedicalRelation,
                MedicalRelation.target_id == parent_id
            ).where(
                MedicalRelation.source_id == MedicalNode.id
            )
        )
        return list(child_result.scalars().all())
    
    @staticmethod
    async def get_parent_chain(db: AsyncSession, node_id: str) -> List[MedicalNode]:
        """获取从节点到root的完整父链，用于搜索展开定位"""
        chain = []
        current_id = node_id
        
        while current_id:
            node = await GraphService.get_node_by_id(db, current_id)
            if not node:
                break
            
            chain.insert(0, node)
            
            parent_result = await db.execute(
                select(MedicalRelation).where(
                    MedicalRelation.target_id == node.id
                )
            )
            parent_relation = parent_result.scalars().first()
            
            if parent_relation:
                current_id = parent_relation.source_id
            else:
                break
        
        return chain

    @staticmethod
    async def search_nodes(db: AsyncSession, keyword: str) -> List[MedicalNode]:
        search_pattern = f"%{keyword}%"
        result = await db.execute(
            select(MedicalNode).where(
                or_(
                    MedicalNode.name.like(search_pattern),
                    MedicalNode.english.like(search_pattern),
                    MedicalNode.description.like(search_pattern)
                )
            )
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_node_by_id(db: AsyncSession, node_id: str) -> Optional[MedicalNode]:
        result = await db.execute(select(MedicalNode).where(MedicalNode.id == node_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_node_detail(db: AsyncSession, node_id: str) -> Dict:
        """获取节点详情"""
        node = await GraphService.get_node_by_id(db, node_id)
        if not node:
            return None
        
        # 获取父节点
        parent_result = await db.execute(
            select(MedicalNode).join(
                MedicalRelation,
                MedicalRelation.source_id == node.id
            ).where(
                MedicalRelation.target_id == MedicalNode.id
            )
        )
        parent_nodes = list(parent_result.scalars().all())
        
        # 获取子节点数量
        children_result = await db.execute(
            select(MedicalRelation).where(
                MedicalRelation.target_id == node.id
            )
        )
        child_relations = list(children_result.scalars().all())
        child_count = len(child_relations)
        
        # 统计关联数量（所有指向或被指向该节点的关系）
        relation_count_result = await db.execute(
            select(MedicalRelation).where(
                (MedicalRelation.source_id == node.id) | (MedicalRelation.target_id == node.id)
            )
        )
        relation_count = len(list(relation_count_result.scalars().all()))
        
        return {
            "node": {
                "id": node.id,
                "name": node.name,
                "level": node.level,
                "category": node.category,
                "type": node.node_type,
                "english": node.english,
                "description": node.description
            },
            "parents": [
                {"id": p.id, "name": p.name, "category": p.category}
                for p in parent_nodes
            ],
            "child_count": child_count,
            "relation_count": relation_count
        }

    @staticmethod
    def convert_to_node_list(nodes: List[MedicalNode]) -> List[Dict]:
        node_list = []
        for node in nodes:
            node_list.append({
                "id": node.id,
                "name": node.name,
                "level": node.level,
                "category": node.category,
                "type": node.node_type,
                "english": node.english,
                "description": node.description
            })
        return node_list
