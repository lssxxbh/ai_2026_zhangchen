from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from services import GraphService
from utils import success_response, error_response, logger

router = APIRouter(prefix="/api", tags=["graph"])


@router.get("/graph/root")
async def get_root_nodes(
    db: AsyncSession = Depends(get_db)
):
    try:
        nodes = await GraphService.get_root_nodes(db)
        node_list = GraphService.convert_to_node_list(nodes)
        return success_response(data=node_list)
    except Exception as e:
        logger.error(f"Get root nodes error: {e}")
        return error_response(msg=str(e))


@router.get("/graph/children/{node_id}")
async def get_children_nodes(
    node_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        nodes = await GraphService.get_children_nodes(db, node_id)
        node_list = GraphService.convert_to_node_list(nodes)
        return success_response(data=node_list)
    except Exception as e:
        logger.error(f"Get children nodes error: {e}")
        return error_response(msg=str(e))


@router.get("/graph/detail/{node_id}")
async def get_node_detail(
    node_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        detail = await GraphService.get_node_detail(db, node_id)
        if not detail:
            return error_response(msg="Node not found")
        return success_response(data=detail)
    except Exception as e:
        logger.error(f"Get node detail error: {e}")
        return error_response(msg=str(e))


@router.get("/graph/search")
async def search_graph(
    keyword: str = Query(..., description="搜索关键词"),
    db: AsyncSession = Depends(get_db)
):
    try:
        nodes = await GraphService.search_nodes(db, keyword)
        node_list = GraphService.convert_to_node_list(nodes)
        
        # 为每个匹配节点获取父链
        results_with_chains = []
        for node in node_list:
            chain = await GraphService.get_parent_chain(db, node["id"])
            chain_ids = [n.id for n in chain]
            results_with_chains.append({
                "node": node,
                "parent_chain": GraphService.convert_to_node_list(chain),
                "parent_chain_ids": chain_ids
            })
        
        return success_response(data=results_with_chains)
    except Exception as e:
        logger.error(f"Search nodes error: {e}")
        return error_response(msg=str(e))
