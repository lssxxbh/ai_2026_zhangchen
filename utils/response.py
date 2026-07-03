from typing import Optional, Dict, Any
from pydantic import BaseModel


class ApiResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: Optional[Any] = None


def success_response(data: Optional[Any] = None, msg: str = "success") -> ApiResponse:
    return ApiResponse(code=0, msg=msg, data=data)


def error_response(code: int = 1, msg: str = "error", data: Optional[Any] = None) -> ApiResponse:
    return ApiResponse(code=code, msg=msg, data=data)
