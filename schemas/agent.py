from pydantic import BaseModel

from enum import StrEnum
from typing import Literal


# ================= INPUT =================
#class ScanAgentRequest(BaseModel):
   # image_base64: str


# ================= OUTPUT =================
#class ScanAgentResponse(BaseModel):
 #   predicted_material: str
 #   confidence_score: float
 #   explanation: str | None = None



class MaterialType(StrEnum):
    PLASTIC = "plastic"
    METAL = "metal"
    E_WASTE = "e_waste"
    GLASS = "glass"
    RUBBER = "rubber"


class RecyclableInfo(BaseModel):
    category: MaterialType


class ProductType(BaseModel):
    recyclable: RecyclableInfo | None = None
    artwork: bool = False


class ClassificationOutput(BaseModel):
    product_type: ProductType
    quality_score: int
    estimated_price: float
    confidence_score: float
    summary: str