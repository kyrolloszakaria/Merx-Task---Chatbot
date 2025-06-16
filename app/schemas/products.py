from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ProductBase(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    stock: int
    category: Optional[str] = None

class ProductCreate(ProductBase):
    pass

class ProductResponse(ProductBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ProductSearchParams(BaseModel):
    query: Optional[str] = Field(None, description="Search term for product name or description")
    brand: Optional[str] = Field(None, description="Brand name (e.g., Dell, Acer, Asus)")
    max_price: Optional[float] = Field(None, gt=0, description="Maximum price")
    min_price: Optional[float] = Field(None, gt=0, description="Minimum price")
    category: Optional[str] = Field(None, description="Product category")
    in_stock: Optional[bool] = Field(None, description="Filter for products in stock")

class ProductSearchResponse(BaseModel):
    items: List[ProductResponse]
    total: int
    page: int
    page_size: int
    
    class Config:
        orm_mode = True 