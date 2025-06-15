from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from app.models.orders import OrderStatus

class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int = Field(..., gt=0)

class OrderItemResponse(OrderItemCreate):
    id: int
    unit_price: float
    total_price: float
    
    class Config:
        orm_mode = True

class OrderCreate(BaseModel):
    items: List[OrderItemCreate]
    shipping_address: Dict[str, str] = Field(..., example={
        "street": "123 Main St",
        "city": "New York",
        "state": "NY",
        "zip": "10001",
        "country": "USA"
    })
    notes: Optional[str] = None

class OrderResponse(BaseModel):
    id: int
    user_id: int
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    total_amount: float
    shipping_address: Dict[str, str]
    notes: Optional[str]
    items: List[OrderItemResponse]
    
    class Config:
        orm_mode = True

class OrderStatusUpdate(BaseModel):
    status: OrderStatus = Field(..., description="New status for the order") 