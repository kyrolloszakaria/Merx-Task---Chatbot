from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models.database import get_db
from app.schemas.orders import OrderCreate, OrderResponse, OrderStatusUpdate
from app.services.orders import OrderService

router = APIRouter()

@router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(
    order: OrderCreate,
    user_id: int,  # TODO: Get from auth token
    db: Session = Depends(get_db)
):
    """
    Create a new order.
    
    Requirements:
    - Valid user ID
    - At least one item
    - Valid product IDs
    - Sufficient stock for each product
    - Valid shipping address
    """
    order_service = OrderService(db)
    db_order = order_service.create_order(user_id, order)
    return OrderResponse.from_orm(db_order)

@router.get("/{order_id}", response_model=OrderResponse)
def get_order(
    order_id: int,
    user_id: int = None,  # TODO: Get from auth token
    db: Session = Depends(get_db)
):
    """
    Get order details by ID.
    If user_id is provided, validates that the order belongs to the user.
    """
    order_service = OrderService(db)
    db_order = order_service.get_order(order_id, user_id)
    return OrderResponse.from_orm(db_order)

@router.get("/user/{user_id}", response_model=List[OrderResponse])
def get_user_orders(
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get all orders for a user with pagination.
    Orders are sorted by creation date (newest first).
    """
    order_service = OrderService(db)
    db_orders = order_service.get_user_orders(user_id, skip, limit)
    return [OrderResponse.from_orm(order) for order in db_orders]

@router.patch("/{order_id}/status", response_model=OrderResponse)
def update_order_status(
    order_id: int,
    status_update: OrderStatusUpdate,
    user_id: int = None,  # TODO: Get from auth token
    db: Session = Depends(get_db)
):
    """
    Update order status.
    
    Valid status transitions:
    - PENDING → CONFIRMED, CANCELLED
    - CONFIRMED → PROCESSING, CANCELLED
    - PROCESSING → SHIPPED, CANCELLED
    - SHIPPED → DELIVERED
    - DELIVERED → (no transitions allowed)
    - CANCELLED → (no transitions allowed)
    
    When cancelling an order:
    - Stock is restored for all items
    - Order cannot be modified further
    """
    order_service = OrderService(db)
    db_order = order_service.update_order_status(order_id, status_update, user_id)
    return OrderResponse.from_orm(db_order)