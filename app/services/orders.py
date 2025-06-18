from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.models.orders import Order, OrderItem, OrderStatus
from app.models.products import Product
from app.schemas.orders import OrderCreate, OrderStatusUpdate
from app.core.exceptions import ResourceNotFoundError

class OrderService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_order(self, user_id: int, order_data: OrderCreate) -> Order:
        """Create a new order with items"""
        # Calculate total amount and validate stock
        total_amount = 0
        order_items = []
        
        for item_data in order_data.items:
            # Get product and validate
            product = self.db.query(Product).filter(Product.id == item_data.product_id).first()
            if not product:
                raise ResourceNotFoundError("Product", item_data.product_id)
            
            # Check stock
            if product.stock < item_data.quantity:
                raise ValueError(f"Insufficient stock for product {product.name}. Available: {product.stock}")
            
            # Calculate item total
            item_total = product.price * item_data.quantity
            total_amount += item_total
            
            # Create OrderItem
            order_item = OrderItem(
                product_id=product.id,
                quantity=item_data.quantity,
                unit_price=product.price,
                total_price=item_total
            )
            order_items.append(order_item)
            
            # Update stock
            product.stock -= item_data.quantity
            self.db.add(product)
        
        # Create order
        order = Order(
            user_id=user_id,
            status=OrderStatus.PENDING,
            total_amount=total_amount,
            shipping_address=order_data.shipping_address,
            notes=order_data.notes,
            items=order_items
        )
        
        try:
            self.db.add(order)
            self.db.commit()
            self.db.refresh(order)
            return order
        except Exception as e:
            self.db.rollback()
            # Restore product stock
            for item in order_items:
                product = self.db.query(Product).filter(Product.id == item.product_id).first()
                if product:
                    product.stock += item.quantity
                    self.db.add(product)
            self.db.commit()
            raise ValueError(f"Failed to create order: {str(e)}")
    
    def get_order(self, order_id: int, user_id: Optional[int] = None) -> Order:
        """Get order by ID, optionally filtering by user_id"""
        query = self.db.query(Order).filter(Order.id == order_id)
        if user_id is not None:
            query = query.filter(Order.user_id == user_id)
        
        order = query.first()
        if not order:
            raise ResourceNotFoundError("Order", order_id)
        return order
    
    def get_user_orders(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Order]:
        """Get all orders for a user"""
        return self.db.query(Order)\
            .filter(Order.user_id == user_id)\
            .order_by(Order.created_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
    
    def update_order_status(self, order_id: int, status_update: OrderStatusUpdate, user_id: Optional[int] = None) -> Order:
        """Update order status"""
        order = self.get_order(order_id, user_id)
        
        # Validate status transition
        if not self._is_valid_status_transition(order.status, status_update.status):
            raise ValueError(f"Invalid status transition from {order.status} to {status_update.status}")
        
        # If cancelling, restore stock
        if status_update.status == OrderStatus.CANCELLED and order.status != OrderStatus.CANCELLED:
            for item in order.items:
                product = self.db.query(Product).filter(Product.id == item.product_id).first()
                if product:
                    product.stock += item.quantity
                    self.db.add(product)
        
        order.status = status_update.status
        order.updated_at = datetime.utcnow()
        
        try:
            self.db.add(order)
            self.db.commit()
            self.db.refresh(order)
            return order
        except Exception as e:
            self.db.rollback()
            raise ValueError(f"Failed to update order status: {str(e)}")
    
    def _is_valid_status_transition(self, current_status: OrderStatus, new_status: OrderStatus) -> bool:
        """Validate order status transitions"""
        # Define valid transitions
        valid_transitions = {
            OrderStatus.PENDING: {OrderStatus.CONFIRMED, OrderStatus.CANCELLED},
            OrderStatus.CONFIRMED: {OrderStatus.PROCESSING, OrderStatus.CANCELLED},
            OrderStatus.PROCESSING: {OrderStatus.SHIPPED, OrderStatus.CANCELLED},
            OrderStatus.SHIPPED: {OrderStatus.DELIVERED},
            OrderStatus.DELIVERED: set(),  # No transitions from DELIVERED
            OrderStatus.CANCELLED: set()  # No transitions from CANCELLED
        }
        
        return new_status in valid_transitions.get(current_status, set())
    
    def cancel_order(self, order_id: int, user_id: Optional[int] = None) -> Order:
        """Cancel an order by ID.
        
        Args:
            order_id: The ID of the order to cancel
            user_id: Optional user ID to validate order ownership
            
        Returns:
            The cancelled order
            
        Raises:
            ResourceNotFoundError: If order not found
            ValueError: If order cannot be cancelled
        """
        status_update = OrderStatusUpdate(status=OrderStatus.CANCELLED)
        return self.update_order_status(order_id, status_update, user_id) 