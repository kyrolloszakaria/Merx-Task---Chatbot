from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import List, Optional, Tuple
import re

from app.models.products import Product
from app.schemas.products import ProductCreate, ProductSearchParams

class ProductService:
    # Dictionary mapping singular to plural forms and vice versa
    CATEGORY_MAPPINGS = {
        'accessory': 'accessories',
        'accessories': 'accessories',
        'laptop': 'laptops',
        'laptops': 'laptops',
        'display': 'displays',
        'displays': 'displays',
        'storage': 'storage',
        'memory': 'memory',
        'networking': 'networking'
    }

    def __init__(self, db: Session):
        self.db = db

    def _normalize_category(self, category: str) -> str:
        """
        Normalize category name to handle singular/plural forms and case sensitivity.
        Returns the standardized category name.
        """
        if not category:
            return None
        
        # Convert to lowercase
        category_lower = category.lower().strip()
        
        # Return the normalized form from our mapping
        return self.CATEGORY_MAPPINGS.get(category_lower, category_lower)

    def create_product(self, product_data: ProductCreate) -> Product:
        # Normalize category before saving
        if product_data.category:
            product_data.category = self._normalize_category(product_data.category)
        
        db_product = Product(**product_data.dict())
        self.db.add(db_product)
        self.db.commit()
        self.db.refresh(db_product)
        return db_product

    def get_product(self, product_id: int) -> Optional[Product]:
        return self.db.query(Product).filter(Product.id == product_id).first()

    def search_products(
        self,
        search_params: ProductSearchParams,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[Product], int]:
        """
        Search for products with various filters.
        If no results found and query filter exists, it will be removed.
        Returns a tuple of (products, total_count).
        """
        query = self.db.query(Product)
        
        # Build filter conditions
        conditions = []
        query_condition = None
        
        # Text search in name and description
        if search_params.query:
            search_term = f"%{search_params.query}%"
            query_condition = or_(
                Product.name.ilike(search_term),
                Product.description.ilike(search_term)
            )
            conditions.append(query_condition)
        
        # Brand search (in name or description)
        if search_params.brand:
            brand_term = f"%{search_params.brand}%"
            conditions.append(
                or_(
                    Product.name.ilike(brand_term),
                    Product.description.ilike(brand_term)
                )
            )
        
        # Price range
        if search_params.max_price is not None:
            conditions.append(Product.price <= search_params.max_price)
        if search_params.min_price is not None:
            conditions.append(Product.price >= search_params.min_price)
        
        # Category (case-insensitive with singular/plural handling)
        if search_params.category:
            normalized_category = self._normalize_category(search_params.category)
            if normalized_category:
                conditions.append(
                    or_(
                        Product.category.ilike(normalized_category),
                        # Also check for the original form in case it's not in our mapping
                        Product.category.ilike(search_params.category)
                    )
                )
        
        # Stock status
        if search_params.in_stock is not None:
            if search_params.in_stock:
                conditions.append(Product.stock > 0)
            else:
                conditions.append(Product.stock == 0)
        
        # Apply all conditions
        if conditions:
            query = query.filter(and_(*conditions))
        
        # Get total count before pagination
        total = query.count()

        # If no results and we have a query filter, try without it
        if total == 0 and query_condition is not None:
            # Remove the query condition from conditions
            conditions.remove(query_condition)
            # Reapply remaining filters
            query = self.db.query(Product)
            if conditions:
                query = query.filter(and_(*conditions))
            total = query.count()
        
        # Apply pagination
        query = query.order_by(Product.name).offset(skip).limit(limit)
        
        return query.all(), total

    def update_product(self, product_id: int, product_data: ProductCreate) -> Optional[Product]:
        db_product = self.get_product(product_id)
        if db_product:
            # Normalize category before updating
            if product_data.category:
                product_data.category = self._normalize_category(product_data.category)
            
            for key, value in product_data.dict(exclude_unset=True).items():
                setattr(db_product, key, value)
            self.db.commit()
            self.db.refresh(db_product)
        return db_product

    def delete_product(self, product_id: int) -> bool:
        db_product = self.get_product(product_id)
        if db_product:
            self.db.delete(db_product)
            self.db.commit()
            return True
        return False 