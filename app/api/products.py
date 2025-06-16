from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.models.database import get_db
from app.schemas.products import ProductCreate, ProductResponse, ProductSearchParams, ProductSearchResponse
from app.services.products import ProductService

router = APIRouter()

@router.post("/", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
def create_product(
    product: ProductCreate,
    db: Session = Depends(get_db)
):
    """Create a new product"""
    product_service = ProductService(db)
    return ProductResponse.from_orm(product_service.create_product(product))

@router.get("/search", response_model=ProductSearchResponse)
def search_products(
    query: Optional[str] = Query(None, description="Search term for product name or description"),
    brand: Optional[str] = Query(None, description="Brand name (e.g., Dell, Acer, Asus)"),
    max_price: Optional[float] = Query(None, gt=0, description="Maximum price"),
    min_price: Optional[float] = Query(None, gt=0, description="Minimum price"),
    category: Optional[str] = Query(None, description="Product category"),
    in_stock: Optional[bool] = Query(None, description="Filter for products in stock"),
    page: int = Query(1, gt=0, description="Page number"),
    page_size: int = Query(20, gt=0, le=100, description="Items per page"),
    db: Session = Depends(get_db)
):
    """
    Search for products with various filters:
    - Free text search in name and description
    - Brand filter
    - Price range
    - Category
    - Stock status
    
    Examples:
    - Search for all Dell laptops: /search?brand=Dell
    - Search for laptops under $1000: /search?max_price=1000
    - Search for Dell laptops under $500: /search?brand=Dell&max_price=500
    - Search for in-stock gaming laptops: /search?category=gaming&in_stock=true
    """
    # Convert page to skip/limit
    skip = (page - 1) * page_size
    
    # Create search params object
    search_params = ProductSearchParams(
        query=query,
        brand=brand,
        max_price=max_price,
        min_price=min_price,
        category=category,
        in_stock=in_stock
    )
    
    # Perform search
    product_service = ProductService(db)
    products, total = product_service.search_products(
        search_params,
        skip=skip,
        limit=page_size
    )
    
    # Return response
    return ProductSearchResponse(
        items=[ProductResponse.from_orm(p) for p in products],
        total=total,
        page=page,
        page_size=page_size
    )

@router.get("/{product_id}", response_model=ProductResponse)
def get_product(
    product_id: int,
    db: Session = Depends(get_db)
):
    """Get a product by ID"""
    product_service = ProductService(db)
    product = product_service.get_product(product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        )
    return ProductResponse.from_orm(product)

@router.put("/{product_id}", response_model=ProductResponse)
def update_product(
    product_id: int,
    product: ProductCreate,
    db: Session = Depends(get_db)
):
    """Update a product"""
    product_service = ProductService(db)
    updated_product = product_service.update_product(product_id, product)
    if not updated_product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        )
    return ProductResponse.from_orm(updated_product)

@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(
    product_id: int,
    db: Session = Depends(get_db)
):
    """Delete a product"""
    product_service = ProductService(db)
    if not product_service.delete_product(product_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        ) 