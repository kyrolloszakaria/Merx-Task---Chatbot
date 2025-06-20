from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models.database import get_db
from app.models.users import User, UserRole
from app.schemas.users import UserCreate, UserResponse, UserModify
from app.services.users import UserService

router = APIRouter()

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user with the following requirements:
    - Name must be between 2 and 100 characters
    - Email must be valid and unique
    - Password must:
        - Be between 8 and 64 characters
        - Contain at least one uppercase letter
        - Contain at least one lowercase letter
        - Contain at least one number
        - Contain at least one special character
    """
    user_service = UserService(db)
    return user_service.create_user(user)

@router.patch("/{user_id}", response_model=UserResponse)
def modify_user(
    user_id: int,
    user_data: UserModify,
    db: Session = Depends(get_db)
):
    """
    Modify an existing user's information.
    You can update any combination of:
    - Name (2-100 characters)
    - Email (must be valid and unique)
    - Password (same requirements as creation)
    Only the provided fields will be updated.
    """
    user_service = UserService(db)
    return user_service.modify_user(user_id, user_data)

@router.get("/", response_model=List[UserResponse])
def get_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Retrieve a list of users with pagination.
    - skip: Number of records to skip
    - limit: Maximum number of records to return
    """
    user_service = UserService(db)
    return user_service.get_users(skip=skip, limit=limit)

@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a user by their ID.
    Raises 404 if user is not found.
    """
    user_service = UserService(db)
    return user_service.get_user(user_id)

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    Delete a user and all their associated data.
    This will also delete:
    - All user conversations
    - All user messages
    - All user orders
    - User preferences
    
    Returns 204 on success.
    Raises 404 if user is not found.
    """
    user_service = UserService(db)
    user_service.delete_user(user_id)
    return None 