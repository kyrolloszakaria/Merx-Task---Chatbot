from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
from typing import List, Optional

from app.models.users import User, UserRole
from app.schemas.users import UserCreate, UserModify
from app.core.exceptions import UserAlreadyExistsError, ResourceNotFoundError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def create_user(self, user_data: UserCreate) -> User:
        # Check if user exists
        if self.get_user_by_email(user_data.email):
            raise UserAlreadyExistsError(user_data.email)

        # Hash password
        hashed_password = pwd_context.hash(user_data.password)
        
        try:
            db_user = User(
                name=user_data.name,
                email=user_data.email,
                password_hash=hashed_password,
                role=UserRole.USER
            )
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)
            return db_user
        except IntegrityError:
            self.db.rollback()
            raise UserAlreadyExistsError(user_data.email)

    def modify_user(self, user_id: int, user_data: UserModify) -> User:
        # Get the user
        user = self.get_user(user_id)
        
        try:
            # Update email if provided
            if user_data.email is not None:
                # Check if new email is already taken by another user
                existing_user = self.get_user_by_email(user_data.email)
                if existing_user and existing_user.id != user_id:
                    raise UserAlreadyExistsError(user_data.email)
                user.email = user_data.email

            # Update name if provided
            if user_data.name is not None:
                user.name = user_data.name

            # Update password if provided
            if user_data.password is not None:
                user.password_hash = pwd_context.hash(user_data.password)

            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
        except IntegrityError:
            self.db.rollback()
            raise UserAlreadyExistsError(user_data.email)

    def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.db.query(User).offset(skip).limit(limit).all()

    def get_user(self, user_id: int) -> User:
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ResourceNotFoundError("User", user_id)
        return user

    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password) 