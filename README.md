# FastAPI Chatbot with E-commerce Integration

A modern chatbot API built with FastAPI, featuring e-commerce capabilities and advanced conversation handling.

## Features

- User Management with role-based access
- Conversation tracking and history
- Product catalog management
- Order processing system
- PostgreSQL database with SQLAlchemy ORM
- Docker containerization
- Alembic migrations
- Input validation and error handling
- Secure password hashing
- API documentation with Swagger UI

## Tech Stack

- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- Docker & Docker Compose
- Pydantic
- Passlib with Bcrypt

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Git

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Build and start the containers:
```bash
docker-compose up --build
```

3. Apply database migrations:
```bash
docker-compose exec web alembic upgrade head
```

The API will be available at:
- API Documentation: http://localhost:8000/docs
- Alternative API Documentation: http://localhost:8000/redoc
- API Base URL: http://localhost:8000/api/v1

## API Endpoints

### Users
- `POST /api/v1/users/` - Create a new user
- `GET /api/v1/users/` - List all users
- `GET /api/v1/users/{user_id}` - Get user by ID

## Development

### Database Migrations

To create a new migration:
```bash
docker-compose exec web alembic revision --autogenerate -m "Description"
```

To apply migrations:
```bash
docker-compose exec web alembic upgrade head
```

### Running Tests
```bash
docker-compose exec web pytest
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_SERVER=db
POSTGRES_PORT=5432
POSTGRES_DB=chatbot
SECRET_KEY=your-secret-key-here
```

## Project Structure
```
/app
    /api/           # API routes
    /core/          # Core functionality
    /models/        # Database models
    /schemas/       # Pydantic models
    /services/      # Business logic
    main.py        # FastAPI application
    config.py      # Configuration
/migrations/        # Alembic migrations
/tests/            # Test files
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## For Problem Solving
### Edge Cases
1. Client asks an off-topic question.
2. Client Shares personal tokens