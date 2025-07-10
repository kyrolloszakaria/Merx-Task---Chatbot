# FastAPI Chatbot with E-commerce Integration

A modern chatbot API built with FastAPI, featuring e-commerce capabilities and advanced Natural Language Understanding (NLU). 
Implemented as a personal assistant for a Laptops and Accessories store, with sophisticated product search and order management capabilities.

## Repository Components

1. Full code base in this repository
2. Docker image available on Docker Hub:
   ```bash
   docker pull kyrolloszakaria/merx-chatbot:latest
   ```
   Or you can build the image locally:
   ```bash
   git clone https://github.com/kyrolloszakaria/Merx-Task---Chatbot
   cd Merx-Task---Chatbot
   docker build -t merx-chatbot .
   docker run -d -p 8000:8000 merx-chatbot
   ```
3. Postman collection with saved responses to showcase the model:
   - File: `Merx Chatbot API Collection.postman_collection.json`
4. API Documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Features

### Core Features
- Advanced Natural Language Understanding (NLU)
- Context-aware conversation handling for multi-step functions
- Multi-step interaction support
- User Management with role-based access
- Conversation tracking and history

### E-commerce Features
- Sophisticated product search with multiple filters:
  - Text search in name and description
  - Brand filtering
  - Price range filtering
  - Category filtering with singular/plural handling
  - Stock status filtering
- Smart search fallback (removes text filters if no results found)
- Order management system:
  - Order creation
  - Order status tracking
  - Order cancellation
- Product catalog management

### Technical Features
- PostgreSQL database with SQLAlchemy ORM
- Docker containerization
- Alembic migrations
- Input validation and error handling
- Secure password hashing
- API documentation with Swagger UI

## Intent Recognition
The chatbot can understand various user intents:
- Product Search (e.g., "Show me Dell laptops under $1500")
- Order Management (e.g., "Cancel order #123", "Track my order")
- User Profile Updates (e.g., "Update my email")
- General Queries (e.g., "Help", "Hello")

## Tech Stack

- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- Docker & Docker Compose
- Pydantic
- Passlib with Bcrypt
- spaCy for NER
- Transformers for zero-shot classification

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kyrolloszakaria/Merx-Task---Chatbot
cd Merx-Task---Chatbot
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
- API Base URL: http://localhost:8000/

## Docker Hub Deployment

The chatbot is available as a Docker image on Docker Hub.

### Pull and Run the Image

1. Pull the latest image:
```bash
docker pull kyrolloszakaria/merx-chatbot:latest
```

2. Run the container:
```bash
docker run -d -p 8000:8000 kyrolloszakaria/merx-chatbot:latest
```

### Building and Pushing Updates

If you've made changes to the code and want to update the Docker image:

1. Build and tag the new version:
```bash
docker build -t kyrolloszakaria/merx-chatbot:latest .
```

2. Push to Docker Hub:
```bash
docker login  # If not already logged in
docker push kyrolloszakaria/merx-chatbot:latest
```

### Environment Setup with Docker Hub Image

When using the Docker Hub image, make sure to:
1. Set up your environment variables
2. Configure your database connection
3. Run migrations before starting the application

For development and testing, using docker-compose (as described in the Installation section) is recommended.

## API Endpoints

### Users
- `POST /users/` - Create a new user
- `GET /users/` - List all users
- `GET /users/{user_id}` - Get user by ID

### Chat
- `POST /conversations/` - Start a new conversation
- `POST /conversations/{conversation_id}/messages` - Send a message
- `GET /conversations/{conversation_id}` - Get conversation history

### Products
- `GET /products/` - Search products with filters
- `POST /products/` - Create a new product
- `GET /products/{product_id}` - Get product details

### Orders
- `POST /orders/` - Create a new order
- `GET /orders/{order_id}` - Get order status
- `DELETE /orders/{order_id}` - Cancel an order

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
        /nlu.py    # Natural Language Understanding service
        /chat.py   # Chat handling service
        /products.py # Product search service
    main.py        # FastAPI application
    config.py      # Configuration
/migrations/        # Alembic migrations
/tests/            # Test files
```

## Chatbot Capabilities

### Product Search
The chatbot can understand complex product search queries like:
- "Show me Dell laptops under $1500"
- "Find gaming laptops with good reviews"
- "Search for laptop accessories in stock"
- "I need a new charger for my laptop"

### Order Management
Users can:
- Create orders with specific quantities
- Track order status
- Cancel orders using natural language
- Add shipping information and notes

### Smart Fallback
- If a product search returns no results, the system automatically removes the text search filter while maintaining other filters (price, category, etc.)
- Maintains context for multi-step interactions
- Handles ambiguous queries by asking for clarification

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
