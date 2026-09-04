PERSON 5 — BACKEND + INTEGRATION

WORK:
Connect the entire SatQuery system.

Connect:
- Person 1 Agent
- Person 2 Satellite Data
- Person 3 Vision AI
- Person 4 Change/SAR AI
- Person 6 Frontend

Handle:
- Image uploads
- User questions
- AI analysis requests
- Results
- Database
- Communication between components

TOOLS:
- Python
- FastAPI
- PostgreSQL
- PostGIS
- Redis
- Docker

FOLDERS:

backend/
Main FastAPI backend and API endpoints.

integration/
Connections between the different team components.

tests/
Backend and full-system integration tests.

EXPECTED API:

POST /upload
POST /query
POST /analyze
GET /result/{id}
GET /execution/{id}
GET /health

Do not implement APIs that are not needed.

FINAL OUTPUT:
One backend through which the frontend can communicate with the entire SatQuery AI system.