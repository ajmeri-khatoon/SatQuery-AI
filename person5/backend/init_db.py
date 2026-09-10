from ..models import Analysis, Execution, Image, Result, User
from .database import Base, engine


def initialize_database() -> None:
	Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
	initialize_database()
	print("Database tables created successfully")