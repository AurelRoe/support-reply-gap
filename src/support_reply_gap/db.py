import os
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv

load_dotenv()


def connection_string() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} "
        f"port={os.environ['POSTGRES_PORT']} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


@contextmanager
def get_connection():
    with psycopg.connect(connection_string()) as conn:
        yield conn
