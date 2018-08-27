from sqlalchemy import create_engine
from sqlalchemy.orm import Session
# from server.config import connection_strings

__all__ = ["DatabaseConnection", "DBSession"]

"""
Database first example: Python classes can be generated from an existing MySQL database using sqlacodegen command.

NB: entities are generated using sqlacodegen:
env/bin/sqlacodegen mysql+pymysql://USERNAME:PASSWORD@IP_ADDRESS_OF_MYSQL_SERVER:3306/DATABASE_NAME > output.py
"""

# engine, this is a singleton if done in conjunction with a web application;
#engine = create_engine('mysql+mysqldb://root:110@localhost:3306/aicamp')


# class DatabaseConnection:

    # def __enter__(self):
        # # make a database connection and return it
        # self.conn = engine.connect()
        # return self.conn

    # def __exit__(self, exc_type, exc_val, exc_tb):
        # # make sure the dbconnection gets closed
        # self.conn.close()


class DBSession:

    def __enter__(self):
        self.session = Session(engine)

        # NB: following line is not needed if instances of SQL entities are replaced with other classes,
        # before passing results to business logic and above layers (which is recommended!)
        # Here for simplicity, data access layer entities are passed above to business logic;
        self.session.expire_on_commit = False

        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        self.session = None
