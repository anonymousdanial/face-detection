import random
import pyodbc

# Connection parameters
server = 'localhost,1433'
database = 'ISL'
username = 'sa'
password = 'Danial123'

# Connection string
conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password}'
)

# Do not open a global connection at import time; each class manages its own connection.


class customers:
    def __init__(self):
        self.conn = pyodbc.connect(conn_str)

    def close(self):
        """Close the underlying database connection."""
        try:
            if self.conn is not None:
                self.conn.close()
                self.conn = None
        except Exception:
            # swallow close errors to avoid exceptions during cleanup
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def get_next_customer_id(self):
        """Return next CustomerID as MAX(CustomerID) + 1 (or 1 if table empty)."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COALESCE(MAX(CustomerID), 0) + 1 FROM dbo.customer")
            row = cursor.fetchone()
            return int(row[0]) if row is not None else 1
        finally:
            cursor.close()

    def add_customer(self, CustomerID=None, CustomerFaceID=None, Name=None):
        """
        Insert a new customer. If CustomerID is None, compute the next available id
        from the database. If Name is None, it will be generated as 'Customer {id}'.

        Returns the CustomerID that was inserted.
        """
        # Try to insert and handle possible duplicate-key race by retrying a few times.
        attempts = 0
        max_attempts = 5
        last_exc = None

        while attempts < max_attempts:
            attempts += 1
            cursor = self.conn.cursor()
            try:
                if CustomerID is None:
                    next_id = self.get_next_customer_id()
                else:
                    next_id = int(CustomerID)

                # If Name is None we intentionally leave it as NULL in the DB

                # If CustomerFaceID not provided, generate the canonical filename
                if not CustomerFaceID:
                    CustomerFaceID = f'customer_{next_id}'

                query = """
                    INSERT INTO dbo.customer (CustomerID, CustomerFaceID, Name)
                    VALUES (?, ?, ?)
                """
                cursor.execute(query, (next_id, CustomerFaceID, Name))
                self.conn.commit()
                cursor.close()
                disp_name = Name if Name is not None else 'NULL'
                print(f"Customer '{disp_name}' (ID={next_id}) added.")
                return next_id
            except pyodbc.IntegrityError as e:
                # Duplicate primary key — another process inserted the same id.
                # On last attempt, re-raise the exception.
                last_exc = e
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                # Loop and retry with a fresh id
                continue
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass

        # If we reach here, all attempts failed
        raise last_exc

    def add_name(self, customer_id, name):
        """
        Update the Name for the given CustomerID.

        Returns True if a row was updated, False if no matching CustomerID was found.
        Raises ValueError for invalid input. Any database exception is re-raised after
        attempting a rollback.
        """
        if customer_id is None:
            raise ValueError("customer_id must be provided")
        if name is None:
            raise ValueError("name must be provided")

        cursor = self.conn.cursor()
        try:
            query = "UPDATE dbo.customer SET Name = ? WHERE CustomerID = ?"
            cursor.execute(query, (name, int(customer_id)))
            rows = cursor.rowcount
            self.conn.commit()
            return rows > 0
        except Exception:
            # Attempt to rollback, then re-raise so callers can see the error.
            try:
                self.conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def fetch_customers(self):
        """Fetch all customers from the Customers table."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM customer"
        cursor.execute(query)
        customers = cursor.fetchall()
        cursor.close()
        return customers
    
    def fetch_customer_by_id(self, customer_id):
        """Fetch a customer by CustomerID."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM customer WHERE CustomerID = ?"
        cursor.execute(query, (customer_id,))
        customer = cursor.fetchone()
        cursor.close()
        return customer

    def fetch_customer_by_faceid(self, face_id):
        """Fetch a customer row by CustomerFaceID."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM customer WHERE CustomerFaceID = ?"
        cursor.execute(query, (face_id,))
        customer = cursor.fetchone()
        cursor.close()
        return customer

    def remove_customer(self, customer_id):
        """Remove a customer by CustomerID. Returns True if a row was deleted."""
        cursor = self.conn.cursor()
        query = "DELETE FROM customer WHERE CustomerID = ?"
        cursor.execute(query, (customer_id,))
        rows = cursor.rowcount
        self.conn.commit()
        cursor.close()
        return rows > 0


if __name__ == "__main__":
    test_customers = customers()
    # try:
    #     print(test_customers.fetch_customers())
    # finally:
    #     test_customers.close()
    test_customers.remove_customer(7)