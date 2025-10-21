DANIAL FACIAL CODE

## Notes about recent changes

- `Danial.customers.add_customer()` now computes the next CustomerID server-side using `SELECT COALESCE(MAX(CustomerID), 0) + 1` and retries on duplicate-key errors a few times. This avoids the race condition that caused IntegrityError when multiple processes attempted to insert the same CustomerID.
- `face_froze.FaceRecognizer.save_new_customer()` was updated to delegate id assignment to the database. It saves the face embedding temporarily, inserts the DB row, then renames the file to `customer_{id}.npy`.

If you want to permanently avoid primary key contention, consider altering the `dbo.customer` table to use an IDENTITY (auto-increment) column for `CustomerID` (backup DB before running migration):

```sql
ALTER TABLE dbo.customer
ADD CustomerID_new INT IDENTITY(1,1);
-- copy data, drop old PK etc.
```

Testing locally:

- Ensure your SQL Server is running and the connection parameters in `Danial.py` are correct.
- Run a small script to add a customer (this will modify your DB):

```bash
python -c "import Danial; c = Danial.customers(); print('next id', c.get_next_customer_id()); c.add_customer(CustomerFaceID='customer_test', Name='Test'); c.close()"
```

Use caution with the command above; it will write to the `ISL` database configured in `Danial.py`.