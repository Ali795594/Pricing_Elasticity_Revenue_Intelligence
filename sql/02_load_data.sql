-- =====================================================================
-- 02_load_data.sql
-- Loads the CLEANED CSV produced by src/data_cleaning.py into Postgres.
--
-- IMPORTANT: run src/generate_data.py and src/data_cleaning.py FIRST so
-- that data/processed/pricing_transactions_sql.csv exists (data_cleaning.py
-- writes this file with columns already renamed/ordered to match the
-- pricing_transactions table defined in 01_schema.sql).
--
-- Run this from the psql client on the machine that can see the file,
-- e.g.:
--   psql -U your_user -d your_database -f sql/01_schema.sql
--   psql -U your_user -d your_database -f sql/02_load_data.sql
--
-- If the Postgres server runs on a different machine/container than the
-- client, replace \copy with server-side COPY and an absolute path the
-- server process can read, or load the CSV with a tool such as pgAdmin.
-- =====================================================================

-- \copy runs client-side, so a relative path from the sql/ folder works:
\copy pricing_transactions (transaction_id, txn_date, product_id, product_name, category, region, store_id, quantity_sold, unit_price, discount_percentage, revenue, customer_segment, competitor_price, marketing_spend) FROM '../data/processed/pricing_transactions_sql.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- Quick sanity check after loading
SELECT COUNT(*) AS row_count FROM pricing_transactions;
