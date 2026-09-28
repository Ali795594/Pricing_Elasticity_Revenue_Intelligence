-- =====================================================================
-- 01_schema.sql
-- Pricing Elasticity & Revenue Intelligence System
-- PostgreSQL table definition for the raw transactions table.
-- =====================================================================

DROP TABLE IF EXISTS pricing_transactions;

CREATE TABLE pricing_transactions (
    transaction_id        VARCHAR(20) PRIMARY KEY,
    txn_date              DATE            NOT NULL,
    product_id            VARCHAR(10)     NOT NULL,
    product_name          VARCHAR(100)    NOT NULL,
    category              VARCHAR(50)     NOT NULL,
    region                VARCHAR(50)     NOT NULL,
    store_id              VARCHAR(10)     NOT NULL,
    quantity_sold         NUMERIC(10, 2)  NOT NULL,
    unit_price            NUMERIC(10, 2),
    discount_percentage   NUMERIC(5, 4),
    revenue               NUMERIC(12, 2)  NOT NULL,
    customer_segment      VARCHAR(20)     NOT NULL,
    competitor_price      NUMERIC(10, 2),
    marketing_spend       NUMERIC(10, 2)
);

-- Indexes that support the analytical queries in 03_analysis_queries.sql
CREATE INDEX idx_pricing_txn_date       ON pricing_transactions (txn_date);
CREATE INDEX idx_pricing_product_id     ON pricing_transactions (product_id);
CREATE INDEX idx_pricing_category       ON pricing_transactions (category);
CREATE INDEX idx_pricing_region         ON pricing_transactions (region);
CREATE INDEX idx_pricing_segment        ON pricing_transactions (customer_segment);
