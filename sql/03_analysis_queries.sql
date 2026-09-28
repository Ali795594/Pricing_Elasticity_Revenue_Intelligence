-- =====================================================================
-- 03_analysis_queries.sql
-- Analytical queries for the Pricing Elasticity & Revenue Intelligence
-- System. Run against the pricing_transactions table (see 01_schema.sql
-- and 02_load_data.sql).
-- =====================================================================

-- 1. Revenue by product
SELECT
    product_id,
    product_name,
    category,
    ROUND(SUM(revenue), 2)      AS total_revenue,
    SUM(quantity_sold)          AS total_units,
    ROUND(AVG(unit_price), 2)   AS avg_price
FROM pricing_transactions
GROUP BY product_id, product_name, category
ORDER BY total_revenue DESC;

-- 2. Revenue by category
SELECT
    category,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    SUM(quantity_sold)              AS total_units,
    ROUND(AVG(discount_percentage), 4) AS avg_discount
FROM pricing_transactions
GROUP BY category
ORDER BY total_revenue DESC;

-- 3. Revenue by region
SELECT
    region,
    ROUND(SUM(revenue), 2)      AS total_revenue,
    COUNT(DISTINCT store_id)    AS store_count,
    ROUND(AVG(unit_price), 2)   AS avg_price
FROM pricing_transactions
GROUP BY region
ORDER BY total_revenue DESC;

-- 4. Monthly revenue trend
SELECT
    DATE_TRUNC('month', txn_date)::DATE AS month,
    ROUND(SUM(revenue), 2)              AS total_revenue,
    SUM(quantity_sold)                  AS total_units
FROM pricing_transactions
GROUP BY 1
ORDER BY 1;

-- 5. Average price by category
SELECT
    category,
    ROUND(AVG(unit_price), 2) AS avg_unit_price,
    ROUND(MIN(unit_price), 2) AS min_price,
    ROUND(MAX(unit_price), 2) AS max_price
FROM pricing_transactions
WHERE unit_price > 0
GROUP BY category
ORDER BY avg_unit_price DESC;

-- 6. Average discount by category and region
SELECT
    category,
    region,
    ROUND(AVG(discount_percentage), 4) AS avg_discount,
    COUNT(*)                           AS n_transactions
FROM pricing_transactions
GROUP BY category, region
ORDER BY category, region;

-- 7. Quantity sold by product (top 20)
SELECT
    product_id,
    product_name,
    SUM(quantity_sold) AS total_quantity
FROM pricing_transactions
GROUP BY product_id, product_name
ORDER BY total_quantity DESC
LIMIT 20;

-- 8. Customer segment performance
SELECT
    customer_segment,
    COUNT(*)                        AS n_transactions,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    ROUND(AVG(unit_price), 2)       AS avg_price,
    ROUND(AVG(discount_percentage), 4) AS avg_discount,
    ROUND(AVG(quantity_sold), 2)    AS avg_quantity
FROM pricing_transactions
GROUP BY customer_segment
ORDER BY total_revenue DESC;

-- 9. Price gap vs. competitor, by category
SELECT
    category,
    ROUND(AVG(unit_price), 2)                                   AS avg_own_price,
    ROUND(AVG(competitor_price), 2)                             AS avg_competitor_price,
    ROUND(AVG(unit_price - competitor_price), 2)                AS avg_price_gap,
    ROUND(AVG((unit_price - competitor_price) / NULLIF(competitor_price, 0)) * 100, 2) AS avg_price_gap_pct
FROM pricing_transactions
WHERE competitor_price IS NOT NULL
GROUP BY category
ORDER BY avg_price_gap_pct DESC;

-- 10. Marketing spend efficiency (revenue per marketing dollar), by category
SELECT
    category,
    ROUND(SUM(revenue), 2)          AS total_revenue,
    ROUND(SUM(marketing_spend), 2)  AS total_marketing_spend,
    ROUND(SUM(revenue) / NULLIF(SUM(marketing_spend), 0), 2) AS revenue_per_marketing_dollar
FROM pricing_transactions
WHERE marketing_spend IS NOT NULL
GROUP BY category
ORDER BY revenue_per_marketing_dollar DESC;
