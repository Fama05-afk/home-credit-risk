

with source as (
    select * from {{ ref('stg_pos_cash_balance') }}
),

aggregated as (
    select
        loan_id,

        -- Counts
        COUNT(*)                                                        as pos_total_months,
        COUNT(CASE WHEN days_past_due > 0 THEN 1 END)                  as pos_late_months,

        -- Late ratio
        round(
            COUNT(CASE WHEN days_past_due > 0 THEN 1 END)
            / nullif(COUNT(*), 0)
        , 4)                                                            as pos_late_ratio,

        -- Overdue
        MAX(days_past_due)                                              as pos_dpd_max,
        AVG(days_past_due)                                              as pos_dpd_avg,

        -- Active contracts
        COUNT(CASE WHEN contract_status = 'Active' THEN 1 END)         as pos_active_count

    from source
    group by loan_id
)

select * from aggregated