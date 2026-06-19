
with source as (
    select * from {{ ref('stg_bureau') }}
),

aggregated as (
    select
        loan_id,

        -- Credit counts
        COUNT(*)                                                    as bureau_credit_count,
        COUNT(CASE WHEN credit_active = 'Active' THEN 1 END)       as bureau_active_count,
        COUNT(CASE WHEN credit_active = 'Closed' THEN 1 END)       as bureau_closed_count,

        -- Debt
        SUM(credit_sum_debt)                                        as bureau_debt_total,
        AVG(credit_sum)                                             as bureau_credit_avg,
        COALESCE(SUM(credit_sum), 0)                                as bureau_credit_sum_total,

        -- Overdue
        MAX(credit_max_overdue)                                     as bureau_overdue_max,
        MAX(credit_day_overdue)                                     as bureau_day_overdue_max,

        -- Timeline
        MIN(days_credit)                                            as bureau_oldest_credit,
        MAX(days_credit)                                            as bureau_most_recent_credit,

        -- Bureau ratios
        COALESCE(SUM(credit_sum_debt) / NULLIF(SUM(credit_sum), 0), 0)         as debt_credit_ratio,
        COALESCE(SUM(credit_sum_overdue) / NULLIF(SUM(credit_sum_debt), 0), 0) as overdue_debt_ratio,
        COUNT(DISTINCT credit_type)                                              as loan_types_bureau

    from source
    group by loan_id
)

select * from aggregated