
with source as (
    select * from {{ ref('stg_credit_card_balance') }}
),

aggregated as (
    select
        loan_id,

        -- Counts
        COUNT(*)                                                        as cc_total_months,
        COUNT(CASE WHEN days_past_due > 0 THEN 1 END)                  as cc_late_months,

        -- Late ratio
        round(
            COUNT(CASE WHEN days_past_due > 0 THEN 1 END)
            / nullif(COUNT(*), 0)
        , 4)                                                            as cc_late_ratio,

        -- Overdue
        MAX(days_past_due)                                              as cc_dpd_max,

        -- Credit utilization
        round(
            AVG(balance / nullif(credit_limit, 0))
        , 4)                                                            as cc_utilization_avg,

        -- ATM usage ratio
        round(
            AVG(drawings_atm / nullif(drawings_total, 0))
        , 4)                                                            as cc_atm_ratio,

        -- Payment behaviour
        round(
            AVG(payment_current / nullif(balance, 0))
        , 4)                                                            as cc_payment_ratio

    from source
    group by loan_id
)

select * from aggregated