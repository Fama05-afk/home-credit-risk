
with source as (
    select * from {{ ref('stg_installments_payments') }}
),

aggregated as (
    select
        loan_id,

        -- Payment counts
        COUNT(*)                                                        as inst_total_count,
        COUNT(CASE WHEN days_entry_payment > days_instalment
              THEN 1 END)                                               as inst_late_count,

        -- Late payment ratio
        round(
            COUNT(CASE WHEN days_entry_payment > days_instalment
                  THEN 1 END)
            / nullif(COUNT(*), 0)
        , 4)                                                            as inst_late_ratio,

        -- Average payment delay (positive = late, negative = early)
        round(
            AVG(days_entry_payment - days_instalment)
        , 2)                                                            as inst_late_avg,

        -- Payment amount ratio
        round(
            AVG(payment_amount / nullif(instalment_amount, 0))
        , 4)                                                            as inst_payment_ratio,

        -- Max delay
        MAX(days_entry_payment - days_instalment)                       as inst_late_max,

        -- Payment amounts
        COALESCE(AVG(payment_amount), 0)                                as instpay_payment_avg,
        COALESCE(SUM(payment_amount), 0)                                as instpay_payment_sum

    from source
    group by loan_id
)

select * from aggregated