

with source as (
    select * from {{ ref('stg_previous_application') }}
),

aggregated as (
    select
        loan_id,

        -- Application counts
        COUNT(*)                                                            as prev_application_count,
        COUNT(CASE WHEN contract_status = 'Approved' THEN 1 END)           as prev_approved_count,
        COUNT(CASE WHEN contract_status = 'Refused' THEN 1 END)            as prev_refused_count,
        COUNT(CASE WHEN contract_status = 'Canceled' THEN 1 END)           as prev_canceled_count,

        -- Refusal ratio
        round(
            COUNT(CASE WHEN contract_status = 'Refused' THEN 1 END)
            / nullif(COUNT(*), 0)
        , 4)                                                                as prev_refused_ratio,

        -- Credit amounts
        AVG(credit_amount)                                                  as prev_credit_avg,
        AVG(application_amount)                                             as prev_application_avg,
        COALESCE(SUM(credit_amount), 0)                                     as prev_app_credit_sum,

        -- Ratio credit granted vs requested
        round(
            AVG(credit_amount / nullif(application_amount, 0))
        , 4)                                                                as prev_credit_ratio,

        -- Timeline
        MIN(days_decision)                                                  as prev_oldest_decision,
        MAX(days_decision)                                                  as prev_most_recent_decision

    from source
    group by loan_id
)

select * from aggregated