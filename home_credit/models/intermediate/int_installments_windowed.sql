{{ config(materialized='table') }}

with base as (
    select
        loan_id,
        days_instalment,
        days_entry_payment,
        instalment_amount,
        payment_amount,
        case
            when days_entry_payment is null then null
            else days_entry_payment - days_instalment
        end as days_late,
        case
            when instalment_amount is null or instalment_amount = 0 then null
            else payment_amount / instalment_amount
        end as payment_ratio,
        case
            when instalment_amount is null then null
            else instalment_amount - coalesce(payment_amount, 0)
        end as payment_diff
    from {{ ref('stg_installments_payments') }}
    where days_instalment is not null
),

agg_3m as (
    select
        loan_id,
        avg(days_late)                                          as inst_late_avg_3m,
        max(days_late)                                          as inst_late_max_3m,
        sum(case when days_late > 0 then 1 else 0 end)::double
            / nullif(count(*), 0)                               as inst_late_ratio_3m,
        sum(case when days_late > 30 then 1 else 0 end)::double
            / nullif(count(*), 0)                               as inst_dpd30_ratio_3m,
        avg(payment_ratio)                                      as inst_payment_ratio_3m,
        sum(payment_diff)                                       as inst_payment_diff_sum_3m,
        count(*)                                                as inst_count_3m
    from base
    where days_instalment >= -90
    group by loan_id
),

agg_6m as (
    select
        loan_id,
        avg(days_late)                                          as inst_late_avg_6m,
        max(days_late)                                          as inst_late_max_6m,
        sum(case when days_late > 0 then 1 else 0 end)::double
            / nullif(count(*), 0)                               as inst_late_ratio_6m,
        sum(case when days_late > 30 then 1 else 0 end)::double
            / nullif(count(*), 0)                               as inst_dpd30_ratio_6m,
        avg(payment_ratio)                                      as inst_payment_ratio_6m,
        count(*)                                                as inst_count_6m
    from base
    where days_instalment >= -180
    group by loan_id
),

agg_12m as (
    select
        loan_id,
        avg(days_late)                                          as inst_late_avg_12m,
        max(days_late)                                          as inst_late_max_12m,
        sum(case when days_late > 0 then 1 else 0 end)::double
            / nullif(count(*), 0)                               as inst_late_ratio_12m,
        sum(case when days_late > 30 then 1 else 0 end)::double
            / nullif(count(*), 0)                               as inst_dpd30_ratio_12m,
        avg(payment_ratio)                                      as inst_payment_ratio_12m,
        count(*)                                                as inst_count_12m
    from base
    where days_instalment >= -365
    group by loan_id
),

agg_all as (
    select
        loan_id,
        avg(days_late)                                          as inst_late_avg_all,
        sum(case when days_late > 0 then 1 else 0 end)::double
            / nullif(count(*), 0)                               as inst_late_ratio_all,
        count(*)                                                as inst_count_all
    from base
    group by loan_id
),

final as (
    select
        coalesce(a3.loan_id, a6.loan_id,
                 a12.loan_id, aa.loan_id)                       as loan_id,

        a3.inst_late_ratio_3m,
        a3.inst_dpd30_ratio_3m,
        a3.inst_payment_ratio_3m,
        a3.inst_late_max_3m,
        a3.inst_count_3m,

        a6.inst_late_ratio_6m,
        a6.inst_dpd30_ratio_6m,
        a6.inst_payment_ratio_6m,

        a12.inst_late_ratio_12m,
        a12.inst_dpd30_ratio_12m,
        a12.inst_payment_ratio_12m,

        case
            when aa.inst_late_ratio_all is null
              or aa.inst_late_ratio_all = 0 then null
            else a3.inst_late_ratio_3m / aa.inst_late_ratio_all
        end                                                     as inst_late_trend_3m_vs_all,

        case
            when aa.inst_late_ratio_all is null
              or aa.inst_late_ratio_all = 0 then null
            else a6.inst_late_ratio_6m / aa.inst_late_ratio_all
        end                                                     as inst_late_trend_6m_vs_all

    from agg_all aa
    left join agg_3m  a3  on aa.loan_id = a3.loan_id
    left join agg_6m  a6  on aa.loan_id = a6.loan_id
    left join agg_12m a12 on aa.loan_id = a12.loan_id
)

select * from final