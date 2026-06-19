{{ config(materialized='table') }}

with source as (
    select * from {{ ref('stg_previous_application') }}
),

-- Agrégations sur applications approuvées uniquement
approved as (
    select
        loan_id,
        avg(credit_amount)                                      as prev_approved_credit_avg,
        max(credit_amount)                                      as prev_approved_credit_max,
        avg(annuity_amount)                                     as prev_approved_annuity_avg,
        avg(credit_amount / nullif(application_amount, 0))      as prev_approved_credit_ratio,
        count(*)                                                as prev_approved_count_check
    from source
    where contract_status = 'Approved'
    group by loan_id
),

-- Agrégations sur applications refusées uniquement
refused as (
    select
        loan_id,
        avg(credit_amount)                                      as prev_refused_credit_avg,
        max(credit_amount)                                      as prev_refused_credit_max,
        min(days_decision)                                      as prev_refused_oldest,
        max(days_decision)                                      as prev_refused_most_recent,
        count(*)                                                as prev_refused_count_check
    from source
    where contract_status = 'Refused'
    group by loan_id
),

-- Dernière application uniquement (la plus récente)
last_app as (
    select
        loan_id,
        credit_amount                                           as prev_last_credit,
        application_amount                                      as prev_last_application,
        annuity_amount                                          as prev_last_annuity,
        contract_status                                         as prev_last_status,
        days_decision                                           as prev_last_days_decision,
        payment_count                                           as prev_last_payment_count
    from (
        select *,
            row_number() over (
                partition by loan_id
                order by days_decision desc
            ) as rn
        from source
    ) ranked
    where rn = 1
),

final as (
    select
        coalesce(a.loan_id, r.loan_id, l.loan_id)              as loan_id,

        -- Approved features
        a.prev_approved_credit_avg,
        a.prev_approved_credit_max,
        a.prev_approved_annuity_avg,
        a.prev_approved_credit_ratio,

        -- Refused features
        r.prev_refused_credit_avg,
        r.prev_refused_credit_max,
        r.prev_refused_most_recent,

        -- Last application features
        l.prev_last_credit,
        l.prev_last_application,
        l.prev_last_annuity,
        l.prev_last_days_decision,
        l.prev_last_payment_count,

        -- Was last application refused ?
        case when l.prev_last_status = 'Refused' then 1 else 0 end
                                                                as prev_last_was_refused

    from approved a
    full outer join refused  r on a.loan_id = r.loan_id
    full outer join last_app l on coalesce(a.loan_id, r.loan_id) = l.loan_id
)

select * from final