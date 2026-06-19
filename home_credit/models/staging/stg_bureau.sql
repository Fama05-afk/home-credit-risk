

with source as (
    select * from {{ source('home_credit_raw', 'bureau') }}
),

cleaned as (
    select
        -- Keys
        SK_ID_CURR                              as loan_id,
        SK_ID_BUREAU                            as bureau_id,

        -- Credit status
        CREDIT_ACTIVE                           as credit_active,
        CREDIT_TYPE                             as credit_type,
        CREDIT_CURRENCY                         as credit_currency,

        -- Credit timeline
        DAYS_CREDIT                             as days_credit,
        DAYS_CREDIT_ENDDATE                     as days_credit_enddate,
        DAYS_ENDDATE_FACT                       as days_enddate_fact,
        DAYS_CREDIT_UPDATE                      as days_credit_update,
        CREDIT_DAY_OVERDUE                      as credit_day_overdue,

        -- Credit amounts
        AMT_CREDIT_SUM                          as credit_sum,
        AMT_CREDIT_SUM_DEBT                     as credit_sum_debt,
        AMT_CREDIT_SUM_LIMIT                    as credit_sum_limit,
        AMT_CREDIT_SUM_OVERDUE                  as credit_sum_overdue,
        AMT_CREDIT_MAX_OVERDUE                  as credit_max_overdue,
        AMT_ANNUITY                             as annuity_amount,

        -- Credit prolongation
        CNT_CREDIT_PROLONG                      as credit_prolong_count

    from source
)

select * from cleaned