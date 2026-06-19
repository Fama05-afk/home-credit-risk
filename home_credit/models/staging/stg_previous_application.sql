-- Staging model for previous_application
-- Renames columns, casts types, and cleans known bad values

with source as (
    select * from {{ source('home_credit_raw', 'previous_application') }}
),

cleaned as (
    select
        -- Keys
        SK_ID_PREV                                      as prev_application_id,
        SK_ID_CURR                                      as loan_id,

        -- Loan characteristics
        NAME_CONTRACT_TYPE                              as contract_type,
        AMT_ANNUITY                                     as annuity_amount,
        AMT_APPLICATION                                 as application_amount,
        AMT_CREDIT                                      as credit_amount,
        AMT_DOWN_PAYMENT                                as down_payment,
        AMT_GOODS_PRICE                                 as goods_price,

        -- Timeline
        DAYS_DECISION                                   as days_decision,
        nullif(DAYS_FIRST_DRAWING, 365243)              as days_first_drawing,
        nullif(DAYS_FIRST_DUE, 365243)                  as days_first_due,
        nullif(DAYS_LAST_DUE_1ST_VERSION, 365243)       as days_last_due_first,
        nullif(DAYS_LAST_DUE, 365243)                   as days_last_due,
        nullif(DAYS_TERMINATION, 365243)                as days_termination,

        -- Application details
        NAME_CASH_LOAN_PURPOSE                          as loan_purpose,
        NAME_CONTRACT_STATUS                            as contract_status,
        NAME_PAYMENT_TYPE                               as payment_type,
        NAME_CLIENT_TYPE                                as client_type,
        NAME_GOODS_CATEGORY                             as goods_category,
        NAME_PORTFOLIO                                  as portfolio,
        NAME_PRODUCT_TYPE                               as product_type,
        CHANNEL_TYPE                                    as channel_type,
        NAME_SELLER_INDUSTRY                            as seller_industry,
        NAME_YIELD_GROUP                                as yield_group,
        PRODUCT_COMBINATION                             as product_combination,

        -- Flags
        FLAG_LAST_APPL_PER_CONTRACT                     as is_last_application,
        NFLAG_LAST_APPL_IN_DAY                          as is_last_appl_in_day,
        CNT_PAYMENT                                     as payment_count

    from source
)

select * from cleaned