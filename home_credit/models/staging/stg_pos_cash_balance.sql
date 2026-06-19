

with source as (
    select * from {{ source('home_credit_raw', 'POS_CASH_balance') }}
),

cleaned as (
    select
        -- Keys
        SK_ID_PREV                                      as prev_application_id,
        SK_ID_CURR                                      as loan_id,

        -- Timeline
        MONTHS_BALANCE                                  as months_balance,

        -- POS/Cash loan details
        CNT_INSTALMENT                                  as installment_count,
        CNT_INSTALMENT_FUTURE                           as installment_future_count,
        NAME_CONTRACT_STATUS                            as contract_status,
        SK_DPD                                          as days_past_due,
        SK_DPD_DEF                                      as days_past_due_def

    from source
)

select * from cleaned