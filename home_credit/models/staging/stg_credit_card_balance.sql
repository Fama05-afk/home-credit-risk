

with source as (
    select * from {{ source('home_credit_raw', 'credit_card_balance') }}
),

cleaned as (
    select
        -- Keys
        SK_ID_PREV                                      as prev_application_id,
        SK_ID_CURR                                      as loan_id,

        -- Timeline
        MONTHS_BALANCE                                  as months_balance,

        -- Credit card details
        AMT_BALANCE                                     as balance,
        AMT_CREDIT_LIMIT_ACTUAL                         as credit_limit,
        AMT_DRAWINGS_ATM_CURRENT                        as drawings_atm,
        AMT_DRAWINGS_CURRENT                            as drawings_total,
        AMT_DRAWINGS_OTHER_CURRENT                      as drawings_other,
        AMT_DRAWINGS_POS_CURRENT                        as drawings_pos,
        AMT_INST_MIN_REGULARITY                         as min_installment,
        AMT_PAYMENT_CURRENT                             as payment_current,
        AMT_PAYMENT_TOTAL_CURRENT                       as payment_total,
        AMT_RECEIVABLE_PRINCIPAL                        as receivable_principal,
        AMT_RECIVABLE                                   as receivable_total,
        AMT_TOTAL_RECEIVABLE                            as total_receivable,
        CNT_DRAWINGS_ATM_CURRENT                        as cnt_drawings_atm,
        CNT_DRAWINGS_CURRENT                            as cnt_drawings_total,
        CNT_DRAWINGS_OTHER_CURRENT                      as cnt_drawings_other,
        CNT_DRAWINGS_POS_CURRENT                        as cnt_drawings_pos,
        CNT_INSTALMENT_MATURE_CUM                       as installment_mature_count,
        NAME_CONTRACT_STATUS                            as contract_status,
        SK_DPD                                          as days_past_due, 
        SK_DPD_DEF                                      as days_past_due_def

    from source
)

select * from cleaned