
with source as (
    select * from {{ source('home_credit_raw', 'installments_payments') }}
),

cleaned as (
    select
        -- Keys
        SK_ID_PREV                                      as prev_application_id,
        SK_ID_CURR                                      as loan_id,

        -- Installment details
        NUM_INSTALMENT_VERSION                          as installment_version,
        NUM_INSTALMENT_NUMBER                           as installment_number,

        -- Timeline
        DAYS_INSTALMENT                                 as days_instalment,
        DAYS_ENTRY_PAYMENT                              as days_entry_payment,

        -- Amounts
        AMT_INSTALMENT                                  as instalment_amount,
        AMT_PAYMENT                                     as payment_amount

    from source
)

select * from cleaned