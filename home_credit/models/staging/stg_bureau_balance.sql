
with source as (
    select * from {{ source('home_credit_raw', 'bureau_balance') }}
),

cleaned as (
    select
        -- Keys
        SK_ID_BUREAU                        as bureau_id,

        -- Timeline
        MONTHS_BALANCE                      as months_balance,

        -- Status
        STATUS                              as status

    from source
)

select * from cleaned