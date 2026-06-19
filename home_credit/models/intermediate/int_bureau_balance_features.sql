
with source as (
    select * from {{ ref('stg_bureau_balance') }}
),

-- Level 1 : aggregate by bureau_id
bureau_agg as (
    select
        bureau_id,

        COUNT(*)                                                        as total_months,
        COUNT(CASE WHEN status IN ('1','2','3','4','5') THEN 1 END)     as bad_months,
        COUNT(CASE WHEN status IN ('4','5') THEN 1 END)                 as severe_months,
        MAX(CASE WHEN status = '5' THEN 1 ELSE 0 END)                   as had_severe_overdue

    from source
    group by bureau_id
),

-- Level 2 : join with bureau to get loan_id, then aggregate by loan_id
bureau_with_loan as (
    select
        b.loan_id,
        ba.*
    from {{ ref('stg_bureau') }} b
    inner join bureau_agg ba on b.bureau_id = ba.bureau_id
),

final as (
    select
        loan_id,

        SUM(total_months)                                               as bb_total_months,
        SUM(bad_months)                                                 as bb_bad_months,
        SUM(severe_months)                                              as bb_severe_months,
        MAX(had_severe_overdue)                                         as bb_had_severe_overdue,

        -- Proportion of bad months over total
        round(
            SUM(bad_months) / nullif(SUM(total_months), 0)
        , 4)                                                            as bb_bad_months_ratio

    from bureau_with_loan
    group by loan_id
)

select * from final