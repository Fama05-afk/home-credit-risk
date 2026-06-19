{{config(materialized='table')}}

with app as (
    select * from {{ ref('int_application_features') }}
),

bureau as (
    select * from {{ ref('int_bureau_features') }}
),

bureau_bal as (
    select * from {{ ref('int_bureau_balance_features') }}
),

prev_app as (
    select * from {{ ref('int_previous_application_features') }}
),

installments as (
    select * from {{ ref('int_installments_features') }}
),

pos_cash as (
    select * from {{ ref('int_pos_cash_features') }}
),

credit_card as (
    select * from {{ ref('int_credit_card_features') }}
),


final as (
    select
        app.loan_id,
        app.target,
        app.income_amount,
        app.credit_amount,
        app.annuity_amount,
        app.ratio_income_credit,
        app.ratio_annuity_income,
        app.days_employed,
        app.years_employed,
        app.is_employed,
        app.age_years,
        app.gender_clean,
        app.age_group,
        app.owns_car,
        app.owns_realty,
        app.ext_source_1,
        app.ext_source_2,
        app.ext_source_3,
        app.ext_source_mean,
        app.ext_source_1_available,
        app.income_type,
        app.occupation_type,
        app.education_type,
        app.family_status,
        app.region_rating,
        app.days_id_publish,
        app.days_registration,
        app.reg_city_not_work_city,
        app.flag_document_3,
        app.is_pensioner,
        app.is_working,
        app.is_unemployed,
        bureau.bureau_credit_count,
        bureau.bureau_active_count,
        bureau.bureau_debt_total,
        bureau.bureau_overdue_max,
        bureau_bal.bb_bad_months,
        bureau_bal.bb_bad_months_ratio,
        bureau_bal.bb_had_severe_overdue,
        installments.inst_late_avg,
        installments.inst_late_ratio,
        installments.inst_payment_ratio,
        installments.inst_late_max,
        prev_app.prev_application_count,
        prev_app.prev_refused_count,
        prev_app.prev_refused_ratio,
        prev_app.prev_credit_ratio,
        credit_card.cc_utilization_avg,
        credit_card.cc_atm_ratio,
        credit_card.cc_late_ratio,
        pos_cash.pos_dpd_max,
        pos_cash.pos_late_ratio,
        app.cnt_children,
        app.cnt_fam_members,
        app.own_car_age,
        app.organization_type,
        app.days_last_phone_change,
        app.obs_30_social_circle,
        app.def_30_social_circle,
        app.obs_60_social_circle,
        app.def_60_social_circle,
        app.req_bureau_hour,
        app.req_bureau_day,
        app.req_bureau_week,
        app.req_bureau_mon,
        app.req_bureau_qrt,
        app.req_bureau_year,
        app.credit_to_annuity_ratio,
        app.credit_to_goods_ratio,
        app.income_per_family_member,
        app.income_per_child,
        app.employed_to_birth_ratio,
        app.phone_to_birth_ratio,
        app.ext_source_1_x_2,
        app.ext_source_2_x_3,
        app.ext_source_1_x_2_x_3,
        app.def_30_social_ratio,
        app.def_60_social_ratio,
        app.req_bureau_recent,
        -- bureau.debt_credit_ratio,
        -- bureau.overdue_debt_ratio,
        -- bureau.loan_types_bureau
        COALESCE(bureau.debt_credit_ratio, 0)  as debt_credit_ratio,
        COALESCE(bureau.overdue_debt_ratio, 0) as overdue_debt_ratio,
        COALESCE(bureau.loan_types_bureau, 0)  as loan_types_bureau
        --bureau.bureau_credit_sum_total,
        -- prev_app.prev_app_credit_sum,
        -- installments.instpay_payment_avg,
        -- installments.instpay_payment_sum
        
       
    from app
    left join bureau               on app.loan_id = bureau.loan_id
    left join bureau_bal           on app.loan_id = bureau_bal.loan_id
    left join prev_app             on app.loan_id = prev_app.loan_id
    left join installments         on app.loan_id = installments.loan_id
    left join pos_cash             on app.loan_id = pos_cash.loan_id
    left join credit_card          on app.loan_id = credit_card.loan_id
    
)

select * from final
order by loan_id