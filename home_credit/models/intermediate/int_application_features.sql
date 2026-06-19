
with source as (
    select * from {{ ref('stg_application_train') }}
),

features as (
    select
        -- Keys
        loan_id,
        target,

        -- Raw features (pass-through)
        income_amount,
        credit_amount,
        annuity_amount,
        goods_price,
        age_years,
        days_birth,
        days_employed,
        gender_clean,
        contract_type,
        owns_car,
        owns_realty,
        has_mobile,
        has_email,
        income_type,
        occupation_type,
        education_type,
        family_status,
        region_rating,
        days_id_publish,
        days_registration,
        reg_city_not_work_city,
        flag_document_3,

        -- External credit scores (pass-through — already clean)
        ext_source_1,
        ext_source_2,
        ext_source_3,

        -- EXT_SOURCE derived features
        round(
            (coalesce(ext_source_1, 0) + ext_source_2 + coalesce(ext_source_3, 0))
            / (1 + case when ext_source_1 is null then 0 else 1 end
                 + case when ext_source_3 is null then 0 else 1 end)
        , 4)                                                        as ext_source_mean,

        case when ext_source_1 is not null then 1 else 0 end        as ext_source_1_available,

        -- Axe 1 : Financial ratios
        round(income_amount / nullif(credit_amount, 0), 4)          as ratio_income_credit,
        round(annuity_amount / nullif(income_amount, 0), 4)         as ratio_annuity_income,

        -- Axe 2 : Employment stability
        case when days_employed is null then 0 else 1 end           as is_employed,
        case
            when days_employed is not null
            then round(days_employed / -365.25, 1)
            else null
        end                                                         as years_employed,

        -- Axe 3 : Demographics
        case
            when age_years < 30 then 'young'
            when age_years between 30 and 50 then 'adult'
            else 'senior'
        end                                                         as age_group,

        -- Income type flags (from EDA — NAME_INCOME_TYPE spread = 0.40)
        case when income_type = 'Pensioner' then 1 else 0 end       as is_pensioner,
        case when income_type = 'Working' then 1 else 0 end         as is_working,
        case when income_type = 'Unemployed' then 1 else 0 end      as is_unemployed,

        -- Stability score — days since last document renewal and registration
        -- More negative = more recently updated = more stable
        days_id_publish,
        days_registration,


        -- Pass-through nouvelles colonnes staging
        cnt_children,
        cnt_fam_members,
        own_car_age,
        organization_type,
        days_last_phone_change,
        obs_30_social_circle,
        def_30_social_circle,
        obs_60_social_circle,
        def_60_social_circle,
        req_bureau_hour,
        req_bureau_day,
        req_bureau_week,
        req_bureau_mon,
        req_bureau_qrt,
        req_bureau_year,

        -- Ratios financiers (Opus section 3)
        round(credit_amount / nullif(annuity_amount, 0), 4)                     as credit_to_annuity_ratio,
        round(credit_amount / nullif(goods_price, 0), 4)                        as credit_to_goods_ratio,
        round(income_amount / nullif(cnt_fam_members, 0), 4)                    as income_per_family_member,
        round(income_amount / nullif(1 + cnt_children, 0), 4)                   as income_per_child,

        -- Stabilité relative à l'âge
        round(days_employed / nullif(days_birth, 0), 4)                         as employed_to_birth_ratio,
        round(days_last_phone_change / nullif(days_birth, 0), 4)                as phone_to_birth_ratio,

        -- EXT_SOURCE interactions (Opus section 1 — simple)
        round(coalesce(ext_source_1, 0) * ext_source_2, 4)                     as ext_source_1_x_2,
        round(ext_source_2 * coalesce(ext_source_3, 0), 4)                     as ext_source_2_x_3,
        round(
            coalesce(ext_source_1, 0) * ext_source_2 * coalesce(ext_source_3, 0)
        , 4)                                                                    as ext_source_1_x_2_x_3,

        -- Social circle risk
        round(
            def_30_social_circle / nullif(obs_30_social_circle, 0)
        , 4)                                                                    as def_30_social_ratio,
        round(
            def_60_social_circle / nullif(obs_60_social_circle, 0)
        , 4)                                                                    as def_60_social_ratio,

        -- Credit bureau enquiries total récent
        coalesce(req_bureau_day, 0) + coalesce(req_bureau_week, 0)
        + coalesce(req_bureau_mon, 0)                                           as req_bureau_recent

    from source
)

select * from features