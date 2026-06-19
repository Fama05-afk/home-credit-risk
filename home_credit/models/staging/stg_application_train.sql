with source as (
    select * from {{ source('home_credit_raw', 'application_train') }}
),

cleaned as (
    select
        -- Keys
        SK_ID_CURR                                          as loan_id,

        -- Target
        TARGET                                              as target,

        -- Loan characteristics
        NAME_CONTRACT_TYPE                                  as contract_type,
        AMT_CREDIT                                          as credit_amount,
        AMT_ANNUITY                                         as annuity_amount,
        AMT_INCOME_TOTAL                                    as income_amount,
        AMT_GOODS_PRICE                                     as goods_price,

        -- Client demographics
        CODE_GENDER                                         as gender,
        nullif(CODE_GENDER, 'XNA')                          as gender_clean,
        DAYS_BIRTH                                          as days_birth,
        CAST(DAYS_BIRTH / -365.25 as INT)                   as age_years,

        -- Employment
        nullif(DAYS_EMPLOYED, 365243)                       as days_employed,

        -- Socio-economic profile (added after EDA — high spread with TARGET)
        NAME_INCOME_TYPE                                    as income_type,
        OCCUPATION_TYPE                                     as occupation_type,
        NAME_EDUCATION_TYPE                                 as education_type,
        NAME_FAMILY_STATUS                                  as family_status,

        -- External credit scores (top predictors |corr| = 0.16-0.18)
        EXT_SOURCE_1                                        as ext_source_1,
        EXT_SOURCE_2                                        as ext_source_2,
        EXT_SOURCE_3                                        as ext_source_3,

        -- Regional risk
        REGION_RATING_CLIENT_W_CITY                         as region_rating,

        -- Stability indicators
        DAYS_ID_PUBLISH                                     as days_id_publish,
        DAYS_REGISTRATION                                   as days_registration,
        REG_CITY_NOT_WORK_CITY                              as reg_city_not_work_city,
        FLAG_DOCUMENT_3                                     as flag_document_3,

        -- Contact flags
        FLAG_MOBIL                                          as has_mobile,
        FLAG_EMAIL                                          as has_email,


        -- Ownership flags (Y/N → 0/1)
        CASE WHEN FLAG_OWN_CAR = 'Y' THEN 1 ELSE 0 END     as owns_car,
        CASE WHEN FLAG_OWN_REALTY = 'Y' THEN 1 ELSE 0 END  as owns_realty,

        -- Family composition
        CNT_CHILDREN                                            as cnt_children,
        CNT_FAM_MEMBERS                                         as cnt_fam_members,

        -- Car
        OWN_CAR_AGE                                             as own_car_age,

        -- Organization
        ORGANIZATION_TYPE                                       as organization_type,
        

        -- Phone stability
        DAYS_LAST_PHONE_CHANGE                                  as days_last_phone_change,

        -- Social circle defaults (very predictive)
        OBS_30_CNT_SOCIAL_CIRCLE                                as obs_30_social_circle,
        DEF_30_CNT_SOCIAL_CIRCLE                                as def_30_social_circle,
        OBS_60_CNT_SOCIAL_CIRCLE                                as obs_60_social_circle,
        DEF_60_CNT_SOCIAL_CIRCLE                                as def_60_social_circle,

        -- Credit bureau enquiries
        AMT_REQ_CREDIT_BUREAU_HOUR                              as req_bureau_hour,
        AMT_REQ_CREDIT_BUREAU_DAY                               as req_bureau_day,
        AMT_REQ_CREDIT_BUREAU_WEEK                              as req_bureau_week,
        AMT_REQ_CREDIT_BUREAU_MON                               as req_bureau_mon,
        AMT_REQ_CREDIT_BUREAU_QRT                               as req_bureau_qrt,
        AMT_REQ_CREDIT_BUREAU_YEAR                              as req_bureau_year

    from source
)

select * from cleaned