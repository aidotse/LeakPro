SELECT DISTINCT
    i.subject_id,
    i.hadm_id,
    i.icustay_id,
    i.gender,
    i.admission_age AS age,
    i.ethnicity,
    i.hospital_expire_flag,
    i.hospstay_seq,
    i.los_icu,
    i.admittime,
    i.dischtime,
    i.intime,
    i.outtime,
    a.diagnosis AS diagnosis_at_admission,
    a.admission_type,
    a.insurance,
    a.deathtime,
    a.discharge_location,
    CASE WHEN a.deathtime BETWEEN i.intime AND i.outtime THEN 1 ELSE 0 END AS mort_icu,
    CASE WHEN a.deathtime BETWEEN i.admittime AND i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
    s.first_careunit,
    c.fullcode_first,
    c.dnr_first,
    c.fullcode,
    c.dnr,
    c.dnr_first_charttime,
    c.cmo_first,
    c.cmo_last,
    c.cmo,
    c.timecmo_chart,
    sofa.sofa,
    sofa.respiration AS sofa_respiratory,
    sofa.coagulation AS sofa_coagulation,
    sofa.liver AS sofa_liver,
    sofa.cardiovascular AS sofa_cardiovascular,
    sofa.cns AS sofa_cns,
    sofa.renal AS sofa_renal,
    sapsii.sapsii,
    sapsii.sapsii_prob,
    oasis.oasis,
    oasis.oasis_prob,
    COALESCE(f.readmission_30, 0) AS readmission_30
FROM mimiciii.icustay_detail i
    INNER JOIN mimiciii.admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN mimiciii.icustays s ON i.icustay_id = s.icustay_id
    INNER JOIN mimiciii.code_status c ON i.icustay_id = c.icustay_id
    LEFT OUTER JOIN (
        SELECT d.icustay_id, 1 AS readmission_30
        FROM mimiciii.icustays c, mimiciii.icustays d
        WHERE c.subject_id = d.subject_id
        AND c.icustay_id > d.icustay_id
        AND c.intime - d.outtime <= INTERVAL '30 days'
        AND c.outtime = (
            SELECT MIN(e.outtime)
            FROM mimiciii.icustays e 
            WHERE e.subject_id = c.subject_id
            AND e.intime > d.outtime
        )
    ) f ON i.icustay_id = f.icustay_id
    LEFT OUTER JOIN (
        SELECT icustay_id, sofa, respiration, coagulation, liver, cardiovascular, cns, renal 
        FROM mimiciii.sofa
    ) sofa ON i.icustay_id = sofa.icustay_id
    LEFT OUTER JOIN (
        SELECT icustay_id, sapsii, sapsii_prob 
        FROM mimiciii.sapsii
    ) sapsii ON sapsii.icustay_id = i.icustay_id
    LEFT OUTER JOIN (
        SELECT icustay_id, oasis, oasis_prob
        FROM mimiciii.oasis
    ) oasis ON oasis.icustay_id = i.icustay_id
WHERE s.first_careunit NOT LIKE 'NICU'
    AND i.hadm_id IS NOT NULL
    AND i.icustay_id IS NOT NULL
    AND i.hospstay_seq = 1
    AND i.icustay_seq = 1
    AND i.admission_age >= {min_age}
    AND i.los_icu >= {min_day}
    AND (i.outtime >= (i.intime + INTERVAL '{min_dur} hours'))
    AND (i.outtime <= (i.intime + INTERVAL '{max_dur} hours'))
ORDER BY subject_id
{limit};
