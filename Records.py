from pydantic import BaseModel


class PatientRecord(BaseModel):
    encounter_id: int
    patient_nbr: int
    race: str
    gender: str
    age: str
    weight: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: str
    medical_specialty: str
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: int
    diag_2: int
    diag_3: int
    number_diagnoses: int
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str
    metformin: str
    # glipizide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str
    change: str
    diabetesMed: str

#         self.encounter_id = encounter_id
#         self.patient_nbr = patient_nbr
#         self.race = race
#         self.gender = gender
#         self.age = age
#         self.weight = weight
#         self.admission_type_id = admission_type_id
#         self.discharge_disposition_id = discharge_disposition_id
#         self.admission_source_id = admission_source_id
#         self.time_in_hospital = time_in_hospital
#         self.payer_code = payer_code
#         self.medical_specialty = medical_specialty
#         self.num_lab_procedures = num_lab_procedures
#         self.num_procedures = num_procedures
#         self.num_medications = num_medications
#         self.number_outpatient = number_outpatient
#         self.number_emergency = number_emergency
#         self.number_inpatient = number_inpatient
#         self.diag_1 = diag_1
#         self.diag_2 = diag_2
#         self.diag_3 = diag_3
#         self.number_diagnoses = number_diagnoses
#         self.max_glu_serum = max_glu_serum
#         self.A1Cresult = A1Cresult
#         self.metformin = metformin
#         self.repaglinide = repaglinide
#         self.nateglinide = nateglinide
#         self.chlorpropamide = chlorpropamide
#         self.glimepiride = glimepiride
#         self.acetohexamide = acetohexamide
#         self.glipizide = glipizide
#         self.glyburide = glyburide
#         self.tolbutamide = tolbutamide
#         self.pioglitazone = pioglitazone
#         self.rosiglitazone = rosiglitazone
#         self.acarbose = acarbose
#         self.miglitol = miglitol
#         self.troglitazone = troglitazone
#         self.tolazamide = tolazamide
#         self.examide = examide
#         self.citoglipton = citoglipton
#         self.insulin = insulin
#         self.glyburide_metformin = glyburide_metformin
#         self.glipizide_metformin = glipizide_metformin
#         self.glimepiride_pioglitazone = glimepiride_pioglitazone
#         self.metformin_rosiglitazone = metformin_rosiglitazone
#         self.metformin_pioglitazone = metformin_pioglitazone
#         self.change = change
#         self.diabetesMed = diabetesMed
#         self.readmitted = readmitted
#
# # Now you can create an object of the `PatientRecord` class using the provided variables.
# # For example:
#
# # Sample data for the patient
# data_for_patient = {
#     "encounter_id": 123456,
#     "patient_nbr": 78901234,
#     "race": "Caucasian",
#     "gender": "Female",
#     "age": "[50-60)",
#     "weight": "180 lbs",
#     "admission_type_id": 1,
#     "discharge_disposition_id": 3,
#     "admission_source_id": 7,
#     "time_in_hospital": 5,
#     "payer_code": "P567",
#     "medical_specialty": "Cardiology",
#     "num_lab_procedures": 45,
#     "num_procedures": 3,
#     "num_medications": 15,
#     "number_outpatient": 1,
#     "number_emergency": 0,
#     "number_inpatient": 0,
#     "diag_1": "250.02",
#     "diag_2": "276.14",
#     "diag_3": "648.00",
#     "number_diagnoses": 9,
#     "max_glu_serum": "None",
#     "A1Cresult": "None",
#     "metformin": "No",
#     "repaglinide": "No",
#     "nateglinide": "No",
#     "chlorpropamide": "No",
#     "glimepiride": "No",
#     "acetohexamide": "No",
#     "glipizide": "No",
#     "glyburide": "No",
#     "tolbutamide": "No",
#     "pioglitazone": "No",
#     "rosiglitazone": "No",
#     "acarbose": "No",
#     "miglitol": "No",
#     "troglitazone": "No",
#     "tolazamide": "No",
#     "examide": "No",
#     "citoglipton": "No",
#     "insulin": "No",
#     "glyburide_metformin": "No",
#     "glipizide_metformin": "No",
#     "glimepiride_pioglitazone": "No",
#     "metformin_rosiglitazone": "No",
#     "metformin_pioglitazone": "No",
#     "change": "No",
#     "diabetesMed": "Yes",
#     "readmitted": "NO"
# }
#
# # Create the patient object
# patient = PatientRecord(**data_for_patient)
#
# # Now you can access the attributes of the patient object, for example:
# print("Patient Race:", patient.race)
# print("Patient Age:", patient.age)
# print("Patient Medical Specialty:", patient.medical_specialty)
