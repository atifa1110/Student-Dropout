import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import io

buffer = io.BytesIO()

st.title("Jaya Jaya Institute Student Prediction")

def model_predict(df):
    model = joblib.load('model_rf.joblib')
    return model.predict(df)

def get_code(mapping, value):
    """Ambil kode (key) dari dictionary mapping berdasarkan value yang dipilih user"""
    return next(k for k, v in mapping.items() if v == value)

def data_preprocessing(data_input, n):
    df = pd.read_csv('student_data_reduced.csv')
    df = df.drop(columns=['Status'], axis=1)
    df = pd.concat([data_input, df])

    df = StandardScaler().fit_transform(df)

    return df[[n]]
    
binary_mapping = {0: 'No', 1: 'Yes'}
marital_map = {1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Divorced', 5: 'Facto Union', 6: 'Legally Seperated'}
daytime_map =  {0: 'Evening', 1: 'Daytime'}
gender_map =  {0: 'Female', 1: 'Male'}

application_mode_map = {
    1: "1st Phase - General Contingent",
    2: "Ordinance No. 612/93",
    5: "1st Phase - Special Contingent (Azores Island)",
    7: "Holders of Other Higher Courses",
    10: "Ordinance No. 854-B/99",
    15: "International Student (Bachelor)",
    16: "1st phase - Special Contingent (Madeira Island)",
    17: "2nd phase - General Contingent",
    18: "3rd phase - General Contingent",
    26: "Ordinance No. 533-A/99, Item B2 (Different Plan)",
    27: "Ordinance No. 533-A/99, Item B3 (Other Institution)",
    39: "Over 23 Years Old",
    42: "Transfer",
    43: "Change of Course",
    44: "Technological Specialization Diploma Holders",
    51: "Change of Institution/Course",
    53: "Short Cycle Diploma Holders",
    57: "Change of Institution/Course (International)"
}

course_map = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (Evening Attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (Evening Attendance)"
}

previous_qualification_map = {
    1: "Secondary Education",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    15: "10th Year of Schooling - Not Completed",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological Specialization Course",
    40: "Higher Education - Degree (1st Cycle)",
    42: "Professional Higher Technical Course",
    43: "Higher Education - Master (2nd Cycle)"
}

nacionality_map = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian"
}

mothers_qualification_map = fathers_qualification_map = {
    1: "Secondary Education - 12th Year of Schooling or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    13: "2nd Year Complementary High School Course",
    14: "10th Year of Schooling",
    18: "General Commerce Course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    20: "Complementary High School Course",
    22: "Technical - Professional Course",
    25: "Complementary High School Course - Not Concluded",
    26: "7th Year of Schooling",
    27: "2nd Cycle of the General High School Course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th Year of Schooling",
    31: "General Course of Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't Read or Write",
    36: "Can Read Without Having a 4th Year of Schooling",
    37: "Basic Education 1st Cycle (4th/5th Year) or Equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological Specialization Course",
    40: "Higher Education - Degree (1st Cycle)",
    41: "Specialized Higher Studies Course",
    42: "Professional Higher Technical Course",
    43: "Higher Education - Master (2nd Cycle)",
    44: "Higher Education - Doctorate (3rd Cycle)"
}

mothers_occupation_map = fathers_occupation_map = {
    0: "Student",
    1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative Staff",
    5: "Personal Services, Security and Safety Workers and Sellers",
    6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    7: "Skilled Workers in Industry, Construction and Craftsmen",
    8: "Installation and Machine Operators and Assembly Workers",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "(Blank)",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces Personnel",
    112: "Directors of Administrative and Commercial Services",
    114: "Hotel, Catering, Trade and Other Services Directors",
    121: "Specialists in the Physical Sciences, Mathematics, Engineering and Related Techniques",
    122: "Health Professionals",
    123: "Teachers",
    124: "Specialists in Finance, Accounting, Administrative Organization, Public and Commercial Relations",
    125: "Specialists in Information and Communication Technologies (ICT)",
    131: "Intermediate Level Science and Engineering Technicians and Professions",
    132: "Technicians and Professionals, of Intermediate Level of Health",
    134: "Intermediate Level Technicians From Legal, Social, Sports, Cultural and Similar Services",
    135: "Information and Communication Technology Technicians",
    141: "Office Workers, Secretaries in General and Data Processing Operators",
    143: "Data, Accounting, Statistical, Financial Services and Registry-Related Operators",
    144: "Other Administrative Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers and the Like",
    154: "Protection and Security Services Personnel",
    161: "Market-Oriented Farmers and Skilled Agricultural and Animal Production Workers",
    163: "Farmers, Livestock Keepers, Fishermen, Hunters and Gatherers, Subsistence",
    171: "Skilled Construction Workers and the Like, Except Electricians",
    172: "Skilled Workers in Metallurgy, Metalworking and Similar",
    173: "Skilled Workers in Printing, Precision Instrument Manufacturing, Jewelers, Artisans and the Like",
    174: "Skilled Workers in Electricity and Electronics",
    175: "Workers in Food Processing, Woodworking, Clothing and Other Industries and Crafts",
    181: "Fixed Plant and Machine Operators",
    182: "Assembly Workers",
    183: "Vehicle Drivers and Mobile Equipment Operators",
    191: "Cleaning Workers",
    192: "Unskilled Workers in Agriculture, Animal Production, Fisheries and Forestry",
    193: "Unskilled Workers in Extractive Industry, Construction, Manufacturing and Transport",
    194: "Meal Preparation Assistants",
    195: "Street Vendors (Except Food) and Street Service Providers"
}

# --- Row 1 ---
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.radio('Gender', gender_map.values(),
                    help='The gender of the student')
with col2:
    age = st.number_input('Age at Enrollment', min_value=17, max_value=70,
                    help='The age of the student at the time of enrollment')
with col3:
    marital_status = st.selectbox('Marital Status', marital_map.values(),
                    help='The marital status of the student')

col1, col2 = st.columns(2)
with col1: 
    prev_qualification = st.selectbox('Previous Qualification', previous_qualification_map.values(),
                    help='The method of application used by the student'
    )
with col2:
    prev_qualification_grade = st.number_input('Previous Qualification Grade',
                    help='Grade of previous qualification (0-200)', min_value=0.0, max_value=200.0, value=0.0,)
    
# --- Row 2 ---
col1, col2 = st.columns(2)
with col1:
    application_mode = st.selectbox('Application Mode', application_mode_map.values(),
                    help='The method of application used by the student'
    )
with col2:
    admission_grade = st.number_input('Admission Grade',
                    help="Student's admission grade (0-200)", min_value=0.0, max_value=200.0, value=0.0)

# --- Row 3 (Checkboxes) ---
col1, col2, col3, col4 = st.columns([1, 1.5, 1, 1])
with col1:
    scholarship_holder = 1 if st.checkbox(
        'Scholarship', help='Whether the student is a scholarship holder') else 0
with col2:
    tuition_fees = 1 if st.checkbox(
        'Tuition up to date', help="Whether the student's tuition fees are up to date") else 0
with col3:
    displaced = 1 if st.checkbox(
        'Displaced', help='Whether the student is a displaced person') else 0
with col4:
    debtor = 1 if st.checkbox(
        'Debtor', help='Whether the student is a debtor') else 0
    

# --- Tambahan Input Sesuai List Kolom ---
col1, col2, col3 = st.columns(3)
with col1:
    application_order = st.number_input(
        "Application Order", min_value=0, max_value=10, value=0,
        help="The order of application of the student"
    )
with col2:
    course = st.selectbox("Course", course_map.values(),
        help="The code of the course the student is enrolled in")
with col3:
    attendance = st.selectbox("Attendance", daytime_map.values(),
        help="The code of the course the student is enrolled in")

# --- Row: Parents Qualification & Occupation ---
col1, col2 = st.columns(2)
with col1:
    mothers_qualification = st.selectbox("Mother's Qualification", mothers_qualification_map.values())
    fathers_qualification = st.selectbox("Father's Qualification", fathers_qualification_map.values())
with col2:
    mothers_occupation = st.selectbox("Mother's Occupation", mothers_occupation_map.values())
    fathers_occupation = st.selectbox("Father's Occupation", fathers_occupation_map.values())

# --- Row: Credit Units ---
col1, col2 = st.columns(2)
with col1:
    units_1st_credited = st.number_input(
        "Units 1st Semester Credited",
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_1st_credited"
    )
    units_1st_enrolled = st.number_input(
        "Units 1st Semester Enrolled",
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_1st_enrolled"
    )
    units_1st_eval = st.number_input(
        "Units 1st Semester Evaluations",
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_1st_eval"
    )
    units_1st_no_eval = st.number_input(
        "Units 1st Semester No Evaluations",
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_1st_no_eval"
    )
    units_2nd_grade = st.number_input(
        "Units 2nd Semester Grade", 
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_2st_grade"
    )
with col2:
    units_2nd_credited = st.number_input(
        "Units 2nd Semester Credited",
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_2nd_credited"
    )
    units_2nd_enrolled = st.number_input(
        "Units 2st Semester Enrolled",
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_2st_enrolled"
    )
    units_2nd_eval = st.number_input(
        "Units 2st Semester Evaluations",
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_2st_eval"
    )
    units_2nd_no_eval = st.number_input(
        "Units 2nd Semester No Evaluations",
          min_value=0.0, max_value=40.0, value=0.0,
          key="units_2st_no_eval"
    )
    units_2nd_approved = st.number_input(
        "Units 2nd Semester Approved", 
        min_value=0.0, max_value=40.0, value=0.0,
        key="units_2st_approved"
    )


# --- Row: Extra Flags ---
col1, col2, col3 = st.columns(3)
with col1:
    educational_special_needs = st.radio("Educational Special Needs", binary_mapping.values())
with col2:
    international = st.radio("International Student", binary_mapping.values())
with col3:
    nacionality = st.selectbox("Nationality", nacionality_map.values())

# --- Row: Macro Indicators ---
col1, col2, col3 = st.columns(3)
with col1:
    unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0, max_value=100.0, value=0.0)
with col2:
    inflation_rate = st.number_input("Inflation Rate", min_value=-10.0, max_value=20.0, value=0.0)
with col3:
    gdp = st.number_input("GDP", min_value=0.0, max_value=1_000_000.0, value=0.0)

gender_code = get_code(gender_map, gender)
marital_status_code = get_code(marital_map, marital_status)
prev_qualification_code = get_code(previous_qualification_map, prev_qualification)
application_mode_code = get_code(application_mode_map, application_mode)
course_code = get_code(course_map,course)
attendance_code = get_code(daytime_map,attendance)
mother_qualification_code = get_code(mothers_qualification_map,mothers_qualification)
father_qualification_code = get_code(fathers_qualification_map,fathers_qualification)
mother_occupation_code = get_code(mothers_occupation_map,mothers_occupation)
father_occupation_code = get_code(fathers_occupation_map,fathers_occupation)
education_needs_code = get_code(binary_mapping, educational_special_needs)
international_code = get_code(binary_mapping, international)
nationality_code = get_code(nacionality_map,nacionality)

columns = ['Marital_status', 'Application_mode', 'Application_order',
 'Course','Daytime_evening_attendance', 'Previous_qualification',
 'Previous_qualification_grade','Nacionality','Mothers_qualification',
 'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
 'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
 'Tuition_fees_up_to_date', 'Gender',
 'Scholarship_holder','Age_at_enrollment',
 'International', 'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled', 
 'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_without_evaluations',
 'Curricular_units_2nd_sem_credited','Curricular_units_2nd_sem_enrolled',
 'Curricular_units_2nd_sem_evaluations','Curricular_units_2nd_sem_approved',
 'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
 'Unemployment_rate', 'Inflation_rate','GDP']

# --- Gabungkan semua input jadi 1 list sesuai urutan kolom model ---
data = [[
    marital_status_code,application_mode_code,application_order,
    course_code,attendance_code,prev_qualification_code,prev_qualification_grade,
    nationality_code,mother_qualification_code,father_qualification_code,
    mother_occupation_code,father_occupation_code,admission_grade, 
    displaced, education_needs_code,debtor, tuition_fees,gender_code,
    scholarship_holder,age, international_code,
    units_1st_credited, units_1st_enrolled,        
    units_1st_eval, units_1st_no_eval,    
    units_2nd_credited, units_2nd_enrolled,        
    units_2nd_eval, units_2nd_approved,        
    units_2nd_grade, units_2nd_no_eval, unemployment_rate,inflation_rate, gdp     
]]

# --- Masukkan ke DataFrame ---
df = pd.DataFrame(data, columns=columns)

# --- Predict Button ---
if st.button("✨ Predict"):
    print("==== Input Data ====")
    print(df)  # tampil di console/terminal
    print("==== Data Setelah Preprocessing ====")
    data_input = data_preprocessing(df, 0)
    print(data_input)

    output = model_predict(data_input)
    print("==== Hasil Prediksi ====")
    print(output)

    if output == 1:
        st.success('Student Status Prediction: **Graduate**')
    else:
        st.error('Student Status Prediction: **Dropout**')
