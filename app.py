import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import io

buffer = io.BytesIO()

st.title("Jaya Jaya Institute Student Prediction")

# --- Row 1 ---
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.radio('Gender', options=['Male', 'Female'],
                    help='The gender of the student')
with col2:
    age = st.number_input('Age at Enrollment', min_value=17, max_value=70,
                    help='The age of the student at the time of enrollment')
with col3:
    marital_status = st.selectbox('Marital Status', ('Single', 'Married',
                    'Widower', 'Divorced', 'Facto Union', 'Legally Seperated'),
                    help='The marital status of the student')
# --- Row 2 ---
col1, col2, col3 = st.columns(3)
with col1:
    application_mode = st.selectbox('Application Mode', (
                    '1st Phase - General Contingent',
                    '1st Phase - Special Contingent (Azores Island)',
                    '1st Phase - Special Contingent (Madeira Island)',
                    '2nd Phase - General Contingent', '3rd Phase - General Contingent',
                    'Ordinance No. 612/93', 'Ordinance No. 854-B/99',
                    'Ordinance No. 533-A/99, Item B2 (Different Plan)',
                    'Ordinance No. 533-A/99, Item B3 (Other Institution)',
                    'International Student (Bachelor)', 'Over 23 Years Old',
                    'Transfer', 'Change of Course', 'Holders of Other Higher Courses',
                    'Short Cycle Diploma Holders',
                    'Technological Specialization Diploma Holders',
                    'Change of Institution/Course',
                    'Change of Institution/Course (International)'),
                    help='The method of application used by the student'
    )
with col2:
    prev_qualification_grade = st.number_input('Previous Qualification Grade',
                    help='Grade of previous qualification (0-200)', min_value=0, max_value=200)
with col3:
    admission_grade = st.number_input('Admission Grade',
                    help="Student's admission grade (0-200)", min_value=0, max_value=200)

# --- Row 3 (Checkboxes) ---
col1, col2, col3, col4 = st.columns(4)
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

# --- Row 4 (Units Enrolled / Evaluations) ---
col1, col2, col3 = st.columns(3)
with col1:
    units_1st_enrolled = st.number_input("Units 1st Semester Enrolled", min_value=0, max_value=40, value=0)
with col2:
    units_2nd_enrolled = st.number_input("Units 2nd Semester Enrolled", min_value=0, max_value=40, value=0)
with col3:
    units_2nd_eval = st.number_input("Units 2nd Semester Evaluations", min_value=0, max_value=40, value=0)

# --- Row 5 (Units Approved / No Evaluations) ---
col1, col2, = st.columns(2)
with col1:
    units_1st_approved = st.number_input("Units 1st Semester Approved", min_value=0, max_value=40, value=0)
with col2:
    units_2nd_approved = st.number_input("Units 2nd Semester Approved", min_value=0, max_value=40, value=0)

# --- Row 6 (Grades) ---
col1, col2 = st.columns(2)
with col1:
    grade_1st = st.number_input("Units 1st Semester Grade", min_value=0, max_value=20, value=0)
with col2:
    grade_2nd = st.number_input("Units 2nd Semester Grade", min_value=0, max_value=20, value=0)

# --- Tambahan Input Sesuai List Kolom ---
col1, col2, col3 = st.columns(3)
with col1:
    application_order = st.number_input(
        "Application Order", min_value=0, max_value=10, value=0,
        help="The order of application of the student"
    )
with col2:
    course = st.selectbox("Course", [1, 2, 3, 4, 5],
        help="The code of the course the student is enrolled in")
with col3:
    attendance = st.selectbox("Attendance", ["Daytime","Evening"],
        help="The code of the course the student is enrolled in")

# --- Row: Parents Qualification & Occupation ---
col1, col2 = st.columns(2)
with col1:
    mothers_qualification = st.selectbox("Mother's Qualification", [1, 2, 3, 4, 5])
    fathers_qualification = st.selectbox("Father's Qualification", [1, 2, 3, 4, 5])
with col2:
    mothers_occupation = st.selectbox("Mother's Occupation", [1, 2, 3, 4, 5])
    fathers_occupation = st.selectbox("Father's Occupation", [1, 2, 3, 4, 5])

# --- Row: Credit Units ---
col1, col2 = st.columns(2)
with col1:
    units_1st_credited = st.number_input(
        "Units 1st Semester Credited",
        min_value=0, max_value=40, value=0,
        key="units_1st_credited"
    )
    units_1st_eval = st.number_input(
        "Units 1st Semester Evaluations",
        min_value=0, max_value=40, value=0,
        key="units_1st_eval"
    )
    units_1st_no_eval = st.number_input(
        "Units 1st Semester No Evaluations",
        min_value=0, max_value=40, value=0,
        key="units_1st_no_eval"
    )
with col2:
    units_2nd_credited = st.number_input(
        "Units 2nd Semester Credited",
        min_value=0, max_value=40, value=0,
        key="units_2nd_credited"
    )
    units_2nd_grade = st.number_input(
        "Units 2nd Semester Grade",
        min_value=0, max_value=20, value=0,
        key="units_2nd_grade"
    )
    units_2nd_no_eval = st.number_input(
        "Units 2nd Semester No Evaluations",
          min_value=0, max_value=40, value=0,
          key="units_2st_no_eval")



# --- Row: Extra Flags ---
col1, col2, col3 = st.columns(3)
with col1:
    educational_special_needs = st.radio("Educational Special Needs", ["Yes", "No"])
with col2:
    international = st.radio("International Student", ["Yes", "No"])
with col3:
    nacionality = st.selectbox("Nationality", [1, 2, 3, 4, 5])

# --- Row: Macro Indicators ---
col1, col2, col3 = st.columns(3)
with col1:
    unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0, max_value=100.0, value=0.0)
with col2:
    inflation_rate = st.number_input("Inflation Rate", min_value=-10.0, max_value=20.0, value=0.0)
with col3:
    gdp = st.number_input("GDP", min_value=0.0, max_value=1_000_000.0, value=0.0)

# --- Predict Button ---
if st.button("✨ Predict"):
    st.success("Prediction will be shown here (connect to ML model).")
