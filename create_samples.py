from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

os.makedirs('uploads', exist_ok=True)

resumes = [
    {
        "filename": "Arjun_Kumar.pdf",
        "lines": [
            "RESUME",
            "Name: Arjun Kumar",
            "Email: arjun.kumar@email.com",
            "Phone: 9876543210",
            "",
            "OBJECTIVE",
            "Seeking CNC Operator position in a reputed manufacturing company.",
            "",
            "SKILLS",
            "CNC Machining, Lathe Operation, Milling, G-Code, M-Code",
            "Blueprint Reading, AutoCAD, Quality Inspection",
            "Communication, Teamwork, MS Office",
            "",
            "EXPERIENCE",
            "CNC Operator - ABC Manufacturing Pvt Ltd",
            "3 years experience in CNC machining and lathe operations",
            "Worked on milling and drilling machines",
            "",
            "EDUCATION",
            "Diploma in Mechanical Engineering - 2019",
            "ITI Certificate in Machinist - 2017",
        ]
    },
    {
        "filename": "Priya_Sharma.pdf",
        "lines": [
            "RESUME",
            "Name: Priya Sharma",
            "Email: priya.sharma@email.com",
            "Phone: 9123456780",
            "",
            "OBJECTIVE",
            "Looking for Mechanical Engineer role in manufacturing sector.",
            "",
            "SKILLS",
            "AutoCAD, SolidWorks, CATIA, CAD, CAM, ANSYS",
            "Mechanical Design, Manufacturing Processes",
            "Leadership, English Communication",
            "",
            "EXPERIENCE",
            "Design Engineer - XYZ Industries",
            "2 years experience in mechanical design and manufacturing",
            "Worked on CAD and CAM projects",
            "",
            "EDUCATION",
            "B.E in Mechanical Engineering - 2020",
            "First Class with Distinction",
        ]
    },
    {
        "filename": "Ravi_Technician.pdf",
        "lines": [
            "RESUME",
            "Name: Ravi Krishnan",
            "Email: ravi.k@email.com",
            "Phone: 9988776655",
            "",
            "OBJECTIVE",
            "Experienced technician seeking maintenance role.",
            "",
            "SKILLS",
            "Electrical Maintenance, Hydraulic Systems, Pneumatic Systems",
            "Welding, Fitting, Repair and Maintenance",
            "Teamwork, Computer basics",
            "",
            "EXPERIENCE",
            "Maintenance Technician - Industrial Works Ltd",
            "4 years experience in electrical and hydraulic maintenance",
            "Worked on repair and fitting jobs",
            "",
            "EDUCATION",
            "ITI Certificate in Electrician - 2018",
            "Diploma in Industrial Maintenance - 2020",
        ]
    },
    {
        "filename": "Meena_QC.pdf",
        "lines": [
            "RESUME",
            "Name: Meena Devi",
            "Email: meena.d@email.com",
            "Phone: 9001122334",
            "",
            "OBJECTIVE",
            "Quality conscious professional seeking QC Inspector role.",
            "",
            "SKILLS",
            "Quality Control, Inspection, Measurement Tools",
            "Micrometer, Vernier Caliper, ISO Standards",
            "Six Sigma, Testing and Verification",
            "MS Office, English, Teamwork",
            "",
            "EXPERIENCE",
            "QC Inspector - Precision Parts Pvt Ltd",
            "2 years experience in quality inspection and testing",
            "Worked on ISO documentation and measurement",
            "",
            "EDUCATION",
            "B.Tech in Mechanical Engineering - 2021",
            "Six Sigma Green Belt Certified",
        ]
    },
    {
        "filename": "Suresh_Basic.pdf",
        "lines": [
            "RESUME",
            "Name: Suresh Babu",
            "Email: suresh.b@email.com",
            "Phone: 9445566778",
            "",
            "OBJECTIVE",
            "Fresher looking for any engineering job opportunity.",
            "",
            "SKILLS",
            "Basic Computer Knowledge",
            "Hard Working, Punctual",
            "",
            "EDUCATION",
            "10th Standard - 2019",
            "Currently pursuing ITI",
        ]
    }
]

def create_pdf(filename, lines):
    filepath = os.path.join('uploads', filename)
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    y = height - 50

    for line in lines:
        if line == "RESUME":
            c.setFont("Helvetica-Bold", 18)
            c.drawString(220, y, line)
        elif line in ["SKILLS","EXPERIENCE","EDUCATION","OBJECTIVE"]:
            c.setFont("Helvetica-Bold", 13)
            y -= 10
            c.drawString(50, y, line)
            c.line(50, y - 3, 500, y - 3)
        elif line == "":
            pass
        else:
            c.setFont("Helvetica", 11)
            c.drawString(50, y, line)
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    print(f"Created: {filepath}")

for resume in resumes:
    create_pdf(resume["filename"], resume["lines"])

print("\nAll 5 sample resumes created in uploads folder!")