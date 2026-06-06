from flask import Flask, request, render_template, jsonify, send_file
import pdfplumber, os, io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from models import db, Analysis, Candidate
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER']   = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recruitment.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

JOB_ROLES = {
    "CNC Operator": {
        "description": """cnc machining lathe milling g-code m-code blueprint autocad
            drilling turning boring grinding fixture jig quality inspection
            mechanical engineering iti diploma experience operator""",
        "skills": ["cnc","machining","lathe","milling","g-code","m-code",
                   "blueprint","autocad","drilling","grinding"],
        "experience_keywords": ["year","years","experience","worked"],
        "education_keywords": ["diploma","iti","polytechnic","mechanical","degree"]
    },
    "Mechanical Engineer": {
        "description": """autocad solidworks catia design mechanical manufacturing
            cad cam ansys analysis simulation drawing production planning
            engineering degree btech be experience engineer""",
        "skills": ["autocad","solidworks","catia","design","mechanical",
                   "manufacturing","cad","cam","ansys","simulation"],
        "experience_keywords": ["year","years","experience","worked"],
        "education_keywords": ["b.e","b.tech","mechanical","engineering","degree"]
    },
    "Technician": {
        "description": """maintenance repair electrical hydraulic pneumatic welding
            fitting plumbing wiring motor pump compressor troubleshoot
            iti diploma certificate technical experience technician""",
        "skills": ["maintenance","repair","electrical","hydraulic","pneumatic",
                   "welding","fitting","wiring","motor","troubleshoot"],
        "experience_keywords": ["year","years","experience","worked"],
        "education_keywords": ["iti","diploma","certificate","vocational"]
    },
    "Quality Inspector": {
        "description": """quality inspection qc measurement micrometer vernier
            iso six sigma testing calibration documentation report
            standards specification tolerance degree diploma experience""",
        "skills": ["quality","inspection","qc","measurement","micrometer",
                   "vernier","iso","six sigma","testing","calibration"],
        "experience_keywords": ["year","years","experience","worked"],
        "education_keywords": ["diploma","degree","b.e","b.tech","quality"]
    }
}

# ── Train ML Model ─────────────────────────────────────────────────────────────
def train_ml_model():
    np.random.seed(42)
    X, y = [], []
    for _ in range(200):
        skill = np.random.uniform(0,100)
        exp   = np.random.choice([0,1])
        edu   = np.random.choice([0,1])
        extra = np.random.choice([0,1])
        tfidf = np.random.uniform(0,100)
        label = 1 if (skill>50 and (exp==1 or edu==1)) else 0
        X.append([skill,exp,edu,extra,tfidf])
        y.append(label)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(Xs, y)
    return model, scaler

ML_MODEL, ML_SCALER = train_ml_model()

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_text(filepath):
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + " "
    except: pass
    return text.lower()

def tfidf_cosine_score(resume_text, job_desc):
    try:
        vec    = TfidfVectorizer(stop_words='english')
        matrix = vec.fit_transform([resume_text, job_desc])
        score  = cosine_similarity(matrix[0:1], matrix[1:2])
        return round(float(score[0][0])*100, 1)
    except: return 0.0

def keyword_score(text, role_name):
    role    = JOB_ROLES[role_name]
    matched = [s for s in role["skills"] if s in text]
    missing = [s for s in role["skills"] if s not in text]
    skill_pct   = (len(matched)/len(role["skills"]))*100
    has_exp     = int(any(w in text for w in role["experience_keywords"]))
    has_edu     = int(any(w in text for w in role["education_keywords"]))
    extra_words = ["communication","teamwork","leadership","ms office","english","computer"]
    has_extra   = int(any(w in text for w in extra_words))
    explanation = []
    if matched:     explanation.append(f"Matched skills: {', '.join(matched)}")
    if has_exp:     explanation.append("Has relevant experience")
    if has_edu:     explanation.append("Has relevant education")
    if has_extra:   explanation.append("Has additional soft skills")
    return {"skill_pct":round(skill_pct,1),"has_exp":has_exp,"has_edu":has_edu,
            "has_extra":has_extra,"matched_skills":matched,
            "missing_skills":missing,"explanation":explanation}

def ml_predict(skill_pct, has_exp, has_edu, has_extra, tfidf):
    features = np.array([[skill_pct,has_exp,has_edu,has_extra,tfidf]])
    scaled   = ML_SCALER.transform(features)
    prob     = ML_MODEL.predict_proba(scaled)[0][1]
    label    = ML_MODEL.predict(scaled)[0]
    return round(prob*100,1), int(label)

def get_radar(text, role_name, tfidf_score):
    role    = JOB_ROLES[role_name]
    matched = sum(1 for s in role["skills"] if s in text)
    has_exp = any(w in text for w in role["experience_keywords"])
    has_edu = any(w in text for w in role["education_keywords"])
    has_extra = any(w in text for w in ["communication","teamwork","leadership"])
    return {
        "Technical Skills": round((matched/len(role["skills"]))*100),
        "Experience":       100 if has_exp   else 15,
        "Education":        100 if has_edu   else 15,
        "Soft Skills":      100 if has_extra else 15,
        "AI Match":         round(tfidf_score)
    }

# ── NLTK-style text cleaning (Tier 2) ─────────────────────────────────────────
STOPWORDS = {"a","an","the","and","or","but","in","on","at","to","for",
             "of","with","by","from","is","was","are","were","be","been",
             "have","has","had","do","does","did","will","would","could",
             "should","may","might","i","my","we","our","you","your"}

def clean_text(text):
    import re
    text = re.sub(r'[^a-z0-9\s\-]','', text.lower())
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w)>2]
    return " ".join(words)

# ── Interview question generator (Tier 2) ─────────────────────────────────────
INTERVIEW_QUESTIONS = {
    "CNC Operator": [
        "Can you explain the difference between G-code and M-code?",
        "How do you set up a CNC machine for a new job?",
        "What safety precautions do you follow while operating CNC machines?",
        "Describe a time you identified and fixed a machining error.",
        "How do you read and interpret engineering blueprints?",
    ],
    "Mechanical Engineer": [
        "What CAD software have you used and for what type of projects?",
        "Explain the difference between CAD and CAM.",
        "How do you approach a new mechanical design problem?",
        "Have you used ANSYS or any simulation tool? Explain your experience.",
        "How do you ensure manufacturing quality in your designs?",
    ],
    "Technician": [
        "Describe your experience with hydraulic and pneumatic systems.",
        "How do you troubleshoot an electrical fault in a machine?",
        "What safety procedures do you follow during maintenance?",
        "Have you worked with welding equipment? What types?",
        "How do you prioritize maintenance tasks during production?",
    ],
    "Quality Inspector": [
        "What measuring instruments have you used for quality inspection?",
        "Explain the concept of Six Sigma and how you have applied it.",
        "How do you document a non-conformance report?",
        "What is the difference between accuracy and precision in measurement?",
        "How do you handle a situation where a batch fails quality standards?",
    ]
}

# ── PDF Report Generator ───────────────────────────────────────────────────────
def generate_pdf_report(analysis_data, candidates_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    # Header
    c.setFillColor(colors.HexColor('#1a365d'))
    c.rect(0, h-80, w, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, h-40, "AI-Based Candidate Selection Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, h-62, f"Sri Krishna Engineering  |  {datetime.now().strftime('%d %B %Y  %H:%M')}")

    # Summary box
    c.setFillColor(colors.HexColor('#ebf8ff'))
    c.rect(30, h-160, w-60, 68, fill=True, stroke=False)
    c.setFillColor(colors.HexColor('#1a365d'))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(44, h-104, f"Job Role: {analysis_data['job_role']}")
    c.setFont("Helvetica", 11)
    c.drawString(44, h-122, f"Total Resumes: {analysis_data['total']}     "
                            f"Recommended: {analysis_data['recommended']}     "
                            f"Avg Score: {analysis_data['avg_score']}/100")
    c.drawString(44, h-140, f"Top Candidate: {analysis_data['top_candidate']['name']}  "
                            f"(Score: {analysis_data['top_candidate']['score']}/100)")

    # Candidates
    y = h - 185
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor('#1a365d'))
    c.drawString(40, y, "Candidate Rankings")
    y -= 8
    c.setStrokeColor(colors.HexColor('#bee3f8'))
    c.line(40, y, w-40, y)
    y -= 20

    for cand in candidates_data:
        if y < 120:
            c.showPage()
            y = h - 60

        # Candidate name row
        rec_color = ('#c6f6d5' if cand['recommendation']=='Highly Recommended'
                     else '#fefcbf' if cand['recommendation']=='Recommended'
                     else '#fed7d7')
        c.setFillColor(colors.HexColor('#f7fafc'))
        c.rect(30, y-44, w-60, 54, fill=True, stroke=False)

        rank_emoji = ['🥇','🥈','🥉']
        rank_label = rank_emoji[cand['rank']-1] if cand['rank']<=3 else f"#{cand['rank']}"

        c.setFillColor(colors.HexColor('#1a365d'))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(44, y+2, f"{cand['rank']}.  {cand['name']}")

        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor('#4a5568'))
        c.drawString(44, y-14, f"Total Score: {cand['score']}/100   |   "
                               f"TF-IDF: {cand['tfidf_score']}%   |   "
                               f"ML Confidence: {cand['ml_prob']}%   |   "
                               f"Skills: {cand['skill_pct']}%")

        c.setFillColor(colors.HexColor(rec_color))
        c.rect(w-160, y-10, 120, 20, fill=True, stroke=False)
        c.setFillColor(colors.HexColor('#276749'))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(w-155, y-2, cand['recommendation'])

        if cand['matched_skills']:
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor('#718096'))
            c.drawString(44, y-30, f"Matched skills: {', '.join(cand['matched_skills'])}")

        y -= 68

    # Interview Questions
    c.showPage()
    y = h - 60
    c.setFillColor(colors.HexColor('#1a365d'))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"Suggested Interview Questions — {analysis_data['job_role']}")
    y -= 20
    c.line(40, y, w-40, y)
    y -= 24

    questions = INTERVIEW_QUESTIONS.get(analysis_data['job_role'], [])
    for i, q in enumerate(questions, 1):
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(colors.HexColor('#2b6cb0'))
        c.drawString(40, y, f"Q{i}.")
        c.setFont("Helvetica", 11)
        c.setFillColor(colors.HexColor('#1a202c'))
        c.drawString(64, y, q)
        y -= 28

    # Footer
    c.setFillColor(colors.HexColor('#f0f4f8'))
    c.rect(0, 0, w, 36, fill=True, stroke=False)
    c.setFillColor(colors.HexColor('#718096'))
    c.setFont("Helvetica", 9)
    c.drawString(40, 13, "AI-Based Candidate Selection System  |  Sri Krishna Engineering  |  IET Industry Project")

    c.save()
    buffer.seek(0)
    return buffer

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', roles=list(JOB_ROLES.keys()))

@app.route('/history')
def history():
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/analyze', methods=['POST'])
def analyze():
    job_role = request.form.get('job_role')
    files    = request.files.getlist('resumes')
    if not files or not job_role:
        return jsonify({"error": "Please upload resumes and select a job role."})

    job_desc   = JOB_ROLES[job_role]["description"]
    candidates = []

    for file in files:
        if file.filename == '': continue
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        raw_text    = extract_text(filepath)
        clean       = clean_text(raw_text)
        tfidf       = tfidf_cosine_score(clean, job_desc)
        kw          = keyword_score(raw_text, job_role)
        ml_prob, ml_label = ml_predict(kw["skill_pct"],kw["has_exp"],
                                       kw["has_edu"],kw["has_extra"],tfidf)
        score = min(100, round(
            (kw["skill_pct"]*0.35)+(tfidf*0.25)+(ml_prob*0.20)+
            (kw["has_exp"]*20)+(kw["has_edu"]*10)+(kw["has_extra"]*5), 1))

        explanation = kw["explanation"].copy()
        explanation.insert(0, f"TF-IDF cosine similarity: {tfidf}%")
        explanation.insert(1, f"ML model confidence: {ml_prob}%")

        candidates.append({
            "name":           file.filename.replace('.pdf','').replace('_',' ').title(),
            "score":          score,
            "tfidf_score":    tfidf,
            "ml_prob":        ml_prob,
            "ml_suitable":    ml_label,
            "skill_pct":      kw["skill_pct"],
            "has_exp":        kw["has_exp"],
            "has_edu":        kw["has_edu"],
            "has_extra":      kw["has_extra"],
            "matched_skills": kw["matched_skills"],
            "missing_skills": kw["missing_skills"],
            "explanation":    explanation,
            "radar":          get_radar(raw_text, job_role, tfidf)
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    for i,c in enumerate(candidates):
        c["rank"] = i+1
        c["recommendation"] = ("Highly Recommended" if c["score"]>=70
                               else "Recommended"   if c["score"]>=40
                               else "Not Recommended")

    result = {
        "job_role":      job_role,
        "candidates":    candidates,
        "top_candidate": candidates[0] if candidates else None,
        "total":         len(candidates),
        "recommended":   sum(1 for c in candidates if c["recommendation"]!="Not Recommended"),
        "avg_score":     round(sum(c["score"] for c in candidates)/len(candidates),1) if candidates else 0,
        "questions":     INTERVIEW_QUESTIONS.get(job_role, []),
        "skill_gap":     candidates[0]["missing_skills"] if candidates else []
    }

    # Save to database
    analysis_row = Analysis(
        job_role=job_role,
        total_resumes=len(candidates),
        top_candidate=candidates[0]["name"] if candidates else "",
        top_score=candidates[0]["score"] if candidates else 0,
        avg_score=result["avg_score"],
        recommended=result["recommended"]
    )
    db.session.add(analysis_row)
    db.session.flush()

    for c in candidates:
        db.session.add(Candidate(
            analysis_id=analysis_row.id,
            name=c["name"], score=c["score"],
            tfidf_score=c["tfidf_score"], ml_prob=c["ml_prob"],
            ml_suitable=c["ml_suitable"], skill_pct=c["skill_pct"],
            has_exp=c["has_exp"], has_edu=c["has_edu"],
            matched_skills=", ".join(c["matched_skills"]),
            recommendation=c["recommendation"], rank=c["rank"]
        ))
    db.session.commit()

    return jsonify(result)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    import json
    data       = json.loads(request.form.get('data'))
    candidates = data['candidates']
    buffer     = generate_pdf_report(data, candidates)
    return send_file(buffer, as_attachment=True,
                     download_name='Candidate_Report.pdf',
                     mimetype='application/pdf')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)