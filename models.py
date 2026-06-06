from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Analysis(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    job_role      = db.Column(db.String(100), nullable=False)
    total_resumes = db.Column(db.Integer)
    top_candidate = db.Column(db.String(100))
    top_score     = db.Column(db.Float)
    avg_score     = db.Column(db.Float)
    recommended   = db.Column(db.Integer)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    candidates    = db.relationship('Candidate', backref='analysis',
                                    lazy=True, cascade='all, delete-orphan')

class Candidate(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    analysis_id    = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=False)
    name           = db.Column(db.String(100))
    score          = db.Column(db.Float)
    tfidf_score    = db.Column(db.Float)
    ml_prob        = db.Column(db.Float)
    ml_suitable    = db.Column(db.Integer)
    skill_pct      = db.Column(db.Float)
    has_exp        = db.Column(db.Integer)
    has_edu        = db.Column(db.Integer)
    matched_skills = db.Column(db.String(300))
    recommendation = db.Column(db.String(50))
    rank           = db.Column(db.Integer)