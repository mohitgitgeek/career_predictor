import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import os


class CareerPredictor:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data_final.csv")

        self.data = pd.read_csv(data_path)
        self.feature_cols = [
            'O_score', 'C_score', 'E_score', 'A_score', 'N_score',
            'Numerical Aptitude', 'Spatial Aptitude', 'Perceptual Aptitude',
            'Abstract Reasoning', 'Verbal Reasoning'
        ]

        X = self.data[self.feature_cols].values
        y = self.data['Career'].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = DecisionTreeClassifier(
            max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42
        )
        self.model.fit(X_scaled, y)

        # ── Domain → Career mapping ──────────────────────────────────────────
        self.domain_career_mapping = {
            'technology': ['Software Developer', 'Web Developer', 'Data Scientist',
                           'IT Support Specialist', 'Database Administrator', 'Game Developer'],
            'healthcare': ['Nurse', 'Physician', 'Physical Therapist', 'Pediatric Nurse',
                           'Pediatrician', 'Chiropractor', 'Rehabilitation Counselor'],
            'finance':    ['Accountant', 'Financial Analyst', 'Financial Planner',
                           'Financial Advisor', 'Investment Banker', 'Tax Accountant'],
            'creative':   ['Graphic Designer', 'Artist', 'Fashion Designer',
                           'Interior Designer', 'Musician', 'Film Director'],
            'engineering':['Mechanical Engineer', 'Electrical Engineer', 'Civil Engineer',
                           'Aerospace Engineer', 'Biomedical Engineer', 'Robotics Engineer'],
        }

        # ── Company → Domain mapping (expanded) ─────────────────────────────
        self.company_domain_mapping = {
            'google': 'technology', 'microsoft': 'technology', 'apple': 'technology',
            'amazon': 'technology', 'meta': 'technology', 'netflix': 'technology',
            'uber': 'technology', 'airbnb': 'technology', 'twitter': 'technology',
            'nvidia': 'technology', 'intel': 'technology', 'salesforce': 'technology',
            'jpmorgan': 'finance', 'jp morgan': 'finance', 'goldman sachs': 'finance',
            'morgan stanley': 'finance', 'blackrock': 'finance', 'deloitte': 'finance',
            'pwc': 'finance', 'kpmg': 'finance', 'ey': 'finance', 'ernst & young': 'finance',
            'citi': 'finance', 'bank of america': 'finance', 'wells fargo': 'finance',
            'pfizer': 'healthcare', 'johnson & johnson': 'healthcare', 'mayo clinic': 'healthcare',
            'unitedhealth': 'healthcare', 'mckesson': 'healthcare', 'abbvie': 'healthcare',
            'boeing': 'engineering', 'lockheed martin': 'engineering', 'spacex': 'engineering',
            'ge': 'engineering', 'siemens': 'engineering', 'tesla': 'engineering',
            'caterpillar': 'engineering', 'honeywell': 'engineering',
            'disney': 'creative', 'sony': 'creative', 'pixar': 'creative',
            'adobe': 'creative', 'warnermedia': 'creative', 'universal': 'creative',
        }

        # ── Careers requiring strong coding ─────────────────────────────────
        self.coding_intensive_careers = [
            'Software Developer', 'Data Scientist', 'Game Developer',
            'Web Developer', 'Database Administrator',
        ]

        # ── Education level ordering & domain score maps ─────────────────────
        self.edu_levels = ['high_school', 'associate', 'bachelor', 'master', 'phd']

        self.education_requirements = {
            'technology': {
                'level_score_map': {'high_school': 0.15, 'associate': 0.35,
                                    'bachelor': 0.70, 'master': 0.90, 'phd': 1.00},
                'preferred_fields': ['stem'],
            },
            'finance': {
                'level_score_map': {'high_school': 0.10, 'associate': 0.30,
                                    'bachelor': 0.65, 'master': 0.90, 'phd': 0.85},
                'preferred_fields': ['stem', 'business'],
            },
            'healthcare': {
                'level_score_map': {'high_school': 0.05, 'associate': 0.25,
                                    'bachelor': 0.55, 'master': 0.80, 'phd': 1.00},
                'preferred_fields': ['medicine'],
            },
            'creative': {
                'level_score_map': {'high_school': 0.40, 'associate': 0.55,
                                    'bachelor': 0.75, 'master': 0.85, 'phd': 0.70},
                'preferred_fields': ['arts', 'stem'],
            },
            'engineering': {
                'level_score_map': {'high_school': 0.10, 'associate': 0.30,
                                    'bachelor': 0.70, 'master': 0.90, 'phd': 1.00},
                'preferred_fields': ['stem'],
            },
        }

        # ── Degree field relevance per domain (0–1) ──────────────────────────
        self.field_relevance = {
            'technology': {'stem': 1.0, 'business': 0.50, 'arts': 0.30, 'medicine': 0.30, 'law': 0.20, 'other': 0.30},
            'finance':    {'stem': 0.80, 'business': 1.00, 'arts': 0.20, 'medicine': 0.20, 'law': 0.55, 'other': 0.30},
            'healthcare': {'stem': 0.60, 'business': 0.20, 'arts': 0.10, 'medicine': 1.00, 'law': 0.20, 'other': 0.20},
            'creative':   {'stem': 0.50, 'business': 0.30, 'arts': 1.00, 'medicine': 0.10, 'law': 0.20, 'other': 0.40},
            'engineering':{'stem': 1.00, 'business': 0.30, 'arts': 0.20, 'medicine': 0.30, 'law': 0.10, 'other': 0.30},
        }

        # ── Relevant certifications per domain ───────────────────────────────
        self.domain_certifications = {
            'technology': ['aws', 'gcp', 'azure', 'docker', 'kubernetes', 'cisco', 'comptia',
                           'google cloud', 'microsoft', 'oracle', 'redhat', 'linux', 'security+',
                           'terraform', 'databricks', 'snowflake'],
            'finance':    ['cfa', 'frm', 'cpa', 'series 7', 'caia', 'cfp', 'acca', 'cma', 'cia',
                           'chartered', 'financial risk'],
            'healthcare': ['bls', 'acls', 'pals', 'rn', 'np', 'md', 'board certified', 'cpt',
                           'cna', 'emt', 'nclex', 'usmle'],
            'creative':   ['adobe', 'figma', 'sketch', 'cinema 4d', 'unreal', 'unity',
                           'autodesk', 'after effects', 'premiere'],
            'engineering':['pe', 'pmp', 'six sigma', 'autocad', 'solidworks', 'matlab', 'lean',
                           'iso', 'eit', 'asme', 'ansys'],
        }

        # ── Company-specific readiness requirements ──────────────────────────
        self.company_requirements = {
            'google':         {'dsa_min': 8, 'yoe_preferred': 2, 'edu_levels': ['bachelor','master','phd'], 'fields': ['stem'], 'key_certs': [], 'culture_fit': 'Data-driven, innovative, collaborative'},
            'microsoft':      {'dsa_min': 7, 'yoe_preferred': 2, 'edu_levels': ['bachelor','master','phd'], 'fields': ['stem'], 'key_certs': ['microsoft azure'], 'culture_fit': 'Growth mindset, inclusive, mission-driven'},
            'apple':          {'dsa_min': 8, 'yoe_preferred': 3, 'edu_levels': ['bachelor','master','phd'], 'fields': ['stem'], 'key_certs': [], 'culture_fit': 'Detail-oriented, quality-first, secretive'},
            'amazon':         {'dsa_min': 6, 'yoe_preferred': 2, 'edu_levels': ['bachelor','master'], 'fields': ['stem'], 'key_certs': ['aws'], 'culture_fit': 'Leadership principles, customer obsessed, high ownership'},
            'meta':           {'dsa_min': 8, 'yoe_preferred': 2, 'edu_levels': ['bachelor','master','phd'], 'fields': ['stem'], 'key_certs': [], 'culture_fit': 'Move fast, bold ideas, impact at scale'},
            'jpmorgan':       {'dsa_min': 0, 'yoe_preferred': 1, 'edu_levels': ['bachelor','master'], 'fields': ['stem','business'], 'key_certs': ['cfa','series 7'], 'culture_fit': 'Analytical, client-focused, compliance-aware'},
            'jp morgan':      {'dsa_min': 0, 'yoe_preferred': 1, 'edu_levels': ['bachelor','master'], 'fields': ['stem','business'], 'key_certs': ['cfa','series 7'], 'culture_fit': 'Analytical, client-focused, compliance-aware'},
            'goldman sachs':  {'dsa_min': 4, 'yoe_preferred': 1, 'edu_levels': ['bachelor','master'], 'fields': ['stem','business'], 'key_certs': ['cfa'], 'culture_fit': 'High performance, precision, competitive'},
            'morgan stanley': {'dsa_min': 0, 'yoe_preferred': 1, 'edu_levels': ['bachelor','master'], 'fields': ['stem','business'], 'key_certs': ['cfa','series 7'], 'culture_fit': 'Client relationships, trust, financial expertise'},
            'boeing':         {'dsa_min': 3, 'yoe_preferred': 2, 'edu_levels': ['bachelor','master'], 'fields': ['stem'], 'key_certs': ['pmp','autocad','solidworks'], 'culture_fit': 'Safety-first, precision, mission-critical'},
            'lockheed martin':{'dsa_min': 3, 'yoe_preferred': 2, 'edu_levels': ['bachelor','master'], 'fields': ['stem'], 'key_certs': ['pmp'], 'culture_fit': 'National security mindset, rigorous, classified clearance'},
            'spacex':         {'dsa_min': 5, 'yoe_preferred': 1, 'edu_levels': ['bachelor','master','phd'], 'fields': ['stem'], 'key_certs': [], 'culture_fit': 'High ownership, fast-paced, first-principles thinking'},
            'disney':         {'dsa_min': 0, 'yoe_preferred': 1, 'edu_levels': ['bachelor','master'], 'fields': ['arts','stem'], 'key_certs': ['adobe'], 'culture_fit': 'Storytelling, creativity, brand excellence'},
            'adobe':          {'dsa_min': 5, 'yoe_preferred': 2, 'edu_levels': ['bachelor','master'], 'fields': ['arts','stem'], 'key_certs': ['adobe'], 'culture_fit': 'Creative and technical blend, user-focused'},
        }

        # ── Per-domain viability scoring weights (must sum to 1.0) ───────────
        self.domain_weights = {
            'technology': {
                'model_confidence': 0.10, 'coding_skill': 0.12, 'dsa': 0.10,
                'portfolio': 0.10, 'experience': 0.08, 'education_level': 0.08,
                'degree_relevance': 0.06, 'domain_match': 0.06, 'company_match': 0.05,
                'certifications': 0.05, 'gpa': 0.05, 'internship': 0.06,
                'communication': 0.05, 'leadership': 0.04,
            },
            'finance': {
                'model_confidence': 0.10, 'coding_skill': 0.03, 'dsa': 0.02,
                'portfolio': 0.03, 'experience': 0.10, 'education_level': 0.12,
                'degree_relevance': 0.10, 'domain_match': 0.08, 'company_match': 0.05,
                'certifications': 0.15, 'gpa': 0.09, 'internship': 0.07,
                'communication': 0.04, 'leadership': 0.02,
            },
            'healthcare': {
                'model_confidence': 0.10, 'coding_skill': 0.01, 'dsa': 0.01,
                'portfolio': 0.02, 'experience': 0.10, 'education_level': 0.18,
                'degree_relevance': 0.14, 'domain_match': 0.08, 'company_match': 0.04,
                'certifications': 0.14, 'gpa': 0.09, 'internship': 0.05,
                'communication': 0.03, 'leadership': 0.01,
            },
            'creative': {
                'model_confidence': 0.10, 'coding_skill': 0.02, 'dsa': 0.01,
                'portfolio': 0.22, 'experience': 0.10, 'education_level': 0.08,
                'degree_relevance': 0.08, 'domain_match': 0.10, 'company_match': 0.05,
                'certifications': 0.07, 'gpa': 0.03, 'internship': 0.06,
                'communication': 0.05, 'leadership': 0.03,
            },
            'engineering': {
                'model_confidence': 0.10, 'coding_skill': 0.04, 'dsa': 0.03,
                'portfolio': 0.09, 'experience': 0.10, 'education_level': 0.14,
                'degree_relevance': 0.12, 'domain_match': 0.08, 'company_match': 0.05,
                'certifications': 0.09, 'gpa': 0.08, 'internship': 0.05,
                'communication': 0.02, 'leadership': 0.01,
            },
            'default': {
                'model_confidence': 0.12, 'coding_skill': 0.06, 'dsa': 0.05,
                'portfolio': 0.08, 'experience': 0.10, 'education_level': 0.10,
                'degree_relevance': 0.08, 'domain_match': 0.08, 'company_match': 0.05,
                'certifications': 0.08, 'gpa': 0.07, 'internship': 0.06,
                'communication': 0.05, 'leadership': 0.02,
            },
        }

        self.alternative_companies = {
            'low_coding_rank': [
                'Startups focusing on product development',
                'Non-tech companies with tech departments',
                'Government tech agencies',
                'Healthcare IT companies',
                'Educational technology companies',
            ],
            'low_yoe': [
                'Startups with training programs',
                'Companies with internship-to-job pipelines',
                'Organizations with strong mentorship programs',
                'Large corporations with structured entry programs',
                'Technology service & consulting providers',
            ],
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _education_level_score(self, education_level, domain):
        req = self.education_requirements.get(domain, self.education_requirements['technology'])
        return req['level_score_map'].get(education_level, 0.50)

    def _degree_relevance_score(self, degree_field, domain):
        return self.field_relevance.get(domain, {}).get(degree_field, 0.30)

    def _certifications_score(self, certifications, domain):
        if not certifications:
            return 0.0
        relevant = self.domain_certifications.get(domain, [])
        certs_lower = [c.lower().strip() for c in certifications if c.strip()]
        count = sum(
            1 for cert in certs_lower
            if any(r in cert or cert in r for r in relevant)
        )
        return min(1.0, count / 2.0)

    def _portfolio_score(self, github_level, open_source, hackathons):
        base = {'none': 0.0, 'low': 0.25, 'medium': 0.60, 'high': 1.0}.get(github_level, 0.0)
        bonus = 0.15 if open_source else 0.0
        hackathon_bonus = min(0.15, hackathons * 0.05)
        return min(1.0, base + bonus + hackathon_bonus)

    def _leadership_score(self, leadership_level):
        return {'none': 0.0, 'some': 0.5, 'extensive': 1.0}.get(leadership_level, 0.0)

    def _is_low_coding_rank(self, rank):
        if not rank or rank.strip() == '':
            return True
        r = rank.lower()
        if 'codechef' in r:
            return not any(x in r for x in ['5*', '6*', '7*'])
        if 'codeforces' in r:
            return not any(x in r for x in ['expert', 'master', 'grandmaster'])
        if 'leetcode' in r:
            try:
                rating = int(''.join(filter(str.isdigit, r)))
                return rating < 2000
            except Exception:
                return True
        if 'hackerrank' in r:
            return not any(x in r for x in ['gold', 'platinum'])
        return True

    def _coding_skill_score(self, competitive_coding_rank, dsa_score):
        rank_score = 0.0 if self._is_low_coding_rank(competitive_coding_rank) else 1.0
        return rank_score * 0.40 + (dsa_score / 10.0) * 0.60

    def _get_career_domain(self, careers):
        counts = {}
        for career in careers[:3]:
            for domain, clist in self.domain_career_mapping.items():
                if career in clist:
                    counts[domain] = counts.get(domain, 0) + 1
        return max(counts, key=counts.get) if counts else 'default'

    def _identify_skill_gaps(self, factors, domain, education_level, degree_field,
                              certifications, target_company, dsa_score, coding_rank):
        gaps = []
        threshold = 0.50

        if factors['education_level'] < threshold:
            gaps.append({
                "area": "Education Level",
                "detail": f"Your {education_level.replace('_', ' ').title()} level is below what {domain} companies typically require. Advancing your degree significantly improves hiring chances.",
                "icon": "🎓"
            })

        if factors['degree_relevance'] < threshold:
            pref = self.education_requirements.get(domain, {}).get('preferred_fields', ['relevant field'])
            gaps.append({
                "area": "Degree Relevance",
                "detail": f"Your degree field has limited alignment with {domain} roles. Bridge the gap with targeted certifications, bootcamps, or online courses in {'/'.join(pref)}.",
                "icon": "📚"
            })

        if factors['gpa'] < 0.70:
            gaps.append({
                "area": "Academic Performance",
                "detail": "A GPA of 3.0+ is often a screening threshold at top companies. Compensate with strong projects, internships, and certifications.",
                "icon": "📊"
            })

        if factors['coding_skill'] < threshold and domain == 'technology':
            gaps.append({
                "area": "Coding & DSA",
                "detail": f"DSA score {dsa_score}/10 needs improvement for tech roles. FAANG-tier companies expect consistent LeetCode practice and a rating above 2000.",
                "icon": "💻"
            })

        if factors['certifications'] < threshold:
            relevant = self.domain_certifications.get(domain, [])[:4]
            gaps.append({
                "area": "Certifications",
                "detail": f"Lacking industry-recognized {domain} certifications. Top picks: {', '.join(c.upper() for c in relevant)}.",
                "icon": "🏅"
            })

        if factors['portfolio'] < threshold:
            detail = ("Low GitHub activity detected. Build 3–5 end-to-end projects and contribute to open source."
                      if domain == 'technology' else
                      "A strong portfolio is critical for creative/engineering roles. Document and showcase your best work.")
            gaps.append({"area": "Portfolio / Projects", "detail": detail, "icon": "🗂️"})

        if factors['experience'] < 0.40:
            gaps.append({
                "area": "Work Experience",
                "detail": "Limited professional experience. Internships, contract work, and freelance projects can substitute early on.",
                "icon": "💼"
            })

        if factors['internship'] < 0.33:
            gaps.append({
                "area": "Internship Experience",
                "detail": "Companies strongly prefer candidates with at least 1–2 relevant internships. Apply aggressively each recruitment cycle.",
                "icon": "🏢"
            })

        if factors['communication'] < threshold:
            gaps.append({
                "area": "Communication Skills",
                "detail": "Recruiter surveys consistently rank communication as a top hiring factor. Practice structured responses (STAR method) and public speaking.",
                "icon": "🗣️"
            })

        if factors['leadership'] < 0.40:
            gaps.append({
                "area": "Leadership Experience",
                "detail": "Even early-career candidates benefit from demonstrable leadership — team leads, club officers, open-source maintainers.",
                "icon": "🌟"
            })

        return gaps

    def _generate_action_plan(self, factors, domain, target_company, certifications,
                               education_level, yoe, dsa_score):
        plan = []

        if domain == 'technology':
            if factors['dsa'] < 0.70:
                plan.append("Solve 3 LeetCode problems daily (Easy→Medium→Hard). Aim for 200+ solved problems and a contest rating above 2000 within 6 months.")
            if factors['portfolio'] < 0.60:
                plan.append("Build 3 GitHub projects: one full-stack web app, one ML/data project, one open-source contribution. Include READMEs and live demos.")
            if factors['certifications'] < 0.50:
                plan.append("Earn AWS Solutions Architect Associate or Google Professional Cloud Developer certification — both are highly valued by top tech employers.")
            if factors['experience'] < 0.40:
                plan.append("Apply to internship programs (Google STEP, Microsoft Explore, Amazon SDE Intern) or contribute to large open-source projects.")

        elif domain == 'finance':
            if factors['certifications'] < 0.50:
                plan.append("Begin CFA Level 1 preparation (pass rate ~40%). It's the most globally recognised finance credential and opens doors at bulge-bracket banks.")
            if factors['gpa'] < 0.75:
                plan.append("Supplement a lower GPA by winning finance case competitions (CFA Institute, CFA Society) and completing financial modelling courses on Wall Street Prep.")
            if factors['internship'] < 0.50:
                plan.append("Secure a summer analyst internship at a bank or asset manager — nearly all full-time finance roles are filled via return-offer pipelines.")
            if factors['communication'] < 0.60:
                plan.append("Practice finance interviews: fit questions (STAR method), valuation technicals, and market walk-throughs. Use platforms like Mergers & Inquisitions.")

        elif domain == 'healthcare':
            if factors['education_level'] < 0.70:
                plan.append("Advance your education — clinical healthcare careers require a bachelor's at minimum, and specialized roles (NP, MD) require graduate or doctoral degrees.")
            if factors['certifications'] < 0.50:
                plan.append("Obtain BLS/ACLS certifications immediately (low cost, quick win). Then pursue role-specific licensure: NCLEX-RN for nursing, USMLE for physicians.")
            if factors['internship'] < 0.40:
                plan.append("Volunteer or shadow in a clinical setting. Hospital volunteer programs and clinical rotations directly translate to hiring preference.")

        elif domain == 'creative':
            if factors['portfolio'] < 0.70:
                plan.append("Create a professional portfolio website (Behance, Dribbble, personal domain) featuring 8–10 of your best works with process documentation.")
            if factors['certifications'] < 0.50:
                plan.append("Earn Adobe Creative Cloud certifications or Figma proficiency — studios and agencies use these as baseline filters.")
            if factors['experience'] < 0.40:
                plan.append("Take on freelance projects via Upwork or Fiverr to build a paid client portfolio while still in early career.")

        elif domain == 'engineering':
            if factors['certifications'] < 0.50:
                plan.append("Pursue a Professional Engineer (PE) license or PMP certification. Six Sigma Green Belt is also highly valued in manufacturing environments.")
            if factors['portfolio'] < 0.50:
                plan.append("Document engineering projects with CAD files, simulation outputs, test results, and technical write-ups. Upload to GitHub or a personal site.")
            if factors['gpa'] < 0.75:
                plan.append("Supplement grades with research papers, patent applications, or competition wins (Formula SAE, NASA design challenges).")

        # Universal actions
        if factors['communication'] < 0.60:
            plan.append("Join Toastmasters or take a structured public speaking course. Practice mock interviews weekly using Pramp or Interviewing.io.")
        if factors['leadership'] < 0.50:
            plan.append("Seek leadership roles in student clubs, open-source projects, or volunteer organisations. Demonstrate impact with measurable outcomes.")
        if factors['internship'] < 0.33 and yoe < 2:
            plan.append("Prioritise internships over early full-time roles — they provide mentorship, network access, and the highest conversion rates to permanent offers.")

        if target_company and target_company.lower() in self.company_requirements:
            req = self.company_requirements[target_company.lower()]
            if req.get('key_certs'):
                plan.append(f"For {target_company.title()} specifically: obtain {', '.join(c.upper() for c in req['key_certs'])} — these are explicitly valued during screening.")
            if req.get('dsa_min', 0) > 0 and dsa_score < req['dsa_min']:
                plan.append(f"{target_company.title()} interviewers expect DSA proficiency of {req['dsa_min']}/10+. Focus on graph algorithms, dynamic programming, and system design patterns.")

        return plan[:6]

    def _assess_company_readiness(self, company_key, factors, education_level, degree_field,
                                   certifications, dsa_score, yoe):
        req = self.company_requirements.get(company_key)
        if not req:
            return None

        checks = []
        score_total = 0.0
        weight_total = 0.0

        # Education level
        passes = education_level in req.get('edu_levels', [])
        check_score = 1.0 if passes else 0.2
        checks.append({"check": "Education Level", "status": passes,
                        "note": f"Required: {' / '.join(req.get('edu_levels', []))}",
                        "score": check_score})
        score_total += check_score * 25
        weight_total += 25

        # Degree field
        passes = degree_field in req.get('fields', [])
        check_score = 1.0 if passes else 0.40
        checks.append({"check": "Degree Field", "status": passes,
                        "note": f"Preferred: {' / '.join(req.get('fields', []))}",
                        "score": check_score})
        score_total += check_score * 20
        weight_total += 20

        # DSA
        dsa_min = req.get('dsa_min', 0)
        if dsa_min > 0:
            passes = dsa_score >= dsa_min
            check_score = 1.0 if passes else (dsa_score / dsa_min) * 0.60
            checks.append({"check": "DSA / Algorithmic Skill", "status": passes,
                            "note": f"Minimum expected: {dsa_min}/10",
                            "score": check_score})
            score_total += check_score * 25
            weight_total += 25

        # Experience
        yoe_pref = req.get('yoe_preferred', 0)
        passes = yoe >= yoe_pref
        check_score = 1.0 if passes else (yoe / max(yoe_pref, 1)) * 0.70
        checks.append({"check": "Years of Experience", "status": passes,
                        "note": f"Preferred: {yoe_pref}+ years",
                        "score": check_score})
        score_total += check_score * 20
        weight_total += 20

        # Key certifications
        if req.get('key_certs'):
            certs_lower = [c.lower().strip() for c in (certifications or [])]
            has_any = any(
                any(k in c or c in k for c in certs_lower)
                for k in req['key_certs']
            )
            check_score = 1.0 if has_any else 0.20
            checks.append({"check": "Key Certifications", "status": has_any,
                            "note": f"Valued: {', '.join(c.upper() for c in req['key_certs'])}",
                            "score": check_score})
            score_total += check_score * 10
            weight_total += 10

        readiness_pct = round((score_total / weight_total) * 100) if weight_total else 0

        return {
            'company': company_key.title(),
            'readiness_percent': readiness_pct,
            'checks': checks,
            'culture_fit': req.get('culture_fit', ''),
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_career(
        self,
        features,
        yoe=0,
        competitive_coding_rank=None,
        favorite_domain=None,
        familiar_disciplines=None,
        target_company=None,
        # ── New company-readiness criteria ──
        education_level='bachelor',
        degree_field='stem',
        gpa=3.0,
        certifications=None,
        github_level='low',
        internships=0,
        communication_score=5.0,
        leadership_level='none',
        dsa_score=5.0,
        system_design_score=5.0,
        open_source=False,
        hackathons=0,
    ):
        # 1. ML prediction
        features_scaled = self.scaler.transform([features])
        proba = self.model.predict_proba(features_scaled)[0]
        top_idx = np.argsort(proba)[-5:][::-1]
        careers = self.model.classes_[top_idx]
        probabilities = proba[top_idx]

        top_careers = [{"career": c, "probability": float(p)} for c, p in zip(careers, probabilities)]
        predicted_names = [c["career"] for c in top_careers]

        # 2. Determine effective domain
        if favorite_domain and favorite_domain.lower() in self.domain_career_mapping:
            effective_domain = favorite_domain.lower()
        elif target_company and target_company.lower() in self.company_domain_mapping:
            effective_domain = self.company_domain_mapping[target_company.lower()]
        else:
            effective_domain = self._get_career_domain(predicted_names)

        weights = self.domain_weights.get(effective_domain, self.domain_weights['default'])

        # 3. Domain / company match scores
        domain_match = 0.0
        if favorite_domain and favorite_domain.lower() in self.domain_career_mapping:
            dcareers = set(self.domain_career_mapping[favorite_domain.lower()])
            domain_match = len([c for c in predicted_names if c in dcareers]) / max(1, min(len(predicted_names), len(dcareers)))

        company_match = 0.0
        if target_company and target_company.lower() in self.company_domain_mapping:
            cdomain = self.company_domain_mapping[target_company.lower()]
            dcareers = set(self.domain_career_mapping.get(cdomain, []))
            company_match = len([c for c in predicted_names if c in dcareers]) / max(1, min(len(predicted_names), len(dcareers)))

        # 4. All readiness factors (0.0 – 1.0)
        factors = {
            'model_confidence':  float(top_careers[0]['probability']),
            'coding_skill':      self._coding_skill_score(competitive_coding_rank, dsa_score),
            'dsa':               dsa_score / 10.0,
            'portfolio':         self._portfolio_score(github_level, open_source, hackathons),
            'experience':        min(1.0, yoe / 5.0),
            'education_level':   self._education_level_score(education_level, effective_domain),
            'degree_relevance':  self._degree_relevance_score(degree_field, effective_domain),
            'domain_match':      domain_match,
            'company_match':     company_match,
            'certifications':    self._certifications_score(certifications, effective_domain),
            'gpa':               min(1.0, gpa / 4.0),
            'internship':        min(1.0, internships / 3.0),
            'communication':     communication_score / 10.0,
            'leadership':        self._leadership_score(leadership_level),
        }

        # 5. Weighted viability score
        viability_score = sum(factors[k] * weights.get(k, 0) for k in factors)

        if viability_score >= 0.70:
            viability_level = "High"
            explanation = "Your profile strongly aligns with industry readiness standards."
        elif viability_score >= 0.45:
            viability_level = "Moderate"
            explanation = "Your profile shows good potential with some addressable gaps."
        else:
            viability_level = "Low"
            explanation = "Your profile has significant gaps vs. what companies screen for."

        # 6. Alternative companies
        tech_recommended = any(c["career"] in self.coding_intensive_careers for c in top_careers[:3])
        alternative_companies = []
        if tech_recommended:
            if self._is_low_coding_rank(competitive_coding_rank):
                alternative_companies.extend(self.alternative_companies['low_coding_rank'])
            if yoe < 2:
                alternative_companies.extend(self.alternative_companies['low_yoe'])

        # 7. Skill gaps + action plan
        skill_gaps = self._identify_skill_gaps(
            factors, effective_domain, education_level, degree_field,
            certifications, target_company, dsa_score, competitive_coding_rank
        )
        action_plan = self._generate_action_plan(
            factors, effective_domain, target_company,
            certifications, education_level, yoe, dsa_score
        )

        # 8. Company-specific readiness
        company_readiness = None
        company_key = (target_company or '').lower()
        if company_key in self.company_requirements:
            company_readiness = self._assess_company_readiness(
                company_key, factors, education_level, degree_field,
                certifications, dsa_score, yoe
            )

        return {
            'predicted_careers':   predicted_names,
            'career_probabilities':[c['probability'] for c in top_careers],
            'viability_assessment': {
                'score':       viability_score,
                'level':       viability_level,
                'explanation': explanation,
                'factors':     factors,
            },
            'readiness_factors':   factors,
            'skill_gaps':          skill_gaps,
            'action_plan':         action_plan,
            'company_readiness':   company_readiness,
            'alternative_companies': alternative_companies,
            'effective_domain':    effective_domain,
        }
