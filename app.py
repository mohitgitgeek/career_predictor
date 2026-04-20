from flask import Flask, render_template, request
from model import CareerPredictor

app = Flask(__name__)
predictor = CareerPredictor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.form

        # ── Original personality + aptitude ─────────────────────────────────
        features = [
            float(f['o_score']), float(f['c_score']),
            float(f['e_score']), float(f['a_score']),
            float(f['n_score']), float(f['numerical']),
            float(f['spatial']), float(f['perceptual']),
            float(f['abstract']), float(f['verbal']),
        ]

        yoe          = int(f.get('yoe', 0) or 0)
        coding_rank  = f.get('coding_rank', '').strip() or None
        fav_domain   = f.get('favorite_domain', '').strip() or None
        target_co    = f.get('target_company', '').strip() or None
        disciplines_raw = f.get('disciplines', '')
        disciplines  = [d.strip() for d in disciplines_raw.split(',')] if disciplines_raw.strip() else None

        # ── New readiness criteria ───────────────────────────────────────────
        education_level   = f.get('education_level', 'bachelor')
        degree_field      = f.get('degree_field', 'stem')
        gpa               = float(f.get('gpa', 3.0) or 3.0)

        certs_raw         = f.get('certifications', '')
        certifications    = [c.strip() for c in certs_raw.split(',') if c.strip()] if certs_raw else []

        github_level      = f.get('github_level', 'low')
        internships       = int(f.get('internships', 0) or 0)
        communication_score = float(f.get('communication_score', 5) or 5)
        leadership_level  = f.get('leadership_level', 'none')
        dsa_score         = float(f.get('dsa_score', 5) or 5)
        system_design_score = float(f.get('system_design_score', 5) or 5)
        open_source       = 'open_source' in f
        hackathons        = int(f.get('hackathons', 0) or 0)

        result = predictor.predict_career(
            features=features,
            yoe=yoe,
            competitive_coding_rank=coding_rank,
            favorite_domain=fav_domain,
            familiar_disciplines=disciplines,
            target_company=target_co,
            education_level=education_level,
            degree_field=degree_field,
            gpa=gpa,
            certifications=certifications,
            github_level=github_level,
            internships=internships,
            communication_score=communication_score,
            leadership_level=leadership_level,
            dsa_score=dsa_score,
            system_design_score=system_design_score,
            open_source=open_source,
            hackathons=hackathons,
        )

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('index.html', error=f"Error processing your request: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
