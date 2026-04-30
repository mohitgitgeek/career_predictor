/* ═══════════════════════════════════════════════════════════════
   CareerPredictor — JavaScript prediction engine
   Mirrors the Python logic in model.py exactly.
════════════════════════════════════════════════════════════════ */

class CareerPredictor {
  constructor(modelData) {
    this.scaler = modelData.scaler;   // { mean: [...], scale: [...] }
    this.tree   = modelData.tree;     // sklearn decision-tree internals exported to JSON

    // ── Domain → Career mapping ──────────────────────────────────
    this.domainCareerMapping = {
      technology:  ['Software Developer', 'Web Developer', 'Data Scientist',
                    'IT Support Specialist', 'Database Administrator', 'Game Developer'],
      healthcare:  ['Nurse', 'Physician', 'Physical Therapist', 'Pediatric Nurse',
                    'Pediatrician', 'Chiropractor', 'Rehabilitation Counselor'],
      finance:     ['Accountant', 'Financial Analyst', 'Financial Planner',
                    'Financial Advisor', 'Investment Banker', 'Tax Accountant'],
      creative:    ['Graphic Designer', 'Artist', 'Fashion Designer',
                    'Interior Designer', 'Musician', 'Film Director'],
      engineering: ['Mechanical Engineer', 'Electrical Engineer', 'Civil Engineer',
                    'Aerospace Engineer', 'Biomedical Engineer', 'Robotics Engineer'],
    };

    // ── Company → Domain mapping ─────────────────────────────────
    this.companyDomainMapping = {
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
    };

    this.codingIntensiveCareers = [
      'Software Developer', 'Data Scientist', 'Game Developer',
      'Web Developer', 'Database Administrator',
    ];

    this.eduLevels = ['high_school', 'associate', 'bachelor', 'master', 'phd'];

    this.educationRequirements = {
      technology: {
        level_score_map: { high_school: 0.15, associate: 0.35, bachelor: 0.70, master: 0.90, phd: 1.00 },
        preferred_fields: ['stem'],
      },
      finance: {
        level_score_map: { high_school: 0.10, associate: 0.30, bachelor: 0.65, master: 0.90, phd: 0.85 },
        preferred_fields: ['stem', 'business'],
      },
      healthcare: {
        level_score_map: { high_school: 0.05, associate: 0.25, bachelor: 0.55, master: 0.80, phd: 1.00 },
        preferred_fields: ['medicine'],
      },
      creative: {
        level_score_map: { high_school: 0.40, associate: 0.55, bachelor: 0.75, master: 0.85, phd: 0.70 },
        preferred_fields: ['arts', 'stem'],
      },
      engineering: {
        level_score_map: { high_school: 0.10, associate: 0.30, bachelor: 0.70, master: 0.90, phd: 1.00 },
        preferred_fields: ['stem'],
      },
    };

    this.fieldRelevance = {
      technology:  { stem: 1.0, business: 0.50, arts: 0.30, medicine: 0.30, law: 0.20, other: 0.30 },
      finance:     { stem: 0.80, business: 1.00, arts: 0.20, medicine: 0.20, law: 0.55, other: 0.30 },
      healthcare:  { stem: 0.60, business: 0.20, arts: 0.10, medicine: 1.00, law: 0.20, other: 0.20 },
      creative:    { stem: 0.50, business: 0.30, arts: 1.00, medicine: 0.10, law: 0.20, other: 0.40 },
      engineering: { stem: 1.00, business: 0.30, arts: 0.20, medicine: 0.30, law: 0.10, other: 0.30 },
    };

    this.domainCertifications = {
      technology:  ['aws', 'gcp', 'azure', 'docker', 'kubernetes', 'cisco', 'comptia',
                    'google cloud', 'microsoft', 'oracle', 'redhat', 'linux', 'security+',
                    'terraform', 'databricks', 'snowflake'],
      finance:     ['cfa', 'frm', 'cpa', 'series 7', 'caia', 'cfp', 'acca', 'cma', 'cia',
                    'chartered', 'financial risk'],
      healthcare:  ['bls', 'acls', 'pals', 'rn', 'np', 'md', 'board certified', 'cpt',
                    'cna', 'emt', 'nclex', 'usmle'],
      creative:    ['adobe', 'figma', 'sketch', 'cinema 4d', 'unreal', 'unity',
                    'autodesk', 'after effects', 'premiere'],
      engineering: ['pe', 'pmp', 'six sigma', 'autocad', 'solidworks', 'matlab', 'lean',
                    'iso', 'eit', 'asme', 'ansys'],
    };

    this.companyRequirements = {
      'google':          { dsa_min: 8, yoe_preferred: 2, edu_levels: ['bachelor','master','phd'], fields: ['stem'], key_certs: [], culture_fit: 'Data-driven, innovative, collaborative' },
      'microsoft':       { dsa_min: 7, yoe_preferred: 2, edu_levels: ['bachelor','master','phd'], fields: ['stem'], key_certs: ['microsoft azure'], culture_fit: 'Growth mindset, inclusive, mission-driven' },
      'apple':           { dsa_min: 8, yoe_preferred: 3, edu_levels: ['bachelor','master','phd'], fields: ['stem'], key_certs: [], culture_fit: 'Detail-oriented, quality-first, secretive' },
      'amazon':          { dsa_min: 6, yoe_preferred: 2, edu_levels: ['bachelor','master'], fields: ['stem'], key_certs: ['aws'], culture_fit: 'Leadership principles, customer obsessed, high ownership' },
      'meta':            { dsa_min: 8, yoe_preferred: 2, edu_levels: ['bachelor','master','phd'], fields: ['stem'], key_certs: [], culture_fit: 'Move fast, bold ideas, impact at scale' },
      'jpmorgan':        { dsa_min: 0, yoe_preferred: 1, edu_levels: ['bachelor','master'], fields: ['stem','business'], key_certs: ['cfa','series 7'], culture_fit: 'Analytical, client-focused, compliance-aware' },
      'jp morgan':       { dsa_min: 0, yoe_preferred: 1, edu_levels: ['bachelor','master'], fields: ['stem','business'], key_certs: ['cfa','series 7'], culture_fit: 'Analytical, client-focused, compliance-aware' },
      'goldman sachs':   { dsa_min: 4, yoe_preferred: 1, edu_levels: ['bachelor','master'], fields: ['stem','business'], key_certs: ['cfa'], culture_fit: 'High performance, precision, competitive' },
      'morgan stanley':  { dsa_min: 0, yoe_preferred: 1, edu_levels: ['bachelor','master'], fields: ['stem','business'], key_certs: ['cfa','series 7'], culture_fit: 'Client relationships, trust, financial expertise' },
      'boeing':          { dsa_min: 3, yoe_preferred: 2, edu_levels: ['bachelor','master'], fields: ['stem'], key_certs: ['pmp','autocad','solidworks'], culture_fit: 'Safety-first, precision, mission-critical' },
      'lockheed martin': { dsa_min: 3, yoe_preferred: 2, edu_levels: ['bachelor','master'], fields: ['stem'], key_certs: ['pmp'], culture_fit: 'National security mindset, rigorous, classified clearance' },
      'spacex':          { dsa_min: 5, yoe_preferred: 1, edu_levels: ['bachelor','master','phd'], fields: ['stem'], key_certs: [], culture_fit: 'High ownership, fast-paced, first-principles thinking' },
      'disney':          { dsa_min: 0, yoe_preferred: 1, edu_levels: ['bachelor','master'], fields: ['arts','stem'], key_certs: ['adobe'], culture_fit: 'Storytelling, creativity, brand excellence' },
      'adobe':           { dsa_min: 5, yoe_preferred: 2, edu_levels: ['bachelor','master'], fields: ['arts','stem'], key_certs: ['adobe'], culture_fit: 'Creative and technical blend, user-focused' },
    };

    this.domainWeights = {
      technology: {
        model_confidence: 0.10, coding_skill: 0.12, dsa: 0.10,
        portfolio: 0.10, experience: 0.08, education_level: 0.08,
        degree_relevance: 0.06, domain_match: 0.06, company_match: 0.05,
        certifications: 0.05, gpa: 0.05, internship: 0.06,
        communication: 0.05, leadership: 0.04,
      },
      finance: {
        model_confidence: 0.10, coding_skill: 0.03, dsa: 0.02,
        portfolio: 0.03, experience: 0.10, education_level: 0.12,
        degree_relevance: 0.10, domain_match: 0.08, company_match: 0.05,
        certifications: 0.15, gpa: 0.09, internship: 0.07,
        communication: 0.04, leadership: 0.02,
      },
      healthcare: {
        model_confidence: 0.10, coding_skill: 0.01, dsa: 0.01,
        portfolio: 0.02, experience: 0.10, education_level: 0.18,
        degree_relevance: 0.14, domain_match: 0.08, company_match: 0.04,
        certifications: 0.14, gpa: 0.09, internship: 0.05,
        communication: 0.03, leadership: 0.01,
      },
      creative: {
        model_confidence: 0.10, coding_skill: 0.02, dsa: 0.01,
        portfolio: 0.22, experience: 0.10, education_level: 0.08,
        degree_relevance: 0.08, domain_match: 0.10, company_match: 0.05,
        certifications: 0.07, gpa: 0.03, internship: 0.06,
        communication: 0.05, leadership: 0.03,
      },
      engineering: {
        model_confidence: 0.10, coding_skill: 0.04, dsa: 0.03,
        portfolio: 0.09, experience: 0.10, education_level: 0.14,
        degree_relevance: 0.12, domain_match: 0.08, company_match: 0.05,
        certifications: 0.09, gpa: 0.08, internship: 0.05,
        communication: 0.02, leadership: 0.01,
      },
      default: {
        model_confidence: 0.12, coding_skill: 0.06, dsa: 0.05,
        portfolio: 0.08, experience: 0.10, education_level: 0.10,
        degree_relevance: 0.08, domain_match: 0.08, company_match: 0.05,
        certifications: 0.08, gpa: 0.07, internship: 0.06,
        communication: 0.05, leadership: 0.02,
      },
    };

    this.alternativeCompanies = {
      low_coding_rank: [
        'Startups focusing on product development',
        'Non-tech companies with tech departments',
        'Government tech agencies',
        'Healthcare IT companies',
        'Educational technology companies',
      ],
      low_yoe: [
        'Startups with training programs',
        'Companies with internship-to-job pipelines',
        'Organizations with strong mentorship programs',
        'Large corporations with structured entry programs',
        'Technology service & consulting providers',
      ],
    };
  }

  // ── sklearn StandardScaler ────────────────────────────────────
  _scale(features) {
    return features.map((v, i) => (v - this.scaler.mean[i]) / this.scaler.scale[i]);
  }

  // ── sklearn DecisionTreeClassifier predict_proba ──────────────
  _predictProba(featuresScaled) {
    const t = this.tree;
    let node = 0;
    while (t.children_left[node] !== -1) {
      const feat = t.feature[node];
      if (featuresScaled[feat] <= t.threshold[node]) {
        node = t.children_left[node];
      } else {
        node = t.children_right[node];
      }
    }
    // t.value[node] is shape [1, n_classes] — normalise to probabilities
    const counts = t.value[node][0];
    const total  = counts.reduce((a, b) => a + b, 0);
    return counts.map(c => c / total);
  }

  // ── Helpers ───────────────────────────────────────────────────
  _educationLevelScore(educationLevel, domain) {
    const req = this.educationRequirements[domain] || this.educationRequirements['technology'];
    return req.level_score_map[educationLevel] ?? 0.50;
  }

  _degreeRelevanceScore(degreeField, domain) {
    return (this.fieldRelevance[domain] || {})[degreeField] ?? 0.30;
  }

  _certificationsScore(certifications, domain) {
    if (!certifications || certifications.length === 0) return 0.0;
    const relevant = this.domainCertifications[domain] || [];
    const certsLower = certifications.map(c => c.toLowerCase().trim()).filter(Boolean);
    const count = certsLower.reduce((acc, cert) => {
      return acc + (relevant.some(r => r.includes(cert) || cert.includes(r)) ? 1 : 0);
    }, 0);
    return Math.min(1.0, count / 2.0);
  }

  _portfolioScore(githubLevel, openSource, hackathons) {
    const base = { none: 0.0, low: 0.25, medium: 0.60, high: 1.0 }[githubLevel] ?? 0.0;
    const bonus = openSource ? 0.15 : 0.0;
    const hackathonBonus = Math.min(0.15, hackathons * 0.05);
    return Math.min(1.0, base + bonus + hackathonBonus);
  }

  _leadershipScore(leadershipLevel) {
    return { none: 0.0, some: 0.5, extensive: 1.0 }[leadershipLevel] ?? 0.0;
  }

  _isLowCodingRank(rank) {
    if (!rank || rank.trim() === '') return true;
    const r = rank.toLowerCase();
    if (r.includes('codechef'))    return !['5*','6*','7*'].some(x => r.includes(x));
    if (r.includes('codeforces'))  return !['expert','master','grandmaster'].some(x => r.includes(x));
    if (r.includes('leetcode')) {
      const digits = r.replace(/\D/g, '');
      return digits ? parseInt(digits, 10) < 2000 : true;
    }
    if (r.includes('hackerrank')) return !['gold','platinum'].some(x => r.includes(x));
    return true;
  }

  _codingSkillScore(codingRank, dsaScore) {
    const rankScore = this._isLowCodingRank(codingRank) ? 0.0 : 1.0;
    return rankScore * 0.40 + (dsaScore / 10.0) * 0.60;
  }

  _getCareerDomain(careers) {
    const counts = {};
    careers.slice(0, 3).forEach(career => {
      for (const [domain, list] of Object.entries(this.domainCareerMapping)) {
        if (list.includes(career)) counts[domain] = (counts[domain] || 0) + 1;
      }
    });
    if (Object.keys(counts).length === 0) return 'default';
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  }

  _identifySkillGaps(factors, domain, educationLevel, degreeField,
                     certifications, targetCompany, dsaScore) {
    const gaps = [];
    const threshold = 0.50;

    if (factors.education_level < threshold)
      gaps.push({ area: 'Education Level', icon: '🎓',
        detail: `Your ${educationLevel.replace('_',' ')} level is below what ${domain} companies typically require. Advancing your degree significantly improves hiring chances.` });

    if (factors.degree_relevance < threshold) {
      const pref = (this.educationRequirements[domain] || {}).preferred_fields || ['relevant field'];
      gaps.push({ area: 'Degree Relevance', icon: '📚',
        detail: `Your degree field has limited alignment with ${domain} roles. Bridge the gap with targeted certifications, bootcamps, or online courses in ${pref.join('/')}.` });
    }

    if (factors.gpa < 0.70)
      gaps.push({ area: 'Academic Performance', icon: '📊',
        detail: 'A GPA of 3.0+ is often a screening threshold at top companies. Compensate with strong projects, internships, and certifications.' });

    if (factors.coding_skill < threshold && domain === 'technology')
      gaps.push({ area: 'Coding & DSA', icon: '💻',
        detail: `DSA score ${dsaScore}/10 needs improvement for tech roles. FAANG-tier companies expect consistent LeetCode practice and a rating above 2000.` });

    if (factors.certifications < threshold) {
      const relevant = (this.domainCertifications[domain] || []).slice(0, 4);
      gaps.push({ area: 'Certifications', icon: '🏅',
        detail: `Lacking industry-recognized ${domain} certifications. Top picks: ${relevant.map(c => c.toUpperCase()).join(', ')}.` });
    }

    if (factors.portfolio < threshold)
      gaps.push({ area: 'Portfolio / Projects', icon: '🗂️',
        detail: domain === 'technology'
          ? 'Low GitHub activity detected. Build 3–5 end-to-end projects and contribute to open source.'
          : 'A strong portfolio is critical for creative/engineering roles. Document and showcase your best work.' });

    if (factors.experience < 0.40)
      gaps.push({ area: 'Work Experience', icon: '💼',
        detail: 'Limited professional experience. Internships, contract work, and freelance projects can substitute early on.' });

    if (factors.internship < 0.33)
      gaps.push({ area: 'Internship Experience', icon: '🏢',
        detail: 'Companies strongly prefer candidates with at least 1–2 relevant internships. Apply aggressively each recruitment cycle.' });

    if (factors.communication < threshold)
      gaps.push({ area: 'Communication Skills', icon: '🗣️',
        detail: 'Recruiter surveys consistently rank communication as a top hiring factor. Practice structured responses (STAR method) and public speaking.' });

    if (factors.leadership < 0.40)
      gaps.push({ area: 'Leadership Experience', icon: '🌟',
        detail: 'Even early-career candidates benefit from demonstrable leadership — team leads, club officers, open-source maintainers.' });

    return gaps;
  }

  _generateActionPlan(factors, domain, targetCompany, certifications, educationLevel, yoe, dsaScore) {
    const plan = [];

    if (domain === 'technology') {
      if (factors.dsa < 0.70)
        plan.push('Solve 3 LeetCode problems daily (Easy→Medium→Hard). Aim for 200+ solved problems and a contest rating above 2000 within 6 months.');
      if (factors.portfolio < 0.60)
        plan.push('Build 3 GitHub projects: one full-stack web app, one ML/data project, one open-source contribution. Include READMEs and live demos.');
      if (factors.certifications < 0.50)
        plan.push('Earn AWS Solutions Architect Associate or Google Professional Cloud Developer certification — both are highly valued by top tech employers.');
      if (factors.experience < 0.40)
        plan.push('Apply to internship programs (Google STEP, Microsoft Explore, Amazon SDE Intern) or contribute to large open-source projects.');
    } else if (domain === 'finance') {
      if (factors.certifications < 0.50)
        plan.push('Begin CFA Level 1 preparation (pass rate ~40%). It\'s the most globally recognised finance credential and opens doors at bulge-bracket banks.');
      if (factors.gpa < 0.75)
        plan.push('Supplement a lower GPA by winning finance case competitions (CFA Institute, CFA Society) and completing financial modelling courses on Wall Street Prep.');
      if (factors.internship < 0.50)
        plan.push('Secure a summer analyst internship at a bank or asset manager — nearly all full-time finance roles are filled via return-offer pipelines.');
      if (factors.communication < 0.60)
        plan.push('Practice finance interviews: fit questions (STAR method), valuation technicals, and market walk-throughs. Use platforms like Mergers & Inquisitions.');
    } else if (domain === 'healthcare') {
      if (factors.education_level < 0.70)
        plan.push('Advance your education — clinical healthcare careers require a bachelor\'s at minimum, and specialized roles (NP, MD) require graduate or doctoral degrees.');
      if (factors.certifications < 0.50)
        plan.push('Obtain BLS/ACLS certifications immediately (low cost, quick win). Then pursue role-specific licensure: NCLEX-RN for nursing, USMLE for physicians.');
      if (factors.internship < 0.40)
        plan.push('Volunteer or shadow in a clinical setting. Hospital volunteer programs and clinical rotations directly translate to hiring preference.');
    } else if (domain === 'creative') {
      if (factors.portfolio < 0.70)
        plan.push('Create a professional portfolio website (Behance, Dribbble, personal domain) featuring 8–10 of your best works with process documentation.');
      if (factors.certifications < 0.50)
        plan.push('Earn Adobe Creative Cloud certifications or Figma proficiency — studios and agencies use these as baseline filters.');
      if (factors.experience < 0.40)
        plan.push('Take on freelance projects via Upwork or Fiverr to build a paid client portfolio while still in early career.');
    } else if (domain === 'engineering') {
      if (factors.certifications < 0.50)
        plan.push('Pursue a Professional Engineer (PE) license or PMP certification. Six Sigma Green Belt is also highly valued in manufacturing environments.');
      if (factors.portfolio < 0.50)
        plan.push('Document engineering projects with CAD files, simulation outputs, test results, and technical write-ups. Upload to GitHub or a personal site.');
      if (factors.gpa < 0.75)
        plan.push('Supplement grades with research papers, patent applications, or competition wins (Formula SAE, NASA design challenges).');
    }

    if (factors.communication < 0.60)
      plan.push('Join Toastmasters or take a structured public speaking course. Practice mock interviews weekly using Pramp or Interviewing.io.');
    if (factors.leadership < 0.50)
      plan.push('Seek leadership roles in student clubs, open-source projects, or volunteer organisations. Demonstrate impact with measurable outcomes.');
    if (factors.internship < 0.33 && yoe < 2)
      plan.push('Prioritise internships over early full-time roles — they provide mentorship, network access, and the highest conversion rates to permanent offers.');

    if (targetCompany) {
      const req = this.companyRequirements[targetCompany.toLowerCase()];
      if (req) {
        if (req.key_certs && req.key_certs.length > 0)
          plan.push(`For ${targetCompany} specifically: obtain ${req.key_certs.map(c => c.toUpperCase()).join(', ')} — these are explicitly valued during screening.`);
        if (req.dsa_min > 0 && dsaScore < req.dsa_min)
          plan.push(`${targetCompany} interviewers expect DSA proficiency of ${req.dsa_min}/10+. Focus on graph algorithms, dynamic programming, and system design patterns.`);
      }
    }

    return plan.slice(0, 6);
  }

  _assessCompanyReadiness(companyKey, factors, educationLevel, degreeField, certifications, dsaScore, yoe) {
    const req = this.companyRequirements[companyKey];
    if (!req) return null;

    const checks = [];
    let scoreTotal = 0.0;
    let weightTotal = 0.0;

    // Education level
    let passes = req.edu_levels.includes(educationLevel);
    let checkScore = passes ? 1.0 : 0.2;
    checks.push({ check: 'Education Level', status: passes,
      note: `Required: ${req.edu_levels.join(' / ')}`, score: checkScore });
    scoreTotal += checkScore * 25; weightTotal += 25;

    // Degree field
    passes = req.fields.includes(degreeField);
    checkScore = passes ? 1.0 : 0.40;
    checks.push({ check: 'Degree Field', status: passes,
      note: `Preferred: ${req.fields.join(' / ')}`, score: checkScore });
    scoreTotal += checkScore * 20; weightTotal += 20;

    // DSA
    const dsaMin = req.dsa_min || 0;
    if (dsaMin > 0) {
      passes = dsaScore >= dsaMin;
      checkScore = passes ? 1.0 : (dsaScore / dsaMin) * 0.60;
      checks.push({ check: 'DSA / Algorithmic Skill', status: passes,
        note: `Minimum expected: ${dsaMin}/10`, score: checkScore });
      scoreTotal += checkScore * 25; weightTotal += 25;
    }

    // Experience
    const yoePref = req.yoe_preferred || 0;
    passes = yoe >= yoePref;
    checkScore = passes ? 1.0 : (yoe / Math.max(yoePref, 1)) * 0.70;
    checks.push({ check: 'Years of Experience', status: passes,
      note: `Preferred: ${yoePref}+ years`, score: checkScore });
    scoreTotal += checkScore * 20; weightTotal += 20;

    // Key certifications
    if (req.key_certs && req.key_certs.length > 0) {
      const certsLower = (certifications || []).map(c => c.toLowerCase().trim());
      const hasAny = req.key_certs.some(k => certsLower.some(c => k.includes(c) || c.includes(k)));
      checkScore = hasAny ? 1.0 : 0.20;
      checks.push({ check: 'Key Certifications', status: hasAny,
        note: `Valued: ${req.key_certs.map(c => c.toUpperCase()).join(', ')}`, score: checkScore });
      scoreTotal += checkScore * 10; weightTotal += 10;
    }

    const readinessPct = weightTotal > 0 ? Math.round((scoreTotal / weightTotal) * 100) : 0;

    return {
      company: companyKey.replace(/\b\w/g, c => c.toUpperCase()),
      readiness_percent: readinessPct,
      checks,
      culture_fit: req.culture_fit || '',
    };
  }

  // ── Public API ────────────────────────────────────────────────
  predictCareer({
    features,
    yoe = 0,
    competitive_coding_rank = null,
    favorite_domain = null,
    familiar_disciplines = null,
    target_company = null,
    education_level = 'bachelor',
    degree_field = 'stem',
    gpa = 3.0,
    certifications = [],
    github_level = 'low',
    internships = 0,
    communication_score = 5.0,
    leadership_level = 'none',
    dsa_score = 5.0,
    system_design_score = 5.0,
    open_source = false,
    hackathons = 0,
  }) {
    // 1. ML prediction
    const featuresScaled = this._scale(features);
    const proba = this._predictProba(featuresScaled);
    const classes = this.tree.classes;

    // Top 5 indices by probability (descending)
    const indices = proba
      .map((p, i) => [p, i])
      .sort((a, b) => b[0] - a[0])
      .slice(0, 5)
      .map(([, i]) => i);

    const careers       = indices.map(i => classes[i]);
    const probabilities = indices.map(i => proba[i]);

    // 2. Effective domain
    let effectiveDomain;
    if (favorite_domain && this.domainCareerMapping[favorite_domain.toLowerCase()]) {
      effectiveDomain = favorite_domain.toLowerCase();
    } else if (target_company && this.companyDomainMapping[target_company.toLowerCase()]) {
      effectiveDomain = this.companyDomainMapping[target_company.toLowerCase()];
    } else {
      effectiveDomain = this._getCareerDomain(careers);
    }

    const weights = this.domainWeights[effectiveDomain] || this.domainWeights.default;

    // 3. Domain / company match
    let domainMatch = 0.0;
    if (favorite_domain && this.domainCareerMapping[favorite_domain.toLowerCase()]) {
      const dcareers = new Set(this.domainCareerMapping[favorite_domain.toLowerCase()]);
      domainMatch = careers.filter(c => dcareers.has(c)).length / Math.max(1, Math.min(careers.length, dcareers.size));
    }

    let companyMatch = 0.0;
    if (target_company && this.companyDomainMapping[target_company.toLowerCase()]) {
      const cdomain = this.companyDomainMapping[target_company.toLowerCase()];
      const dcareers = new Set(this.domainCareerMapping[cdomain] || []);
      companyMatch = careers.filter(c => dcareers.has(c)).length / Math.max(1, Math.min(careers.length, dcareers.size));
    }

    // 4. Readiness factors
    const factors = {
      model_confidence: probabilities[0],
      coding_skill:     this._codingSkillScore(competitive_coding_rank, dsa_score),
      dsa:              dsa_score / 10.0,
      portfolio:        this._portfolioScore(github_level, open_source, hackathons),
      experience:       Math.min(1.0, yoe / 5.0),
      education_level:  this._educationLevelScore(education_level, effectiveDomain),
      degree_relevance: this._degreeRelevanceScore(degree_field, effectiveDomain),
      domain_match:     domainMatch,
      company_match:    companyMatch,
      certifications:   this._certificationsScore(certifications, effectiveDomain),
      gpa:              Math.min(1.0, gpa / 4.0),
      internship:       Math.min(1.0, internships / 3.0),
      communication:    communication_score / 10.0,
      leadership:       this._leadershipScore(leadership_level),
    };

    // 5. Viability score
    const viabilityScore = Object.keys(factors)
      .reduce((sum, k) => sum + factors[k] * (weights[k] || 0), 0);

    let viabilityLevel, explanation;
    if (viabilityScore >= 0.70) {
      viabilityLevel = 'High';
      explanation = 'Your profile strongly aligns with industry readiness standards.';
    } else if (viabilityScore >= 0.45) {
      viabilityLevel = 'Moderate';
      explanation = 'Your profile shows good potential with some addressable gaps.';
    } else {
      viabilityLevel = 'Low';
      explanation = 'Your profile has significant gaps vs. what companies screen for.';
    }

    // 6. Alternative companies
    const techRecommended = careers.slice(0, 3).some(c => this.codingIntensiveCareers.includes(c));
    let alternativeCompanies = [];
    if (techRecommended) {
      if (this._isLowCodingRank(competitive_coding_rank))
        alternativeCompanies.push(...this.alternativeCompanies.low_coding_rank);
      if (yoe < 2)
        alternativeCompanies.push(...this.alternativeCompanies.low_yoe);
    }

    // 7. Skill gaps + action plan
    const skillGaps = this._identifySkillGaps(
      factors, effectiveDomain, education_level, degree_field,
      certifications, target_company, dsa_score
    );
    const actionPlan = this._generateActionPlan(
      factors, effectiveDomain, target_company,
      certifications, education_level, yoe, dsa_score
    );

    // 8. Company readiness
    let companyReadiness = null;
    if (target_company) {
      const companyKey = target_company.toLowerCase();
      if (this.companyRequirements[companyKey]) {
        companyReadiness = this._assessCompanyReadiness(
          companyKey, factors, education_level, degree_field,
          certifications, dsa_score, yoe
        );
      }
    }

    return {
      predicted_careers:    careers,
      career_probabilities: probabilities,
      viability_assessment: { score: viabilityScore, level: viabilityLevel, explanation },
      readiness_factors:    factors,
      skill_gaps:           skillGaps,
      action_plan:          actionPlan,
      company_readiness:    companyReadiness,
      alternative_companies: alternativeCompanies,
      effective_domain:     effectiveDomain,
    };
  }
}
