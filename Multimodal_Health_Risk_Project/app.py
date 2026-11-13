from pathlib import Path
import json
import io

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session,
    redirect,
    url_for,
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)

from src.preprocessing import (
    load_data,
    FEATURES,
    LAB_FEATURES,
    HISTORY_FEATURES,
    VITAL_FEATURES,
    IMAGING_FEATURES,
)

from src.predictor import predict


# ============================================================
# APPLICATION SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent

app = Flask(__name__)

# Needed for storing the latest prediction temporarily.
app.secret_key = "healthrisk-ai-project-secret-key"


# ============================================================
# MODEL CHECK
# ============================================================

def model_ready():
    return (ROOT / "models" / "disease_model.joblib").exists()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        ready=model_ready()
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def prediction():

    if request.method == "GET":
        return render_template(
            "prediction.html",
            lab=LAB_FEATURES,
            history=HISTORY_FEATURES,
            vital=VITAL_FEATURES,
            imaging=IMAGING_FEATURES
        )

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    if not model_ready():
        return render_template(
            "error.html",
            message=(
                "Models are not trained yet. "
                "Run: python src/train_models.py"
            )
        )

    # --------------------------------------------------------
    # Check whether at least one value was entered
    # --------------------------------------------------------

    submitted_data = request.form.to_dict()

    has_input = False

    for value in submitted_data.values():

        if value is not None and str(value).strip() != "":
            has_input = True
            break

    if not has_input:

        return render_template(
            "error.html",
            message="Please enter at least one health factor before analysis."
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        result = predict(submitted_data)

    except Exception as e:

        return render_template(
            "error.html",
            message=f"Prediction could not be completed: {str(e)}"
        )

    # --------------------------------------------------------
    # Store latest result in session
    # --------------------------------------------------------

    session["last_result"] = result

    return render_template(
        "result.html",
        result=result
    )


# ============================================================
# PATIENT-SPECIFIC RESULT DASHBOARD
# ============================================================

@app.route("/result-dashboard")
def result_dashboard():

    result = session.get("last_result")

    if not result:

        return redirect(url_for("prediction"))

    return render_template(
        "result_dashboard.html",
        result=result
    )


# ============================================================
# CREATE PIE CHART
# ============================================================

def create_probability_chart(result):

    probabilities = result.get(
        "disease_probabilities",
        {}
    )

    if not probabilities:
        return None

    labels = list(probabilities.keys())

    values = [
        float(probabilities[label]) * 100
        for label in labels
    ]

    # --------------------------------------------------------
    # Colors for four diseases
    # --------------------------------------------------------

    color_map = {
        "Diabetes": "#4F8EF7",
        "Heart Disease": "#F05D7B",
        "Bronchitis": "#F39A3D",
        "Breast Cancer": "#F5C84B",
    }

    chart_colors = [
        color_map.get(label, "#9AA6B2")
        for label in labels
    ]

    # --------------------------------------------------------
    # Create pie chart
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6.5, 4.5),
        dpi=180
    )

    wedges, _ = ax.pie(
        values,
        colors=chart_colors,
        startangle=90,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 2
        }
    )

    ax.set_title(
        "Disease Probability Distribution",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_labels = [
        f"{label} — {value:.2f}%"
        for label, value in zip(labels, values)
    ]

    ax.legend(
        wedges,
        legend_labels,
        title="Disease",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=9
    )

    ax.axis("equal")

    plt.tight_layout()

    # --------------------------------------------------------
    # Save chart in memory
    # --------------------------------------------------------

    chart_buffer = io.BytesIO()

    plt.savefig(
        chart_buffer,
        format="png",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    chart_buffer.seek(0)

    return chart_buffer


# ============================================================
# DOWNLOAD PDF RESULT
# ============================================================

@app.route("/download-result")
def download_result():

    result = session.get("last_result")

    if not result:

        return redirect(url_for("prediction"))

    # --------------------------------------------------------
    # PDF buffer
    # --------------------------------------------------------

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    # --------------------------------------------------------
    # Custom styles
    # --------------------------------------------------------

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#17324E"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#718096"),
        alignment=TA_CENTER,
        spaceAfter=15,
    )

    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#193451"),
        spaceBefore=8,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#52657B"),
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#68788C"),
    )

    center_small = ParagraphStyle(
        "CenterSmall",
        parent=small_style,
        alignment=TA_CENTER,
    )

    # --------------------------------------------------------
    # Story
    # --------------------------------------------------------

    story = []

    # ========================================================
    # HEADER
    # ========================================================

    story.append(
        Paragraph(
            "HealthRisk AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Health Risk Analysis Report",
            subtitle_style
        )
    )

    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "Prediction Summary",
            section_style
        )
    )

    summary_data = [
        [
            Paragraph("<b>Predicted Disease</b>", small_style),
            Paragraph("<b>Risk Level</b>", small_style),
            Paragraph("<b>Model Confidence</b>", small_style),
        ],
        [
            Paragraph(
                str(result.get("disease", "N/A")),
                normal_style
            ),
            Paragraph(
                str(result.get("risk", "N/A")),
                normal_style
            ),
            Paragraph(
                f"{result.get('confidence', 0)}%",
                normal_style
            ),
        ]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[58 * mm, 58 * mm, 58 * mm]
    )

    summary_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#F2F6FB")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                colors.HexColor("#D7E1EC")
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#E2E8F0")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
        ])
    )

    story.append(summary_table)

    story.append(Spacer(1, 8))

    # ========================================================
    # PIE CHART
    # ========================================================

    story.append(
        Paragraph(
            "Disease Probability Distribution",
            section_style
        )
    )

    chart_buffer = create_probability_chart(result)

    if chart_buffer:

        chart_image = Image(
            chart_buffer,
            width=120 * mm,
            height=75 * mm
        )

        story.append(chart_image)

        story.append(Spacer(1, 5))

        probabilities = result.get(
            "disease_probabilities",
            {}
        )

        probability_rows = [
            [
                Paragraph("<b>Disease</b>", small_style),
                Paragraph("<b>Probability</b>", small_style)
            ]
        ]

        for disease, probability in probabilities.items():

            probability_rows.append([
                Paragraph(
                    str(disease),
                    small_style
                ),
                Paragraph(
                    f"{float(probability) * 100:.2f}%",
                    small_style
                )
            ])

        probability_table = Table(
            probability_rows,
            colWidths=[110 * mm, 45 * mm]
        )

        probability_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F2F6FB")
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#DCE5EF")
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
            ])
        )

        story.append(probability_table)

    # ========================================================
    # HEALTH FACTORS
    # ========================================================

    story.append(
        Paragraph(
            "Health Factors Used",
            section_style
        )
    )

    factors = [
        ["Laboratory Reports", "Laboratory measurements"],
        ["Medical History", "Patient history information"],
        ["Vital Signs", "Physical health measurements"],
        ["Medical Imaging", "Imaging-related findings"],
    ]

    factor_table = Table(
        factors,
        colWidths=[65 * mm, 90 * mm]
    )

    factor_table.setStyle(
        TableStyle([
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#DCE5EF")
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#E6ECF2")
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#F7FAFD")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor("#52657B")
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
        ])
    )

    story.append(factor_table)

    # ========================================================
    # SHAP
    # ========================================================

    story.append(
        Paragraph(
            "Major Contributing Factors — SHAP",
            section_style
        )
    )

    shap_rows = [
        [
            Paragraph("<b>Feature</b>", small_style),
            Paragraph("<b>Impact</b>", small_style),
            Paragraph("<b>Direction</b>", small_style),
        ]
    ]

    for item in result.get("shap", []):

        feature = item.get("feature", "Unknown")
        impact = item.get("impact", 0)
        direction = item.get("direction", "neutral")

        try:
            impact_text = f"{float(impact):.4f}"
        except:
            impact_text = str(impact)

        shap_rows.append([
            Paragraph(str(feature), small_style),
            Paragraph(impact_text, small_style),
            Paragraph(str(direction), small_style),
        ])

    if len(shap_rows) > 1:

        shap_table = Table(
            shap_rows,
            colWidths=[75 * mm, 40 * mm, 55 * mm]
        )

        shap_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F2F6FB")
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#DCE5EF")
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
            ])
        )

        story.append(shap_table)

    # ========================================================
    # CURRENT PREVENTION
    # ========================================================

    story.append(
        Paragraph(
            "Current Disease Prevention",
            section_style
        )
    )

    current_prevention_table = Table(
        [[
            Paragraph(
                str(result.get(
                    "current_prevention",
                    "No prevention information available."
                )),
                normal_style
            )
        ]],
        colWidths=[170 * mm]
    )

    current_prevention_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#F8FBFF")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#DCE5EF")
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                10
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                10
            ),
        ])
    )

    story.append(current_prevention_table)

    # ========================================================
    # NEXT DISEASE
    # ========================================================

    story.append(
        Paragraph(
            "Future Risk Analysis",
            section_style
        )
    )

    next_data = [
        [
            Paragraph("<b>Next Possible Disease</b>", small_style),
            Paragraph("<b>Prevention & Guidance</b>", small_style),
        ],
        [
            Paragraph(
                str(result.get("next_disease", "N/A")),
                normal_style
            ),
            Paragraph(
                str(result.get(
                    "next_prevention",
                    "No information available."
                )),
                normal_style
            ),
        ]
    ]

    next_table = Table(
        next_data,
        colWidths=[65 * mm, 105 * mm]
    )

    next_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#F2F8F6")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor("#D5E9E1")
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#E2ECE8")
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
        ])
    )

    story.append(next_table)

    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This system is a machine-learning "
            "research prototype intended for analytical and educational "
            "purposes. The predictions should not be considered a medical "
            "diagnosis. Professional medical advice should be obtained "
            "from a qualified healthcare professional.",
            small_style
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(story)

    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="HealthRisk_AI_Analysis_Report.pdf"
    )


# ============================================================
# ORIGINAL DATASET DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    df = load_data()

    metrics = {}

    p = ROOT / "reports" / "metrics.json"

    if p.exists():

        metrics = json.loads(
            p.read_text(
                encoding="utf-8"
            )
        )

    disease_counts = (
        df["Disease_Label"]
        .value_counts()
        .to_dict()
    )

    risk_counts = (
        df["Risk_Level"]
        .value_counts()
        .to_dict()
    )

    next_counts = (
        df["Next_Possible_Disease"]
        .value_counts()
        .to_dict()
    )

    return render_template(
        "dashboard.html",
        rows=len(df),
        columns=len(df.columns),
        disease_counts=disease_counts,
        risk_counts=risk_counts,
        next_counts=next_counts,
        metrics=metrics
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )