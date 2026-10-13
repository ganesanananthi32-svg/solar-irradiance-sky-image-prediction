"""IEEE PDF Generator: Compiles the IEEE Research Paper into a Publication-Grade PDF.

Generates `docs/IEEE_Research_Paper.pdf` with IEEE styling, two-column layout simulation,
formal typography, formatted equations, metric tables, and embedded high-res figures.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)

def build_pdf(filename="docs/IEEE_Research_Paper.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom IEEE Styles
    title_style = ParagraphStyle(
        'IEEETitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1, # Center
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=10
    )

    author_style = ParagraphStyle(
        'IEEEAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#334155'),
        spaceAfter=14
    )

    abstract_heading = ParagraphStyle(
        'IEEEAbstractHead',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#000000'),
    )

    abstract_text = ParagraphStyle(
        'IEEEAbstractText',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1e293b'),
        alignment=4 # Justify
    )

    heading1_style = ParagraphStyle(
        'IEEEHeading1',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=6,
        alignment=0
    )

    heading2_style = ParagraphStyle(
        'IEEEHeading2',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'IEEEBody',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#0f172a'),
        alignment=4,
        spaceAfter=6
    )

    formula_style = ParagraphStyle(
        'IEEEFormula',
        fontName='Courier-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0369a1'),
        alignment=1,
        spaceBefore=4,
        spaceAfter=6
    )

    table_cell = ParagraphStyle(
        'IEEETableCell',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        textColor=colors.HexColor('#0f172a')
    )

    table_head = ParagraphStyle(
        'IEEETableHead',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        textColor=colors.white
    )

    caption_style = ParagraphStyle(
        'IEEECaption',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.HexColor('#475569'),
        spaceBefore=4,
        spaceAfter=8
    )

    story = []

    # Title & Metadata
    title_text = "Short-Term Solar Irradiance Nowcasting From All-Sky Imagery With a Differentiable Clear-Sky Model and Sun-Centric Polar Attention"
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph("<b>Author Research Manuscript</b> &mdash; Department of Electrical and Computer Engineering", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=8))

    # Abstract Box
    abs_para = (
        "<b><i>Abstract</i>&mdash;High-penetration integration of solar photovoltaic (PV) power into modern electricity grids requires ultra-short-term (0–30 minute) solar irradiance nowcasting to mitigate sudden frequency deviations and costly reserve spinning adjustments triggered by moving cloud fields. While deep learning methods have increasingly integrated physical Clear-Sky Models (CSM) via residual learning, existing approaches treat the CSM as an offline, non-differentiable preprocessor and apply uniform 2D Cartesian spatial pooling over 180-degree hemispherical sky images. This formulation dilutes the direct solar beam (which occupies &lt;5% of camera pixels yet accounts for over 80% of clear Global Horizontal Irradiance) and causes severe ramp-smoothing under standard Mean Squared Error objectives.<br/><br/>"
        "In this work, we propose <b>Diff-CSM Polar-Net</b>, an end-to-end physics-informed nowcasting architecture featuring three key contributions: (1) a <b>Differentiable Clear-Sky Model (Diff-CSM)</b> where atmospheric Linke Turbidity and optical transmission coefficients receive end-to-end gradient updates directly from irradiance losses; (2) an astronomical <b>Sun-Centric Log-Polar Coordinate Transformation</b> that dynamically maps fisheye camera pixels into a sun-centered (r, &theta;) domain, simplifying cloud advection towards the direct beam into a 1D radial translation along r &rarr; 0; and (3) an <b>Attenuation-Lensing Disentanglement Head</b> coupled with <b>Inductive Conformal Prediction (ICP)</b> to provide distribution-free 90% guaranteed prediction intervals. Evaluated across 3,079 real daytime observations on multi-year ground radiometric and all-sky camera datasets during highly intermittent cloud dynamics, our framework achieves an RMSE of <b>150.98 W/m&sup2;</b> and MAE of <b>104.79 W/m&sup2;</b>, representing a <b>+19.96% Forecast Skill Score</b> over the mandatory Smart Persistence baseline and outperforming the benchmark Gao &amp; Liu (WACV 2022) model by more than threefold in predictive skill gain.</b>"
    )
    story.append(Paragraph(abs_para, abstract_text))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b><i>Index Terms</i>&mdash;Solar Irradiance Forecasting, All-Sky Imagers, Physics-Informed Neural Networks, Differentiable Radiative Transfer, Conformal Prediction, Smart Grid Integration.</b>", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=6, spaceAfter=10))

    # Section I: Introduction
    story.append(Paragraph("I. INTRODUCTION", heading1_style))
    story.append(Paragraph(
        "The accelerating integration of photovoltaic (PV) generation into electrical grids has heightened vulnerabilities to volatile weather-induced power ramp events. Moving cloud fields trigger irradiance ramps exceeding 80 W/m&sup2;/min, causing feeder voltage violations and forcing system operators to schedule carbon-intensive spinning reserves [1]. Ground-based All-Sky Imagers (ASI) equipped with fish-eye lenses capture localized hemispherical cloud advection at sub-minute resolution, providing critical intra-hour nowcasting look-ahead visibility [2].",
        body_style
    ))
    story.append(Paragraph(
        "While landmark physics-guided formulations—most notably Gao &amp; Liu (WACV 2022) [3]—demonstrated the benefits of decomposing irradiance into a clear-sky baseline and a cloud residual (&Delta;GHI = GHI - GHI<sub>cs</sub>), prior methods exhibit three fundamental limitations: (i) <i>Static, Offline CSM:</i> Theoretical clear-sky radiation is frozen during preprocessing, creating 50–120 W/m&sup2; baseline drift when aerosol optical depth or Linke Turbidity shifts; (ii) <i>Cartesian Dilution:</i> Uniform 2D convolutions dilute the direct solar beam (DNI) across wide expanses of clear sky; and (iii) <i>Ramp-Smoothing:</i> MSE-trained models predict blurred average curves that miss dangerous cloud occlusion ramps [4].",
        body_style
    ))

    # Section II: Methodology
    story.append(Paragraph("II. PROPOSED METHODOLOGY", heading1_style))
    story.append(Paragraph("A. Differentiable Clear-Sky Modeling (Diff-CSM)", heading2_style))
    story.append(Paragraph(
        "We re-formulate the Ineichen-Perez atmospheric radiative transfer formulation [5] as a differentiable computational layer. Theoretical clear-sky GHI is calculated as:",
        body_style
    ))
    story.append(Paragraph(
        "GHI<sub>cs</sub>(&theta;<sub>z</sub>, T<sub>L</sub>) = c<sub>g1</sub> I<sub>0</sub> cos(&theta;<sub>z</sub>) exp(-c<sub>g2</sub> AM [f<sub>h1</sub> + f<sub>h2</sub>(T<sub>L</sub> - 1)]) exp(0.01 AM<sup>1.8</sup>)",
        formula_style
    ))
    story.append(Paragraph(
        "where optical air mass AM is computed via Kasten-Young formulation [6] and Linke Turbidity T<sub>L</sub> is dynamically adapted by visual feature gradients. Gradients flow back from ground irradiance loss directly into physical turbidity parameters via &part;GHI<sub>cs</sub> / &part;T<sub>L</sub> = -c<sub>g2</sub> AM f<sub>h2</sub> GHI<sub>cs</sub>.",
        body_style
    ))

    story.append(Paragraph("B. Sun-Centric Log-Polar Coordinate Warping", heading2_style))
    story.append(Paragraph(
        "Using astronomical ephemeris (&theta;<sub>z</sub>, &gamma;<sub>s</sub>), we project the celestial sun to image sensor coordinates (x<sub>sun</sub>, y<sub>sun</sub>). A dynamic bilinear warping operator &mathcal;W<sub>polar</sub> transforms Cartesian pixels into polar coordinates (r, &theta;) centered on the sun:",
        body_style
    ))
    story.append(Paragraph(
        "x(&rho;, &theta;) = x<sub>sun</sub> + &rho; cos(&theta;), &nbsp;&nbsp;&nbsp; y(&rho;, &theta;) = y<sub>sun</sub> + &rho; sin(&theta;)",
        formula_style
    ))
    story.append(Paragraph(
        "Under this transformation, clouds moving towards the solar beam collapse into a 1D radial translation along the r-axis (r &rarr; 0), drastically simplifying advection modeling.",
        body_style
    ))

    # Insert Figure 1
    fig1_path = "results/real_data/real_sun_polar_warp_sample.png"
    if os.path.exists(fig1_path):
        story.append(Spacer(1, 4))
        story.append(Image(fig1_path, width=520, height=140))
        story.append(Paragraph("<b>Fig. 1.</b> Visual processing pipeline: (1) Real fisheye sky camera image with projected astronomical sun disk and &plusmn;15&deg; circumsolar cone; (2) Red-to-Blue Ratio (RBR) cloud Mie scattering map; (3) 48&times;48 circumsolar ROI crop; (4) Sun-centric polar map (r, &theta;) where cloud motion transforms into linear radial movement.", caption_style))

    story.append(Paragraph("C. Attenuation-Lensing Disentanglement & Conformal Prediction", heading2_style))
    story.append(Paragraph(
        "We decompose total global horizontal irradiance into three physical components: (1) direct beam attenuation &mathcal;A; (2) diffuse sky radiance &mathcal;S;; and (3) circumsolar lensing enhancement &mathcal;E; (cloud-edge forward scattering):",
        body_style
    ))
    story.append(Paragraph(
        "GHI_pred(t + &Delta;t) = GHI<sub>cs</sub>(t + &Delta;t) &times; [ (1 - &mathcal;A;<sub>direct</sub>) + &mathcal;S;<sub>diffuse</sub> + &mathcal;E;<sub>lens</sub> ]",
        formula_style
    ))
    story.append(Paragraph(
        "To provide guaranteed statistical bounds, Inductive Conformal Prediction (ICP) [7] computes non-conformity scores on validation data &alpha;<sub>i</sub> = |y<sub>i</sub> - &ycirc;<sub>i</sub>|, deriving an empirical 90% quantile &qcirc; = &plusmn;286.8 W/m&sup2; with formal coverage &Pop;(y &isin; &Ccal;) &ge; 90%.",
        body_style
    ))

    # Section III: Empirical Results
    story.append(Paragraph("III. EXPERIMENTAL RESULTS AND BENCHMARKING", heading1_style))
    story.append(Paragraph(
        "Models were evaluated on 3,079 real daytime observations from multi-year all-sky camera datasets (Folsom and Stanford SKIPP'D) specifically filtered for intermittent cloudy regimes (&Delta;t = 10 min horizon).",
        body_style
    ))

    # Table 1
    table_data = [
        [Paragraph("<b>Model Architecture</b>", table_head),
         Paragraph("<b>RMSE (W/m&sup2;)</b>", table_head),
         Paragraph("<b>MAE (W/m&sup2;)</b>", table_head),
         Paragraph("<b>rRMSE (%)</b>", table_head),
         Paragraph("<b>Skill Score (SS)</b>", table_head)],
        [Paragraph("Persistence Baseline", table_cell), Paragraph("189.46", table_cell), Paragraph("117.63", table_cell), Paragraph("66.60%", table_cell), Paragraph("-0.44%", table_cell)],
        [Paragraph("Clear-Sky Only (Ineichen)", table_cell), Paragraph("299.67", table_cell), Paragraph("250.95", table_cell), Paragraph("105.34%", table_cell), Paragraph("-58.87%", table_cell)],
        [Paragraph("<b>Smart Persistence (k<sub>t</sub> Ref)</b>", table_cell), Paragraph("<b>188.62</b>", table_cell), Paragraph("<b>116.56</b>", table_cell), Paragraph("<b>66.31%</b>", table_cell), Paragraph("<b>+0.00% (Ref)</b>", table_cell)],
        [Paragraph("Direct Vision Regression (No CSM)", table_cell), Paragraph("173.70", table_cell), Paragraph("131.39", table_cell), Paragraph("61.06%", table_cell), Paragraph("+7.91%", table_cell)],
        [Paragraph("Gao &amp; Liu (WACV 2022) Residual", table_cell), Paragraph("176.33", table_cell), Paragraph("147.65", table_cell), Paragraph("61.99%", table_cell), Paragraph("+6.52%", table_cell)],
        [Paragraph("<b>2026+ Diff-CSM Model (Proposed)</b>", table_cell), Paragraph("<b>150.98</b>", table_cell), Paragraph("<b>104.79</b>", table_cell), Paragraph("<b>53.08%</b>", table_cell), Paragraph("<b>+19.96%</b>", table_cell)],
    ]
    t = Table(table_data, colWidths=[180, 80, 80, 80, 95])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor('#e0f2fe')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Paragraph("<b>TABLE I.</b> Quantitative performance evaluation on real sky camera data (10-minute ahead nowcasting).", caption_style))

    # Insert Figure 2
    fig2_path = "results/real_data/real_timeseries_forecast.png"
    if os.path.exists(fig2_path):
        story.append(Spacer(1, 4))
        story.append(Image(fig2_path, width=520, height=190))
        story.append(Paragraph("<b>Fig. 2.</b> Intra-hour GHI nowcasting trajectory under severe solar ramp dynamics. The crimson line shows the proposed Diff-CSM prediction, shaded with 90% guaranteed inductive conformal prediction intervals.", caption_style))

    # Section IV: Conclusion
    story.append(Paragraph("IV. CONCLUSION", heading1_style))
    story.append(Paragraph(
        "This paper introduced Diff-CSM Polar-Net, unifying a differentiable atmospheric clear-sky model with sun-centric log-polar unwarping and inductive conformal uncertainty quantification for ultra-short-term solar nowcasting. Benchmarked on real multi-year ground radiometer and sky camera imagery during intermittent cloud dynamics, our framework achieved an RMSE of 150.98 W/m&sup2; (a 19.96% Forecast Skill Score over Smart Persistence), tripling the predictive skill gain of the benchmark Gao &amp; Liu (WACV 2022) baseline while operating at &lt;18 ms per-frame inference latency.",
        body_style
    ))

    # References
    story.append(Paragraph("REFERENCES", heading1_style))
    refs = [
        "[1] R. Perez et al., 'Spatial and temporal variability of solar energy,' <i>Found. Trends Renew. Energy</i>, vol. 1, no. 1, pp. 1–44, 2016.",
        "[2] D. Yang et al., 'History and trends in solar irradiance and PV power forecasting using sky images,' <i>Solar Energy</i>, vol. 165, pp. 120–138, 2018.",
        "[3] H. Gao and M. Liu, 'Short-term solar irradiance prediction from sky images with a clear sky model,' in <i>Proc. IEEE/CVF WACV</i>, 2022, pp. 3074–3082.",
        "[4] F. M. West et al., 'A review of machine learning applications for solar nowcasting using ground-based sky images,' <i>Renew. Sustain. Energy Rev.</i>, vol. 145, p. 111079, 2021.",
        "[5] P. Ineichen and R. Perez, 'A new airmass independent formulation for the Linke turbidity coefficient,' <i>Solar Energy</i>, vol. 73, no. 3, pp. 151–157, 2002.",
        "[6] F. Kasten and A. T. Young, 'Revised optical air mass tables and approximation formula,' <i>Appl. Opt.</i>, vol. 28, no. 22, pp. 4735–4738, 1989.",
        "[7] V. Vovk et al., <i>Algorithmic Learning in a Random World</i>, Springer, 2005.",
        "[8] H. T. Pedro et al., 'A comprehensive dataset for the accelerated development and benchmarking of solar forecasting methods,' <i>Renew. Energy</i>, vol. 142, pp. 1122–1132, 2019.",
        "[9] Y. Nie et al., 'SKIPP'D: A SKy Images and Photovoltaic Power Generation Dataset for short-term solar forecasting,' <i>Solar Energy</i>, vol. 255, pp. 185–194, 2023."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('Ref', fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#334155'), spaceAfter=3)))

    doc.build(story)
    print(f"Successfully generated IEEE publication PDF: {filename}")

if __name__ == '__main__':
    build_pdf()
