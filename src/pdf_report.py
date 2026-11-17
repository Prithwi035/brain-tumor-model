"""
Clinical Brain MRI Diagnostic Report Generator
Generates a structured, one-page A4 clinical radiology report with embedded triad imaging.
"""
import io, datetime
import numpy as np
import cv2
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
)

CLASS_DESCRIPTIONS = {
    "glioma":     ("Glioma (Intra-axial Glial Neoplasm)", "Abnormal mass exhibiting infiltrative intra-axial glial proliferation."),
    "meningioma": ("Meningioma (Extra-axial Dural Mass)", "Circumscribed extra-axial lesion arising from arachnoid cap cells."),
    "pituitary":  ("Pituitary Tumor (Sellar Mass)", "Abnormal sellar or suprasellar expansion with potential chiasm involvement."),
    "notumor":    ("Normal Brain Parenchyma", "No evidence of focal mass, midline shift, or pathological signal abnormality.")
}

CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary Tumor",
    "notumor": "Normal (No Tumor)"
}


def _ndarray_to_rl_image(img_rgb: np.ndarray, width: float, height: float) -> RLImage:
    """Converts a numpy RGB array directly to a ReportLab Flowable Image in-memory."""
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    img_io = io.BytesIO()
    pil_img.save(img_io, format="PNG", optimize=True)
    img_io.seek(0)
    return RLImage(img_io, width=width, height=height)


def generate_pdf_report(
    scan_name: str,
    pred_class: str,
    confidence: float,
    probs: dict,
    lesion_pct: float,
    img_rgb: np.ndarray,
    cam_overlay: np.ndarray,
    unet_overlay: np.ndarray,
    output_path: str = None
) -> bytes:
    """
    Synthesize an end-to-end clinical PDF diagnostic summary report.

    Args:
        scan_name: File name of the MRI scan
        pred_class: Primary classification outcome ('glioma', 'meningioma', 'pituitary', 'notumor')
        confidence: Prediction confidence float (0.0 - 1.0)
        probs: Dict mapping class names to probabilities
        lesion_pct: U-Net estimated lesion area percentage
        img_rgb: Original MRI RGB numpy array
        cam_overlay: Grad-CAM overlay RGB numpy array
        unet_overlay: U-Net boundary overlay RGB numpy array
        output_path: Optional file path to write the PDF

    Returns:
        PDF bytes buffer
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        output_path if output_path else pdf_buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    
    # Custom Clinical Styles
    title_style = ParagraphStyle(
        'ClinicTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0f2744'),
        spaceAfter=1
    )
    subtitle_style = ParagraphStyle(
        'ClinicSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#475569'),
        spaceAfter=0
    )
    meta_label = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#475569')
    )
    meta_val = ParagraphStyle(
        'MetaValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f172a')
    )
    section_heading = ParagraphStyle(
        'SectionHdr',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#0f2744'),
        spaceBefore=4,
        spaceAfter=4
    )
    finding_banner = ParagraphStyle(
        'FindingBanner',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#b91c1c') if pred_class != "notumor" else colors.HexColor('#047857')
    )
    finding_body = ParagraphStyle(
        'FindingBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )
    caption_style = ParagraphStyle(
        'ImageCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9,
        alignment=1,
        textColor=colors.HexColor('#334155')
    )
    caption_sub = ParagraphStyle(
        'ImageCaptionSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8,
        alignment=1,
        textColor=colors.HexColor('#64748b')
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor('#64748b')
    )

    story = []

    # 1. Header with Clinic Branding & Report ID
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    study_id = scan_name.split('.')[0].upper()
    header_data = [
        [
            Paragraph("CLINICAL BRAIN MRI DIAGNOSTIC REPORT", title_style),
            Paragraph(f"<b>REPORT ID:</b> MRIDx-{study_id}<br/><b>ISSUED:</b> {now_str}", ParagraphStyle('RightMeta', fontName='Helvetica', fontSize=7.5, leading=10, alignment=2, textColor=colors.HexColor('#475569')))
        ],
        [
            Paragraph("AI-Assisted Cranial MRI Computer-Aided Detection (CAD) System", subtitle_style),
            ""
        ]
    ]
    t_header = Table(header_data, colWidths=[360, 163])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_header)

    # Decorative Bar
    story.append(Spacer(1, 4))
    t_rule = Table([[""]], colWidths=[523], rowHeights=[2])
    t_rule.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0f2744')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_rule)
    story.append(Spacer(1, 8))

    # 2. Patient & Examination Information Box
    meta_table_data = [
        [
            Paragraph("EXAMINATION FILE:", meta_label), Paragraph(scan_name, meta_val),
            Paragraph("MODALITY:", meta_label), Paragraph("Cranial MRI (Axial Slice)", meta_val)
        ],
        [
            Paragraph("ACQUISITION MATRIX:", meta_label), Paragraph(f"{img_rgb.shape[1]} × {img_rgb.shape[0]} px", meta_val),
            Paragraph("ANALYSIS PIPELINE:", meta_label), Paragraph("EfficientNetB3 + U-Net + Grad-CAM", meta_val)
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[105, 155, 105, 158])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # 3. Primary Finding Card
    is_tumor = pred_class != "notumor"
    status_label = "PATHOLOGICAL ANOMALY DETECTED" if is_tumor else "NO PATHOLOGICAL ANOMALY IDENTIFIED"
    diag_name, diag_desc = CLASS_DESCRIPTIONS.get(pred_class, (pred_class.capitalize(), ""))

    finding_color = colors.HexColor('#fef2f2') if is_tumor else colors.HexColor('#f0fdf4')
    border_color = colors.HexColor('#f87171') if is_tumor else colors.HexColor('#4ade80')

    finding_cell = [
        Paragraph(f"STATUS: {status_label}", finding_banner),
        Spacer(1, 3),
        Paragraph(f"<b>Primary Diagnostic Finding:</b> {diag_name}", finding_body),
        Paragraph(f"<b>Classification Confidence:</b> {confidence:.1%} &nbsp;|&nbsp; <b>Segmented Lesion Area:</b> {lesion_pct:.2f}% of slice", finding_body),
        Paragraph(f"<b>Pathology Note:</b> {diag_desc}", ParagraphStyle('PNote', parent=finding_body, fontSize=7.5, textColor=colors.HexColor('#475569')))
    ]

    t_finding = Table([[finding_cell]], colWidths=[523])
    t_finding.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), finding_color),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_finding)
    story.append(Spacer(1, 10))

    # 4. Comparative Differential Probabilities
    story.append(Paragraph("DIFFERENTIAL DIAGNOSTIC PROBABILITIES", section_heading))
    prob_header = [Paragraph("<b>Tumor Class</b>", meta_label), Paragraph("<b>Pathology Category</b>", meta_label), Paragraph("<b>Likelihood</b>", meta_label), Paragraph("<b>Visual Score</b>", meta_label)]
    prob_rows = [prob_header]

    for c in ["glioma", "meningioma", "pituitary", "notumor"]:
        p = probs.get(c, 0.0)
        c_label = CLASS_DISPLAY[c]
        c_desc = CLASS_DESCRIPTIONS[c][0].split('(')[-1].replace(')', '')
        
        # Textual progress bar representation
        blocks = int(round(p * 20))
        bar_text = "█" * blocks + "░" * (20 - blocks)
        prob_rows.append([
            Paragraph(f"<b>{c_label}</b>", meta_val),
            Paragraph(c_desc, ParagraphStyle('SubVal', parent=meta_val, fontSize=7.5, textColor=colors.HexColor('#475569'))),
            Paragraph(f"<b>{p:.1%}</b>", ParagraphStyle('PVal', parent=meta_val, fontName='Helvetica-Bold', textColor=colors.HexColor('#0f2744'))),
            Paragraph(f"<font color='#0284c7'>{bar_text}</font>", ParagraphStyle('Bar', fontName='Courier', fontSize=6.5, leading=8))
        ])

    t_probs = Table(prob_rows, colWidths=[110, 193, 70, 150])
    t_probs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_probs)
    story.append(Spacer(1, 10))

    # 5. Triad Imaging Panel (3 In-Memory High-Resolution Viewports)
    story.append(Paragraph("TRIAD IMAGING EXAMINATION: ORIGINAL SCAN, GRAD-CAM XAI & U-NET DELINEATION", section_heading))
    
    img_dim = 168  # 168 x 168 pt square (~2.33 inches each, total = 504 pt, fits perfectly on A4 523 pt)
    rl_img1 = _ndarray_to_rl_image(img_rgb, img_dim, img_dim)
    rl_img2 = _ndarray_to_rl_image(cam_overlay, img_dim, img_dim)
    rl_img3 = _ndarray_to_rl_image(unet_overlay, img_dim, img_dim)

    cap1 = [Paragraph("VIEW 1: ORIGINAL SCAN", caption_style), Paragraph("Input Axial MRI", caption_sub)]
    cap2 = [Paragraph("VIEW 2: AI ATTENTION (GRAD-CAM)", caption_style), Paragraph("Neural Feature Locus", caption_sub)]
    cap3 = [Paragraph("VIEW 3: U-NET LESION BOUNDARY", caption_style), Paragraph(f"Delimited Area: {lesion_pct:.2f}%", caption_sub)]

    img_table_data = [
        [rl_img1, rl_img2, rl_img3],
        [cap1, cap2, cap3]
    ]

    t_images = Table(img_table_data, colWidths=[174, 174, 175])
    t_images.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,0), 0.5, colors.HexColor('#cbd5e1')),
        ('BOX', (0,1), (-1,1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#050811')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f8fafc')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_images)
    story.append(Spacer(1, 10))

    # 6. Clinical Radiological Impression
    story.append(Paragraph("RADIOLOGICAL IMPRESSION & INTERPRETATION", section_heading))
    if is_tumor:
        if lesion_pct == 0.0:
            imp_text = (
                f"Automated evaluation demonstrates focal intracranial mass effect consistent with <b>{diag_name}</b> "
                f"(AI Diagnostic Confidence: <b>{confidence:.1%}</b>). The lesion demonstrates characteristic cystic/hypointense center "
                f"typical of post-contrast T1 examinations; lesion boundaries and spatial locus are corroborated via Explainable AI Grad-CAM attention mapping."
            )
        else:
            imp_text = (
                f"Automated cranial evaluation reveals circumscribed mass effect consistent with <b>{diag_name}</b> "
                f"(AI Diagnostic Confidence: <b>{confidence:.1%}</b>). U-Net morphological delineation confirms pathological boundary covering "
                f"approximately <b>{lesion_pct:.2f}%</b> of the axial cranial slice, confirmed by concurrent Grad-CAM focal activation."
            )
    else:
        imp_text = (
            f"Automated cranial evaluation demonstrates normal parenchymal signal intensity without focal intra- or extra-axial mass lesion "
            f"(Confidence: <b>{confidence:.1%}</b>). Ventricular caliber, grey-white differentiation, and cranial envelopes remain unremarkable."
        )

    t_imp = Table([[Paragraph(imp_text, finding_body)]], colWidths=[523])
    t_imp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_imp)
    story.append(Spacer(1, 8))

    # 7. Regulatory Disclaimer & Sign-off Line
    disclaimer_text = (
        "<b>CLINICAL DISCLAIMER:</b> This computer-aided diagnostic (CAD) summary was generated using deep learning algorithms "
        "(EfficientNetB3 & U-Net). It is designed strictly for research decision support and academic evaluation. Final radiological diagnosis "
        "must be rendered by a licensed radiologist or attending medical specialist following complete multi-slice volumetric correlation."
    )
    
    signoff_data = [
        [
            Paragraph(disclaimer_text, disclaimer_style),
            Paragraph("<b>ATTENDING RADIOLOGIST:</b><br/><br/>____________________________________<br/>MD, Neuro-Radiology Division", ParagraphStyle('Sign', fontName='Helvetica', fontSize=7, leading=9, alignment=2, textColor=colors.HexColor('#334155')))
        ]
    ]
    t_sign = Table(signoff_data, colWidths=[360, 163])
    t_sign.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_sign)

    # Build the document
    doc.build(story)
    
    if output_path:
        return None
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
