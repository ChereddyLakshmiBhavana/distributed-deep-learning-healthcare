"""
Medical Report PDF Generator
Generates professional medical reports for pneumonia detection analysis
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, 
    Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from PIL import Image as PILImage
import os
import numpy as np


class MedicalReportGenerator:
    """Generate professional medical reports for X-ray analysis"""
    
    def __init__(self, output_dir='reports'):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for medical reports"""
        
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Section header style
        self.section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#2c5aa0'),
            borderPadding=5,
            backColor=colors.HexColor('#e8f4f8')
        )
        
        # Result text style (for PNEUMONIA/NORMAL)
        self.result_normal_style = ParagraphStyle(
            'ResultNormal',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#28a745'),
            spaceAfter=15,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        )
        
        self.result_pneumonia_style = ParagraphStyle(
            'ResultPneumonia',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#dc3545'),
            spaceAfter=15,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        )
        
        # Clinical note style
        self.clinical_note_style = ParagraphStyle(
            'ClinicalNote',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            spaceAfter=20,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=10,
            backColor=colors.HexColor('#f9f9f9'),
            alignment=TA_JUSTIFY
        )
    
    def generate_report(self, prediction_data):
        """
        Generate complete medical report PDF
        
        Args:
            prediction_data: Dictionary containing:
                - prediction: 'NORMAL' or 'PNEUMONIA'
                - confidence: float (0.0-1.0)
                - model_name: name of model used
                - image_path: path to X-ray image
                - all_models_results: results from all models (optional)
                - patient_id: patient identifier (optional)
                
        Returns:
            str: Path to generated PDF file
        """
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        patient_id = prediction_data.get('patient_id', 'UNKNOWN')
        filename = f"xray_report_{patient_id}_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Container for PDF elements
        elements = []
        
        # Add header
        elements.extend(self._add_header())
        
        # Add metadata table
        elements.extend(self._add_metadata(prediction_data))
        
        # Add main result
        elements.extend(self._add_prediction_result(prediction_data))
        
        # Add X-ray image
        elements.extend(self._add_xray_image(prediction_data.get('image_path')))
        
        # Add model comparison if available
        if prediction_data.get('all_models_results'):
            elements.extend(self._add_model_comparison(prediction_data['all_models_results']))
        
        # Add confidence interpretation
        elements.extend(self._add_confidence_interpretation(prediction_data))
        
        # Add clinical disclaimer
        elements.extend(self._add_clinical_disclaimer())
        
        # Add signature section
        elements.extend(self._add_signature_section())
        
        # Add footer
        elements.extend(self._add_footer())
        
        # Build PDF
        doc.build(elements)
        
        return filepath
    
    def _add_header(self):
        """Add report header"""
        elements = []
        
        title = Paragraph("CHEST X-RAY ANALYSIS REPORT", self.title_style)
        elements.append(title)
        
        subtitle = Paragraph(
            "AI-Assisted Pneumonia Detection System",
            ParagraphStyle(
                'Subtitle',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#555555'),
                alignment=TA_CENTER,
                spaceAfter=20
            )
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _add_metadata(self, prediction_data):
        """Add metadata table"""
        elements = []
        
        # Generate metadata
        report_date = datetime.now().strftime("%B %d, %Y")
        report_time = datetime.now().strftime("%H:%M:%S")
        exam_id = prediction_data.get('exam_id', 
                                      f"XR-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}")
        
        section_header = Paragraph("EXAMINATION INFORMATION", self.section_header_style)
        elements.append(section_header)
        
        data = [
            ["Examination ID:", exam_id],
            ["Report Date:", report_date],
            ["Report Time:", report_time],
            ["Analysis Model:", prediction_data.get('model_name', 'Random Forest')],
            ["Model Version:", "v1.0"],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _add_prediction_result(self, prediction_data):
        """Add main prediction result"""
        elements = []
        
        section_header = Paragraph("ANALYSIS RESULT", self.section_header_style)
        elements.append(section_header)
        elements.append(Spacer(1, 0.1*inch))
        
        prediction = prediction_data['prediction']
        confidence = prediction_data['confidence']
        
        # Choose style based on prediction
        result_style = (self.result_normal_style if prediction == 'NORMAL' 
                       else self.result_pneumonia_style)
        
        # Create result text with icon
        icon = "✓" if prediction == 'NORMAL' else "⚠"
        result_text = f"{icon} PREDICTION: {prediction}"
        
        result_para = Paragraph(result_text, result_style)
        elements.append(result_para)
        
        # Confidence score
        confidence_pct = confidence * 100
        confidence_text = f"Confidence Level: <b>{confidence_pct:.1f}%</b>"
        
        confidence_style = ParagraphStyle(
            'Confidence',
            parent=self.styles['Normal'],
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=20
        )
        
        confidence_para = Paragraph(confidence_text, confidence_style)
        elements.append(confidence_para)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _add_xray_image(self, image_path):
        """Add X-ray image preview"""
        elements = []
        
        if not image_path or not os.path.exists(image_path):
            elements.append(Paragraph("X-ray image preview unavailable", self.styles['Normal']))
            return elements
        
        section_header = Paragraph("X-RAY IMAGE", self.section_header_style)
        elements.append(section_header)
        elements.append(Spacer(1, 0.1*inch))
        
        try:
            # Create image with max dimensions
            img = RLImage(image_path, width=4*inch, height=4*inch)
            
            # Center the image
            img_table = Table([[img]], colWidths=[7*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            elements.append(img_table)
            elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            elements.append(Paragraph(f"Error loading image: {str(e)}", self.styles['Normal']))
        
        return elements
    
    def _add_model_comparison(self, all_models_results):
        """Add comparison of all models"""
        elements = []
        
        section_header = Paragraph("MODEL COMPARISON", self.section_header_style)
        elements.append(section_header)
        elements.append(Spacer(1, 0.1*inch))
        
        # Create comparison table
        data = [['Model Name', 'Prediction', 'Confidence', 'Status']]
        
        for model_result in all_models_results:
            model_name = model_result['model']
            prediction = model_result['prediction']
            confidence = f"{model_result['confidence']*100:.1f}%"
            status = "✓" if model_result.get('agrees_with_majority', True) else "✗"
            
            data.append([model_name, prediction, confidence, status])
        
        table = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.3*inch, 0.7*inch])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add consensus note
        pneumonia_count = sum(1 for r in all_models_results if r['prediction'] == 'PNEUMONIA')
        total_models = len(all_models_results)
        
        consensus_text = f"<b>Consensus:</b> {pneumonia_count} out of {total_models} models predict PNEUMONIA"
        consensus_para = Paragraph(consensus_text, self.styles['Normal'])
        elements.append(consensus_para)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _add_confidence_interpretation(self, prediction_data):
        """Add interpretation of confidence level"""
        elements = []
        
        confidence = prediction_data['confidence']
        
        section_header = Paragraph("CONFIDENCE INTERPRETATION", self.section_header_style)
        elements.append(section_header)
        
        # Interpret confidence level
        if confidence >= 0.90:
            interpretation = "Very High Confidence: The model is very confident in this prediction."
        elif confidence >= 0.75:
            interpretation = "High Confidence: The model shows strong confidence in this prediction."
        elif confidence >= 0.60:
            interpretation = "Moderate Confidence: The prediction should be verified by a radiologist."
        else:
            interpretation = "Low Confidence: Manual review by a radiologist is strongly recommended."
        
        interp_para = Paragraph(interpretation, self.styles['Normal'])
        elements.append(interp_para)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _add_clinical_disclaimer(self):
        """Add clinical disclaimer"""
        elements = []
        
        section_header = Paragraph("IMPORTANT CLINICAL NOTICE", self.section_header_style)
        elements.append(section_header)
        
        disclaimer_text = """
        This report is generated by an AI-assisted diagnostic support system and is intended 
        for use only as a clinical decision support tool. <b>This analysis does NOT constitute 
        a medical diagnosis.</b> Final diagnosis and treatment decisions must be made by qualified 
        medical professionals based on clinical judgment, patient history, and additional diagnostic 
        information. This system should be used in conjunction with, not as a replacement for, 
        professional radiological expertise.
        <br/><br/>
        The system's predictions are based on machine learning models trained on historical chest 
        X-ray data and may not account for all clinical factors or imaging artifacts. Medical 
        professionals should independently verify all findings before making clinical decisions.
        """
        
        disclaimer_para = Paragraph(disclaimer_text, self.clinical_note_style)
        elements.append(disclaimer_para)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _add_signature_section(self):
        """Add signature section for healthcare providers"""
        elements = []
        
        section_header = Paragraph("REVIEW AND AUTHORIZATION", self.section_header_style)
        elements.append(section_header)
        elements.append(Spacer(1, 0.15*inch))
        
        signature_data = [
            ["Reviewing Radiologist:", "_" * 50, "Date:", "_" * 20],
            ["", "", "", ""],
            ["Medical License Number:", "_" * 50, "Signature:", "_" * 20],
            ["", "", "", ""],
            ["Institution/Facility:", "_" * 50, "Stamp/Seal:", ""],
        ]
        
        sig_table = Table(signature_data, colWidths=[1.8*inch, 2.7*inch, 0.8*inch, 1.7*inch])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('ALIGN', (3, 0), (3, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(sig_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _add_footer(self):
        """Add report footer"""
        elements = []
        
        footer_text = f"""
        <para align=center>
        ─────────────────────────────────────────────────────────────────<br/>
        <font size=9 color="#666666">
        Generated by: AI Pneumonia Detection System v1.0<br/>
        Report Generated: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}<br/>
        © 2026 Distributed Deep Learning Smart Healthcare Project
        </font>
        </para>
        """
        
        footer_para = Paragraph(footer_text, self.styles['Normal'])
        elements.append(footer_para)
        
        return elements


# Standalone function for easy use
def generate_medical_report(prediction, confidence, model_name, image_path, 
                           all_models_results=None, output_dir='reports'):
    """
    Convenience function to generate a medical report
    
    Args:
        prediction: 'NORMAL' or 'PNEUMONIA'
        confidence: float (0.0-1.0)
        model_name: name of the model
        image_path: path to X-ray image
        all_models_results: optional list of results from all models
        output_dir: directory to save reports
        
    Returns:
        str: path to generated PDF
    """
    
    generator = MedicalReportGenerator(output_dir=output_dir)
    
    prediction_data = {
        'prediction': prediction,
        'confidence': confidence,
        'model_name': model_name,
        'image_path': image_path,
        'all_models_results': all_models_results
    }
    
    return generator.generate_report(prediction_data)
