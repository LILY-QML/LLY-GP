from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime

class Visual:
    def __init__(self, filename="LLY-GP_Report.pdf"):
        self.filename = filename
        self.styles = getSampleStyleSheet()

    def generate_report(self, circuit_image_path, token_matrix, training_matrix, initial_heatmap_path, final_heatmap_path, loss_plot_path, histogram_path):
        # Create the PDF document
        doc = SimpleDocTemplate(self.filename, pagesize=letter)
        story = []

        # Add title page
        self.add_title_page(story)

        # Add table of contents
        self.add_table_of_contents(story)

        # Add Introduction
        self.add_introduction(story)

        # Add Start Values (Matrices and Quantum Circuit)
        self.add_start_values(story, token_matrix, training_matrix, circuit_image_path)

        # Add Training section with Heatmaps and Loss Plot
        self.add_training_section(story, initial_heatmap_path, final_heatmap_path, loss_plot_path)

        # Add Changes section with Histogram
        self.add_changes_section(story, histogram_path)

        # Build the PDF document
        doc.build(story)

    def add_title_page(self, story):
        title_style = ParagraphStyle(
            name="Title",
            fontSize=60,  
            alignment=0,  
            textColor=colors.grey,  
        )
        subtitle_style = ParagraphStyle(
            name="Subtitle",
            fontSize=20,  
            alignment=0,  
            textColor=colors.grey,  
        )
        info_style = ParagraphStyle(
            name="Info",
            fontSize=14,
            alignment=0,  
            textColor=colors.grey,
        )
        block_text_style = ParagraphStyle(
            name="BlockText",
            fontSize=12,
            alignment=4,  
            spaceBefore=20,  
        )
        license_text_style = ParagraphStyle(
            name="LicenseText",
            fontSize=12,
            alignment=0,  
            spaceBefore=20,  
        )

        title_paragraph = Paragraph("LLY-GP", title_style)
        subtitle_paragraph = Paragraph("Quantum Machine Learning Report", subtitle_style)
        version_paragraph = Paragraph("Version 1.0 Beta", info_style)
        date_paragraph = Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", info_style)

        block_text = """
        LLY-GP is part of the LILY Project and focuses on optimization parameter-based quantum circuits. It enhances the efficiency of quantum algorithms by fine-tuning parameters of quantum gates. GP stands for Generativ Qubit Processing, which assigns each word the state of a multi-qubit system and recognizes words through a quantum machine learning process. This approach leverages gradient-based optimization techniques to improve the performance of quantum circuits.

        The primary goal of LLY-GP is to recognize and assign languages, making it a foundational element in the development of language-aware models. As the LILY Project evolves, LLY-GP will become increasingly important, serving as a critical component in more advanced quantum machine learning models.

        LLY-GP is available on the LILY QML platform, making it accessible for researchers and developers.

        For inquiries or further information, please contact: info@lilyqml.de.
        """
        block_text_paragraph = Paragraph(block_text, block_text_style)

        license_text = "This model is licensed under the MIT License."
        license_paragraph = Paragraph(license_text, license_text_style)

        story.append(Spacer(1, 50))  
        story.append(title_paragraph)
        story.append(Spacer(1, 50))  
        story.append(subtitle_paragraph)
        story.append(Spacer(1, 60))  
        story.append(HRFlowable(width="100%", thickness=2, color=colors.grey, spaceBefore=1, spaceAfter=1))
        story.append(Spacer(1, 30))
        story.append(version_paragraph)
        story.append(date_paragraph)
        story.append(Spacer(1, 40))  
        story.append(block_text_paragraph)
        story.append(Spacer(1, 40))  
        story.append(license_paragraph)
        story.append(PageBreak())

    def add_table_of_contents(self, story):
        toc_style = ParagraphStyle(name="TOC", fontSize=18, spaceAfter=20)
        story.append(Paragraph("Table of Contents", toc_style))
        toc_content = """
        1. Introduction<br/>
        2. Start Values<br/>
        3. Training<br/>
        4. Changes<br/>
        """
        story.append(Paragraph(toc_content, self.styles['Normal']))
        story.append(PageBreak())

    def add_introduction(self, story):
        intro_style = ParagraphStyle(name="Intro", fontSize=14, spaceAfter=20)
        intro_content = """
        In this model, each token is assigned a state. Through the use of optimizers, the likelihood of correctly assigning tokens improves over time.
        """
        story.append(Paragraph("1. Introduction", self.styles['Heading1']))
        story.append(Paragraph(intro_content, intro_style))
        story.append(PageBreak())

    def add_start_values(self, story, token_matrix, training_matrix, circuit_image_path):
        story.append(Paragraph("2. Start Values", self.styles['Heading1']))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Token Matrix:", self.styles['Heading2']))
        story.append(Paragraph(str(token_matrix), self.styles['Normal']))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Training Matrix:", self.styles['Heading2']))
        story.append(Paragraph(str(training_matrix), self.styles['Normal']))
        story.append(Spacer(1, 20))
        story.append(Image(circuit_image_path, width=400, height=300))
        story.append(PageBreak())

    def add_training_section(self, story, initial_heatmap_path, final_heatmap_path, loss_plot_path):
        story.append(Paragraph("3. Training", self.styles['Heading1']))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Initial Training Matrix Heatmap:", self.styles['Heading2']))
        story.append(Image(initial_heatmap_path, width=400, height=300))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Optimized Training Matrix Heatmap:", self.styles['Heading2']))
        story.append(Image(final_heatmap_path, width=400, height=300))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Training Loss Curve:", self.styles['Heading2']))
        story.append(Image(loss_plot_path, width=400, height=300))
        story.append(PageBreak())

    def add_changes_section(self, story, histogram_path):
        story.append(Paragraph("4. Changes", self.styles['Heading1']))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Quantum State Probability Distribution:", self.styles['Heading2']))
        story.append(Image(histogram_path, width=400, height=300))
