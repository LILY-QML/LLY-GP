import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
from qiskit.visualization import plot_histogram
import os
import seaborn as sns


class Visual:
    def __init__(
        self,
        results,
        target_states,
        initial_training_phases,
        activation_matrices,
        circuits,
        num_iterations,
        qubits,
        depth,
    ):
        self.results = results
        self.target_states = target_states
        self.initial_training_phases = initial_training_phases
        self.activation_matrices = activation_matrices
        self.circuits = circuits
        self.num_iterations = num_iterations
        self.qubits = qubits
        self.depth = depth
        self.styles = getSampleStyleSheet()

    def generate_report(self, filename="QuantumCircuitReport.pdf"):
        # Create the document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []

        # Title Page
        self.add_title_page(story)

        # Table of Contents
        self.add_table_of_contents(story)

        # Initiated Data
        self.add_initiated_data(story)

        # List of Optimization Methods
        self.add_optimization_methods(story)

        # Comparison Between Methods
        self.add_comparison_section(story)

        # Final Results
        self.add_final_results_section(story)

        # Build the PDF
        doc.build(story)

    def add_title_page(self, story):
        # Create the title page
        title_page = TitlePage(
            title="LLY-DML",
            subtitle="Part of the LILY Project",
            copyright_info="""Copyright Protection and All Rights Reserved.<br/>
            Contact: <a href="mailto:info@lilyqml.de">info@lilyqml.de</a><br/>
            Website: <a href="http://lilyqml.de">lilyqml.de</a>""",
            description="""<hr/>
            This is LLY-DML, a model of the LILY Quantum Machine Learning Project.<br/>
            Its task is to train datasets to a state using so-called L-Gates, quantum machine learning gates.<br/>
            Input data is used in parts of the machine learning gates, and other phases are optimized so that a state becomes particularly likely.<br/>
            <hr/>""",
            date=datetime.now().strftime("%d.%m.%Y"),
            additional_info="""<b>Date:</b> 01.08.2024<br/>
            <b>Author:</b> LILY Team<br/>
            <b>Version:</b> 1.0<br/>
            <b>Contact:</b> info@lilyqml.de<br/>
            <b>Website:</b> <a href="http://lilyqml.de">lilyqml.de</a><br/>""",
        )
        title_page.build(story, self.styles)

    def add_table_of_contents(self, story):
        toc = TableOfContents(
            contents=[
                "<link href='#section1' color='blue'>1. Initiated Data</link>",
                "<link href='#section2' color='blue'>2. List of Optimization Methods</link>",
                "<link href='#section2.1' color='blue'>2.1 Basic Gradient Descent (GD)</link>",
                "<link href='#section2.2' color='blue'>2.2 Momentum</link>",
                "<link href='#section2.3' color='blue'>2.3 Adam (Adaptive Moment Estimation)</link>",
                "<link href='#section2.4' color='blue'>2.4 Genetic Algorithm (GA)</link>",
                "<link href='#section2.5' color='blue'>2.5 Particle Swarm Optimization (PSO)</link>",
                "<link href='#section2.6' color='blue'>2.6 Bayesian Optimization</link>",
                "<link href='#section2.7' color='blue'>2.7 Simulated Annealing</link>",
                "<link href='#section2.8' color='blue'>2.8 Quantum Natural Gradient (QNG)</link>",
                "<link href='#section3' color='blue'>3. Comparison Between Methods</link>",
                "<link href='#section4' color='blue'>4. Final Results</link>",
            ]
        )
        toc.build(story, self.styles)

    def add_initiated_data(self, story):
        # Add initial data section
        story.append(
            Paragraph("<a name='section1'/>1. Initiated Data", self.styles["Heading2"])
        )
        story.append(Spacer(1, 20))

        # Initial Quantum Circuit
        circuit_image_path = os.path.join("var", "circuit_initial_1.png")
        story.append(
            Paragraph("Initial Quantum Circuit:", self.styles["Heading3"])
        )  # Add a title for the section
        if os.path.exists(circuit_image_path):
            story.append(Image(circuit_image_path, width=400, height=200))
        story.append(Spacer(1, 20))

        # Table with initial data
        data = {
            "Qubits": self.qubits,
            "Depth": self.depth,
            "Shots": self.results[0]["Final Counts"]["shots"],
            "Max Iterations": self.num_iterations,
        }
        data_df = pd.DataFrame([data])
        table = Table([data_df.columns.tolist()] + data_df.values.tolist())
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 20))
        story.append(PageBreak())

    def add_optimization_methods(self, story):
        # Section 2: List of Optimization Methods
        story.append(
            Paragraph(
                "<a name='section2'/>2. List of Optimization Methods",
                self.styles["Heading2"],
            )
        )
        story.append(Spacer(1, 20))

        optimization_methods = [
            OptimizationMethod(
                title="<a name='section2.1'/>Basic Gradient Descent (GD)",
                description="A simple optimization algorithm that updates the parameters in the opposite direction of the gradient of the objective function.",
                use_case="Provides a foundational approach to minimize the loss function by iteratively adjusting phases.",
            ),
            OptimizationMethod(
                title="<a name='section2.2'/>Momentum",
                description="An enhancement of basic gradient descent that accelerates convergence by considering a fraction of the previous update direction, thereby reducing oscillations.",
                use_case="Useful for speeding up convergence, especially in scenarios with a zig-zag path or shallow local minima.",
            ),
            OptimizationMethod(
                title="<a name='section2.3'/>Adam (Adaptive Moment Estimation)",
                description="Combines the benefits of both RMSProp and Momentum, adapting the learning rate for each parameter and maintaining moving averages of both the gradients and their squares.",
                use_case="Suitable for handling sparse gradients and noisy data, often providing faster convergence and more robust performance.",
            ),
            OptimizationMethod(
                title="<a name='section2.4'/>Genetic Algorithm (GA)",
                description="Inspired by natural selection, this algorithm uses operations such as mutation, crossover, and selection to evolve solutions over generations.",
                use_case="Effective for optimization problems where gradient information is unavailable or unreliable. Good for exploring a wide solution space.",
            ),
            OptimizationMethod(
                title="<a name='section2.5'/>Particle Swarm Optimization (PSO)",
                description="A population-based optimization algorithm that simulates social behavior, where particles adjust their positions based on their own and neighbors' experiences.",
                use_case="Useful for multi-modal optimization problems, finding optima in large, complex search spaces without requiring gradient information.",
            ),
            OptimizationMethod(
                title="<a name='section2.6'/>Bayesian Optimization",
                description="Utilizes a probabilistic model to estimate the objective function, focusing on regions with a high probability of finding the minimum.",
                use_case="Particularly suited for expensive-to-evaluate functions, optimizing hyperparameters and other scenarios where evaluations are costly.",
            ),
            OptimizationMethod(
                title="<a name='section2.7'/>Simulated Annealing",
                description="Mimics the annealing process in metallurgy, reducing the 'temperature' over time to escape local minima and find a global minimum.",
                use_case="Effective for discrete or combinatorial optimization problems, especially when the landscape has many local optima.",
            ),
            OptimizationMethod(
                title="<a name='section2.8'/>Quantum Natural Gradient (QNG)",
                description="A quantum-aware optimization technique that considers the geometric properties of parameter space, often leading to better convergence in quantum circuit optimization.",
                use_case="Particularly advantageous in quantum machine learning and quantum circuit optimization where traditional gradient methods fall short.",
            ),
        ]

        for method in optimization_methods:
            method.build(story, self.styles)

    def add_comparison_section(self, story):
        # Section 3: Comparison Between Methods
        story.append(
            Paragraph(
                "<a name='section3'/>3. Comparison Between Methods",
                self.styles["Heading2"],
            )
        )
        story.append(Spacer(1, 20))

        # Create a DataFrame for comparison
        comparison_df = pd.DataFrame(self.results)
        comparison_df["Improvement"] = (
            comparison_df["Final Probability"] - comparison_df["Initial Probability"]
        )

        # Plot comparison as heatmap
        plt.figure(figsize=(10, 8))
        heatmap_data = comparison_df.pivot_table(
            values="Improvement", index="Optimizer", columns="Target State"
        )
        plt.title("Improvement Heatmap")
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
        heatmap_path = os.path.join("var", "heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()

        # Add heatmap image to the report
        story.append(Image(heatmap_path, width=400, height=300))
        story.append(Spacer(1, 20))

        # Add the comparison table
        table_data = [
            [str(i) for i in row] for row in comparison_df.round(4).values.tolist()
        ]  # Convert all elements to strings
        comparison_table = Table([comparison_df.columns.tolist()] + table_data)
        comparison_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(comparison_table)
        story.append(Spacer(1, 20))
        story.append(PageBreak())

    def add_final_results_section(self, story):
        # Section 4: Final Results
        story.append(
            Paragraph("<a name='section4'/>4. Final Results", self.styles["Heading2"])
        )
        story.append(Spacer(1, 20))

        # Determine the best optimizer
        final_df = pd.DataFrame(self.results)
        final_df["Improvement"] = (
            final_df["Final Probability"] - final_df["Initial Probability"]
        )
        best_optimizer = final_df.loc[
            final_df["Improvement"].idxmax(), "Optimizer"
        ]

        final_results_content = f"The most effective optimization method was <b>{best_optimizer}</b>, which achieved the highest improvement in target state probability."

        final_results_section = FinalResultsSection(content=final_results_content)
        final_results_section.build(story, self.styles)


class TitlePage:
    def __init__(
        self, title, subtitle, copyright_info, description, date, additional_info
    ):
        self.title = title
        self.subtitle = subtitle
        self.copyright_info = copyright_info
        self.description = description
        self.date = date
        self.additional_info = additional_info

    def build(self, story, styles):
        # Custom styles for specific formatting
        title_style = ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontSize=36,
            textColor=colors.HexColor("#000080"),  # Dunkelblau
            spaceAfter=24,  # Abstand nach der Hauptüberschrift
        )
        subtitle_style = ParagraphStyle(
            "subtitle",
            parent=styles["Heading2"],
            fontSize=16,
            spaceAfter=5,  # Weniger Abstand zwischen Untertitel und Copyright-Text
        )
        normal_style = ParagraphStyle(
            "normal",
            parent=styles["Normal"],
            alignment=4,  # Blocksatz
            spaceBefore=10,  # Abstand vor dem Text
        )
        big_heading_style = ParagraphStyle(
            "bigHeading",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#000080"),  # Dunkelblau
            spaceAfter=20,  # Abstand nach der Überschrift
        )

        # Paragraphs with the new styles
        title_paragraph = Paragraph(self.title, title_style)
        subtitle_paragraph = Paragraph(self.subtitle, subtitle_style)
        copyright_paragraph = Paragraph(self.copyright_info, normal_style)
        description_paragraph = Paragraph(self.description, normal_style)
        big_heading_paragraph = Paragraph(
            "QUANTUM LLY-DML TRAINING REPORT", big_heading_style
        )
        additional_info_paragraph = Paragraph(self.additional_info, normal_style)

        date_paragraph = Paragraph(
            f"""<hr/>
            This report shows all data related to the training conducted on: <b>{self.date}</b>
            <hr/>""",
            normal_style,
        )

        # Adding content to the story
        story.extend(
            [
                title_paragraph,
                subtitle_paragraph,
                copyright_paragraph,
                Spacer(1, 40),  # Adjusted space between copyright and description
                description_paragraph,
                Spacer(1, 20),
                big_heading_paragraph,
                additional_info_paragraph,
                Spacer(1, 40),
                date_paragraph,
                Spacer(1, 40),
                PageBreak(),  # Page break for the table of contents
            ]
        )


class TableOfContents:
    def __init__(self, contents):
        self.contents = contents

    def build(self, story, styles):
        toc_title_style = styles["Heading2"]
        toc_title = Paragraph("Table of Contents", toc_title_style)

        # Create paragraphs for each item in the table of contents
        toc_entries = [toc_title]
        toc_style = ParagraphStyle(
            "toc",
            parent=styles["Normal"],
            spaceBefore=5,
            spaceAfter=5,
            leftIndent=20,
            fontSize=12,
        )

        for entry in self.contents:
            toc_entry = Paragraph(entry, toc_style)
            toc_entries.append(toc_entry)

        # Adding content to the story
        story.extend(toc_entries + [Spacer(1, 40)])


class OptimizationMethod:
    def __init__(self, title, description, use_case):
        self.title = title
        self.description = description
        self.use_case = use_case

    def build(self, story, styles):
        method_title = Paragraph(f"{self.title}", styles["Heading3"])
        method_description = Paragraph(
            f"<b>Description:</b> {self.description}", styles["Normal"]
        )
        method_use_case = Paragraph(
            f"<b>Use Case:</b> {self.use_case}", styles["Normal"]
        )

        # Adding content to the story
        story.extend(
            [
                method_title,
                Spacer(1, 10),
                method_description,
                Spacer(1, 5),
                method_use_case,
                Spacer(1, 20),
            ]
        )


class ComparisonSection:
    def __init__(self, content):
        self.content = content

    def build(self, story, styles):
        comparison_title_style = styles["Heading2"]
        comparison_title = Paragraph("Comparison Between Methods", comparison_title_style)

        comparison_content = Paragraph(self.content, styles["Normal"])

        # Adding content to the story
        story.extend(
            [
                comparison_title,
                Spacer(1, 20),
                comparison_content,
                Spacer(1, 40),
            ]
        )


class FinalResultsSection:
    def __init__(self, content):
        self.content = content

    def build(self, story, styles):
        final_results_title_style = styles["Heading2"]
        final_results_title = Paragraph("Final Results", final_results_title_style)

        final_results_content = Paragraph(self.content, styles["Normal"])

        # Adding content to the story
        story.extend(
            [
                final_results_title,
                Spacer(1, 20),
                final_results_content,
                Spacer(1, 40),
            ]
        )
