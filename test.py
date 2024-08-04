import unittest
import numpy as np
import os
import pandas as pd
from module.circuit import Circuit
from module.tokenizer import Tokenizer
from module.optimizer import AdamOptimizer  # Import the optimizer to be tested
from module.visual import Visual, TitlePage, TableOfContents, FinalResultsSection
from unittest.mock import patch, MagicMock
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_tokenize_length(self):
        # Testet, ob der Token eine Länge von 20 hat
        word = "Quantencomputer"
        token = self.tokenizer.tokenize(word)
        self.assertEqual(len(token), 20, "Der Token sollte eine Länge von 20 haben")

    def test_tokenize_floats_per_char(self):
        # Testet, ob jeder Buchstabe im Token durch 3 Float-Werte repräsentiert wird
        word = "Wissenschaft"
        token = self.tokenizer.tokenize(word)
        for floats in token:
            self.assertEqual(len(floats), 3, "Jeder Buchstabe sollte durch 3 Float-Werte repräsentiert werden")

    def test_tokenize_filled_token(self):
        # Testet, ob kürzere Wörter korrekt gefüllt werden
        word = "Hi"
        token = self.tokenizer.tokenize(word)
        expected_filler_char = 'X'
        # Der letzte Charakter sollte ein Füllcharakter sein
        self.assertEqual(token[-1][0], ord(expected_filler_char) / 255.0, "Das Token sollte mit 'X' gefüllt werden")

    def test_tokenizer_consistency(self):
        # Testet, ob das Token für das gleiche Wort konsistent ist
        word = "Konsistenz"
        token1 = self.tokenizer.tokenize(word)
        token2 = self.tokenizer.tokenize(word)
        self.assertEqual(token1, token2, "Das Token sollte konsistent für das gleiche Wort sein")


class TestCircuit(unittest.TestCase):
    def setUp(self):
        # Setup für die Tests mit einer Beispiel-Input-Matrix und einer zufälligen Trainingsmatrix
        self.qubits = 5  # Reduzierte Anzahl von Qubits für schnellere Tests
        self.tokenizer = Tokenizer()
        word = "Testwort"
        self.input_matrix = np.array(self.tokenizer.tokenize(word))
        self.training_matrix = np.random.rand(self.qubits, 3)
        self.circuit = Circuit(self.input_matrix, self.training_matrix, self.qubits)

    def test_create_circuit(self):
        # Testet, ob der Schaltkreis korrekt erstellt wird
        self.circuit.create_circuit()
        self.assertTrue(self.circuit.circuit, "Der Schaltkreis sollte nach der Erstellung Operationen enthalten")

    def test_measure(self):
        # Testet, ob die Messungen korrekt hinzugefügt werden
        self.circuit.create_circuit()
        self.circuit.measure()
        self.assertIn("Measure", self.circuit.circuit, "Der Schaltkreis sollte eine Messoperation enthalten")

    def test_run(self):
        # Mocken der Ausführung des Schaltkreises, um die Ergebnisse zu simulieren
        with patch.object(self.circuit, 'get_counts', return_value={"00": 50, "01": 30, "10": 10, "11": 10}):
            self.circuit.run()
            results = self.circuit.get_counts()
            self.assertIsInstance(results, dict, "Das Ergebnis sollte ein Dictionary sein")
            self.assertGreaterEqual(sum(results.values()), 100, "Die Summe der Ergebnisse sollte mindestens 100 sein")

    def test_complete_run(self):
        # Testet die vollständige Ausführung, bis alle Zustände gemessen wurden
        with patch.object(self.circuit, 'get_counts', return_value={"00": 50, "01": 30, "10": 10, "11": 10}):
            results = self.circuit.complete_run()
            self.assertIsInstance(results, dict, "Das Ergebnis der vollständigen Ausführung sollte ein Dictionary sein")


class TestAdamOptimizer(unittest.TestCase):
    def setUp(self):
        # Setup für die Tests mit einer Beispiel-Input-Matrix und einer zufälligen Trainingsmatrix
        self.qubits = 5  # Reduzierte Anzahl von Qubits für schnellere Tests
        self.tokenizer = Tokenizer()
        word = "Optimierung"
        self.input_matrix = np.array(self.tokenizer.tokenize(word))
        self.training_matrix = np.random.rand(self.qubits, 3)
        self.circuit = Circuit(self.input_matrix, self.training_matrix, self.qubits)
        self.target_state = '00'  # Definiere einen Zielzustand
        self.optimizer = AdamOptimizer(
            circuit=self.circuit,
            target_state=self.target_state,
            learning_rate=0.01,
            max_iterations=10  # Reduzierte Anzahl von Iterationen für Tests
        )

    def test_initial_loss(self):
        # Testet, ob der anfängliche Verlust korrekt berechnet wird
        self.circuit.run()
        initial_counts = self.circuit.get_counts()
        initial_loss = self.optimizer.loss_function(initial_counts)
        self.assertIsInstance(initial_loss, float, "Der anfängliche Verlust sollte ein Float-Wert sein")
        self.assertLessEqual(initial_loss, 0, "Der Verlust sollte negativ oder null sein, da er den negativen Zielzustands-Wahrscheinlichkeit darstellt")

    def test_optimize(self):
        # Testet die Optimierung, um zu überprüfen, ob die Phasen sich ändern
        initial_phases = np.copy(self.circuit.training_phases)
        optimized_phases, losses = self.optimizer.optimize()
        self.assertNotEqual(initial_phases.tolist(), optimized_phases, "Die optimierten Phasen sollten sich von den anfänglichen Phasen unterscheiden")
        self.assertEqual(len(losses), self.optimizer.max_iterations, "Die Anzahl der Verlusteinträge sollte der maximalen Anzahl von Iterationen entsprechen")
        self.assertIsInstance(losses, list, "Die Verluste sollten als Liste zurückgegeben werden")

    def test_loss_trend(self):
        # Testet, ob der Verlust während der Optimierung abnimmt
        _, losses = self.optimizer.optimize()
        # Überprüfe, ob die Verluste im Laufe der Iterationen abnehmen
        for earlier_loss, later_loss in zip(losses, losses[1:]):
            self.assertLessEqual(later_loss, earlier_loss, "Der Verlust sollte im Laufe der Iterationen abnehmen oder gleich bleiben")


class TestVisual(unittest.TestCase):
    def setUp(self):
        # Setup a mock dataset
        self.results = [
            {
                "Optimizer": "Adam",
                "Initial Probability": 0.2,
                "Final Probability": 0.8,
                "Final Counts": {"00": 80, "01": 10, "10": 5, "11": 5, "shots": 100},
                "Target State": "00"
            },
            {
                "Optimizer": "Momentum",
                "Initial Probability": 0.25,
                "Final Probability": 0.75,
                "Final Counts": {"00": 75, "01": 15, "10": 5, "11": 5, "shots": 100},
                "Target State": "00"
            },
        ]
        self.target_states = ["00", "01", "10", "11"]
        self.initial_training_phases = np.random.rand(5, 3) * 2 * np.pi
        self.activation_matrices = [np.random.rand(5, 5) for _ in range(2)]
        self.circuits = [MagicMock(), MagicMock()]
        self.num_iterations = 100
        self.qubits = 5
        self.depth = 3
        self.visual = Visual(
            self.results,
            self.target_states,
            self.initial_training_phases,
            self.activation_matrices,
            self.circuits,
            self.num_iterations,
            self.qubits,
            self.depth,
        )
        # Load the stylesheet for report creation
        self.styles = getSampleStyleSheet()

    def test_generate_report_creates_file(self):
        # Testet, ob die Berichtserstellung tatsächlich eine PDF-Datei generiert
        filename = "test_QuantumCircuitReport.pdf"
        self.visual.generate_report(filename=filename)
        self.assertTrue(os.path.exists(filename), "Der Bericht sollte als PDF-Datei erstellt werden")
        os.remove(filename)  # Clean up the generated file

    def test_title_page_in_report(self):
        # Testet, ob die Titelseite korrekt erstellt wird
        story = []
        title_page = TitlePage(
            title="Test Title",
            subtitle="Test Subtitle",
            copyright_info="Test Copyright",
            description="Test Description",
            date="01.01.2024",
            additional_info="Test Additional Info",
        )
        title_page.build(story, self.styles)
        # Prüft, ob alle Absätze korrekt hinzugefügt wurden
        self.assertEqual(len(story), 12, "Es sollten 12 Elemente auf der Titelseite vorhanden sein")

    def test_table_of_contents_in_report(self):
        # Testet, ob das Inhaltsverzeichnis korrekt erstellt wird
        story = []
        toc = TableOfContents(contents=["<link href='#section1' color='blue'>1. Test Section</link>"])
        toc.build(story, self.styles)
        # Prüft, ob das Inhaltsverzeichnis hinzugefügt wurde
        self.assertEqual(len(story), 3, "Das Inhaltsverzeichnis sollte 3 Elemente enthalten")

    def test_add_optimization_methods(self):
        # Testet, ob die Optimierungsmethoden zur Geschichte hinzugefügt werden
        story = []
        self.visual.add_optimization_methods(story)
        # Überprüft, ob die richtige Anzahl von Optimierungsmethoden hinzugefügt wurde
        self.assertGreaterEqual(len(story), 16, "Es sollten mindestens 16 Absätze zur Beschreibung der Optimierungsmethoden vorhanden sein")

    def test_add_comparison_section(self):
        # Testet, ob der Vergleich korrekt zur Geschichte hinzugefügt wird
        story = []
        self.visual.add_comparison_section(story)
        # Überprüft, ob die Vergleichsdaten und der Plot hinzugefügt wurden
        self.assertGreaterEqual(len(story), 3, "Der Vergleichsabschnitt sollte mindestens 3 Elemente enthalten")
        # Prüft, ob die Heatmap-Datei erstellt wurde
        self.assertTrue(os.path.exists(os.path.join("var", "heatmap.png")), "Die Heatmap sollte als PNG-Datei erstellt werden")
        os.remove(os.path.join("var", "heatmap.png"))  # Clean up the generated file

    def test_final_results_section(self):
        # Testet, ob die Endergebnisse korrekt zur Geschichte hinzugefügt werden
        story = []
        final_results_section = FinalResultsSection(content="Test Final Results Content")
        final_results_section.build(story, self.styles)
        # Überprüft, ob der Abschnitt der Endergebnisse hinzugefügt wurde
        self.assertEqual(len(story), 4, "Der Abschnitt der Endergebnisse sollte 4 Elemente enthalten")


if __name__ == "__main__":
    unittest.main()
