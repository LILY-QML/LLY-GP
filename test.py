import unittest
import numpy as np
from module.circuit import Circuit
from module.tokenizer import Tokenizer
from unittest.mock import patch

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
        expected_operations = self.qubits * 2 * 3  # 2 L-Gates mit jeweils 3 Operationen pro Qubit
        actual_operations = len(self.circuit.circuit.data)
        self.assertEqual(actual_operations, expected_operations, f"Der Schaltkreis sollte {expected_operations} Operationen enthalten, hat aber {actual_operations}")

    def test_measure(self):
        # Testet, ob die Messungen korrekt hinzugefügt werden
        self.circuit.create_circuit()
        self.circuit.measure()
        expected_operations = self.qubits * 2 * 3 + self.qubits  # Zusätzliche Messungen
        actual_operations = len(self.circuit.circuit.data)
        self.assertEqual(actual_operations, expected_operations, f"Der Schaltkreis sollte nach Messungen {expected_operations} Operationen enthalten, hat aber {actual_operations}")

    def test_run(self):
        # Mocken der Ausführung des Schaltkreises, um die Ergebnisse zu simulieren
        with patch('module.circuit.Aer.get_backend') as mock_get_backend:
            mock_get_backend.return_value = Aer.get_backend('aer_simulator')
            self.circuit.create_circuit()
            self.circuit.measure()
            results = self.circuit.run(shots=100)
            self.assertIsInstance(results, dict, "Das Ergebnis sollte ein Dictionary sein")
            self.assertGreaterEqual(sum(results.values()), 100, "Die Summe der Ergebnisse sollte mindestens 100 sein")

    def test_complete_run(self):
        # Testet die vollständige Ausführung, bis alle Zustände gemessen wurden
        self.circuit.create_circuit()
        self.circuit.measure()
        results = self.circuit.complete_run()
        num_possible_states = 2 ** self.qubits
        self.assertEqual(len(results), num_possible_states, f"Alle möglichen {num_possible_states} Zustände sollten gemessen werden")

    def test_random_training_matrix(self):
        # Testet die Erstellung einer zufälligen Trainingsmatrix und die Schaltkreisintegration
        new_training_matrix = np.random.rand(self.qubits, 3)
        self.circuit = Circuit(self.input_matrix, new_training_matrix, self.qubits)
        self.circuit.create_circuit()
        self.assertEqual(len(self.circuit.circuit.data), self.qubits * 2 * 3, "Der Schaltkreis sollte 60 Operationen enthalten nach zufälliger Trainingsmatrix")

if __name__ == '__main__':
    unittest.main()
