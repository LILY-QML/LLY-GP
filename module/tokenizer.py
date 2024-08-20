class Tokenizer:
    def __init__(self, token_length=25):
        self.token_length = token_length  # Setze die Token-Länge auf 25 Zeichen
        self.float_components = 3  # Jeder Buchstabe wird durch drei Werte repräsentiert: ASCII, Position, Kontext

    def tokenize(self, word):
        # Kürzen oder auffüllen, um die richtige Länge zu erreichen
        word = self.prepare_word(word)

        # Erzeuge den Token
        token = []
        for char in word:
            floats = self.char_to_floats(char, word)
            token.append(floats)

        return token

    def prepare_word(self, word):
        # Konvertiere das Wort zu einer Länge von 25 Zeichen
        if len(word) > self.token_length:
            # Kürze das Wort
            return word[:self.token_length]
        else:
            # Fülle das Wort auf mit einem häufigen Buchstaben, z.B. 'X'
            filler = 'X' * (self.token_length - len(word))
            return word + filler

    def char_to_floats(self, char, word):
        # ASCII-Basierter Wert (zwischen 0 und 1)
        ascii_value = ord(char)
        ascii_normalized = ascii_value / 255.0  # Normierung auf [0,1]

        # Positionsbasierter Wert (zwischen 0 und 1)
        position_value = word.index(char) / len(word)

        # Kontextueller Einfluss: Verhältnis des ASCII-Werts zu einem häufigen Buchstaben (z.B. 'E')
        common_letter_value = ord('E') / 255.0
        context_value = ascii_normalized / common_letter_value

        return (ascii_normalized, position_value, context_value)
