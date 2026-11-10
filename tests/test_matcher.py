import random
import unittest
from matcher import Matcher


class MatcherTests(unittest.TestCase):
    def test_exact_whitespace_overlap_and_multiple_words(self):
        matcher = Matcher(['[NO_REPLY]', '[SILENCE]', 'A', 'A B', 'B C'])
        for text in ['[NO_REPLY]', '\n [SILENCE]\t', 'A', ' A B ', 'B C']:
            self.assertTrue(matcher.matches(text, 'exact'), text)
        for text in ['', ' ', '[NO_REPLY].', 'text [NO_REPLY]', 'A B C', '[NO_REPLY][SILENCE]']:
            self.assertFalse(matcher.matches(text, 'exact'), text)

    def test_all_marker_chunk_boundaries(self):
        for mode in ['contains', 'exact']:
            matcher = Matcher(['[NO_REPLY]', '不回复'])
            for text in ['[NO_REPLY]', '不回复']:
                for cut in range(len(text) + 1):
                    cursor = matcher.cursor(mode)
                    cursor.feed(text[:cut])
                    cursor.feed(text[cut:])
                    self.assertTrue(cursor.matches)

    def test_unicode_casefold_and_literal_symbols(self):
        matcher = Matcher(['Straße', '[.*]'], True)
        self.assertTrue(matcher.matches(' STRASSE ', 'exact'))
        self.assertTrue(matcher.matches('前[.*]后', 'contains'))
        self.assertFalse(matcher.matches('[abc]', 'contains'))

    def test_suffix_failure_links(self):
        matcher = Matcher(['he', 'she', 'hers', 'his'])
        self.assertTrue(matcher.matches('ushers', 'contains'))
        self.assertFalse(matcher.matches('ushers', 'exact'))

    def test_randomized_against_simple_oracle(self):
        rng = random.Random(20261111)
        for _ in range(1200):
            words = list({''.join(rng.choice('ab中 ß') for _ in range(rng.randint(1, 8))).strip() for _ in range(8)} - {''})
            text = ''.join(rng.choice('ab中 ßS') for _ in range(rng.randint(0, 60)))
            matcher = Matcher(words, True)
            normalized_words = [w.casefold() for w in words]
            for mode in ['exact', 'contains']:
                cursor = matcher.cursor(mode)
                for char in text:
                    cursor.feed(char)
                expected = text.strip().casefold() in normalized_words if mode == 'exact' else any(w in text.casefold() for w in normalized_words)
                self.assertEqual(cursor.matches, expected, (mode, words, text))

    def test_invalid_empty_markers(self):
        for words in [[], [' ', ''], '[NO_REPLY]']:
            with self.assertRaises(ValueError):
                Matcher(words)


if __name__ == '__main__':
    unittest.main()
