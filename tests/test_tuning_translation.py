import unittest
import feedpak_common as fc

class TuningTranslationTests(unittest.TestCase):
    def check(self, source, is_bass, expected):
        actual, unsupported = fc.tuning_offsets_from_absolute(source, is_bass)
        self.assertEqual(actual, expected)
        self.assertFalse(unsupported)
    def test_spellbound_guitar(self):
        self.check([64,59,55,50,45,40,33], False, [-2,0,0,0,0,0,0])
    def test_spellbound_bass(self):
        self.check([43,38,33,28,21], True, [-2,0,0,0,0])
    def test_against_you_guitar(self):
        self.check([62,57,53,48,43,36], False, [-4,-2,-2,-2,-2,-2])
    def test_against_you_bass(self):
        self.check([41,36,31,24], True, [-4,-2,-2,-2])
    def test_standard_extended_range(self):
        self.check([64,59,55,50,45,40,35], False, [0]*7)
        self.check([43,38,33,28,23], True, [0]*5)

if __name__ == "__main__":
    unittest.main()
