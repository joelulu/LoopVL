import unittest

from .core import draft_verify, summarize


class ProtocolTests(unittest.TestCase):
    # Toy autoregressive target: next token is determined by the whole prefix.
    @staticmethod
    def target(prefix):
        return (sum(prefix) * 3 + len(prefix) + 2) % 17

    def verify(self, prefix, drafts):
        return [self.target(prefix + drafts[:i]) for i in range(len(drafts)+1)]

    def test_mixed_proposals_equal_ar(self):
        for width in (1, 2, 4, 16):
            for budget in range(1, 30):
                def propose(prefix):
                    exact = self.target(prefix)
                    return exact if len(prefix) % 3 else (exact+1) % 17
                got, rounds = draft_verify(propose, self.verify, block_size=width,
                                           max_new_tokens=budget, eos_token_id=None)
                expected = []
                for _ in range(budget):
                    expected.append(self.target(expected))
                self.assertEqual(got, expected)
                self.assertEqual(sum(len(r["committed_ids"]) for r in rounds), budget)

    def test_all_accepted_includes_bonus(self):
        got, rounds = draft_verify(self.target, self.verify, block_size=2,
                                   max_new_tokens=7, eos_token_id=None)
        self.assertEqual(len(got), 7)
        self.assertEqual([len(r["committed_ids"]) for r in rounds], [3, 3, 1])
        self.assertEqual(rounds[0]["accepted_draft_tokens"], 2)

    def test_wrong_eos_is_not_committed(self):
        got, rounds = draft_verify(lambda p: 99, self.verify, block_size=4,
                                   max_new_tokens=4, eos_token_id=99)
        self.assertNotIn(99, got)
        self.assertTrue(all(r["accepted_draft_tokens"] == 0 for r in rounds))

    def test_eos_in_accepted_prefix_has_no_bonus(self):
        got, rounds = draft_verify(lambda p: len(p)+1,
                                   lambda p,d: list(range(len(p)+1, len(p)+len(d)+2)),
                                   block_size=4, max_new_tokens=9, eos_token_id=2)
        self.assertEqual(got, [1, 2])
        self.assertEqual(rounds[0]["accepted_draft_tokens"], 2)

    def test_correction_eos(self):
        got, _ = draft_verify(lambda p: 8, lambda p,d: [2]*(len(d)+1),
                              block_size=4, max_new_tokens=9, eos_token_id=2)
        self.assertEqual(got, [2])

    def test_bad_verifier_alignment_fails(self):
        with self.assertRaises(ValueError):
            draft_verify(lambda p: 1, lambda p,d: [1]*len(d),
                         block_size=2, max_new_tokens=5, eos_token_id=None)

    def test_actual_counts_and_weighted_agreement(self):
        rows = [{"id":"a", "h1_id":1, "h2_id":1, "reference_id":1,
                 "h1_confidence":.95, "confidence_threshold":.9},
                {"id":"b", "h1_id":2, "h2_id":1, "reference_id":1,
                 "h1_confidence":.95, "confidence_threshold":.9}]
        samples = [{"id":"a", "category":"ocr", "token_ids":[1]},
                   {"id":"b", "category":"math", "token_ids":[1]}]
        result = summarize(rows, samples)["__all__"]
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["h1_h2_agreement"], .5)
        self.assertEqual(result["high_confidence_disagreement_rate"], .5)


if __name__ == "__main__":
    unittest.main()
