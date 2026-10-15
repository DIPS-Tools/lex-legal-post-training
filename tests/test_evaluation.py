from __future__ import annotations

import unittest

from src.evaluation.common import (
    accuracy,
    balanced_accuracy,
    build_mmlu_prompt,
    legalbench_views,
    normalize_answer,
    summarize_legalbench,
)
from src.evaluation.runner import (
    extract_legalbench_exact_match_answer,
    prepare_legalbench_generations_for_scoring,
)


class EvaluationTests(unittest.TestCase):
    def test_legalbench_exact_match_extraction_uses_first_nonempty_line(self) -> None:
        generation = "  generic  \n\nQ: The next few-shot question?\nA: descriptive"
        self.assertEqual(extract_legalbench_exact_match_answer(generation), "generic")

    def test_legalbench_exact_match_extraction_removes_answer_cue(self) -> None:
        self.assertEqual(
            extract_legalbench_exact_match_answer("\n Answer: Yes\nExplanation"),
            "Yes",
        )
        self.assertEqual(extract_legalbench_exact_match_answer("A: No"), "No")
        self.assertEqual(extract_legalbench_exact_match_answer("\n\t\n"), "")

    def test_legalbench_scoring_extraction_is_task_aware(self) -> None:
        raw = [" generic\n\nQ: continued", "Answer: descriptive\nExplanation"]
        parsed, method = prepare_legalbench_generations_for_scoring(
            "abercrombie",
            raw,
            frozenset({"abercrombie"}),
        )
        self.assertEqual(parsed, ["generic", "descriptive"])
        self.assertEqual(method, "first_nonempty_line")

        preserved, method = prepare_legalbench_generations_for_scoring(
            "citation_prediction_open",
            raw,
            frozenset({"abercrombie"}),
        )
        self.assertEqual(preserved, raw)
        self.assertEqual(method, "raw_generation")

    def test_mmlu_prompt_contains_demonstration_and_unanswered_test(self) -> None:
        demonstration = {
            "question": "Demo?",
            "choices": ["one", "two", "three", "four"],
            "answer": "B",
        }
        test = {
            "question": "Test?",
            "choices": ["alpha", "beta", "gamma", "delta"],
            "answer": "D",
        }
        prompt = build_mmlu_prompt("professional_law", test, [demonstration])
        self.assertIn("Answer: B", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertNotIn("Answer: D", prompt)

    def test_answer_normalization_is_deliberately_conservative(self) -> None:
        self.assertEqual(normalize_answer(" Answer:  YES \nExplanation"), "yes")
        self.assertEqual(normalize_answer("No, because the rule does not apply."), "no, because the rule does not apply.")

    def test_accuracy_and_balanced_accuracy(self) -> None:
        gold = ["Yes", "Yes", "Yes", "No"]
        predictions = ["Yes", "Yes", "Yes", "Yes"]
        self.assertEqual(accuracy(gold, predictions), 0.75)
        self.assertEqual(balanced_accuracy(gold, predictions), 0.5)

    def test_paper_views_apply_name_patterns(self) -> None:
        views = legalbench_views(
            {
                "contract_nli_confidentiality_of_agreement",
                "cuad_governing_law",
                "international_citizenship_questions",
                "abercrombie",
                "function_of_decision_section",
            }
        )
        self.assertEqual(views["dataset_held_out"], {"abercrombie", "function_of_decision_section"})
        self.assertEqual(views["task_held_out"], {"international_citizenship_questions", "abercrombie"})

    def test_legalbench_summary_is_macro_averaged_by_task(self) -> None:
        metrics = {
            "abercrombie": {
                "task_type": "rule",
                "examples": 100,
                "primary_score": 1.0,
            },
            "corporate_lobbying": {
                "task_type": "issue",
                "examples": 1,
                "primary_score": 0.0,
            },
        }
        summary = summarize_legalbench(metrics)
        self.assertEqual(summary["full"]["macro_task_score"], 0.5)
        self.assertEqual(summary["full"]["example_count"], 101)

    def test_empty_held_out_view_is_reported_for_single_task_smoke_test(self) -> None:
        summary = summarize_legalbench(
            {
                "contract_nli_confidentiality_of_agreement": {
                    "task_type": "rule",
                    "examples": 2,
                    "primary_score": 0.5,
                }
            }
        )
        self.assertEqual(summary["dataset_held_out"]["task_count"], 0)
        self.assertIsNone(summary["dataset_held_out"]["macro_task_score"])


if __name__ == "__main__":
    unittest.main()
