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


class EvaluationTests(unittest.TestCase):
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
