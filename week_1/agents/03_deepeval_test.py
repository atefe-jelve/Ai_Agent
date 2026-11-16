#  deepeval test run 03_deepeval_test.py

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def test_multiplication_correctness():
    test_case = LLMTestCase(
        input="What is 12 multiplied by 8?",
        actual_output="96",
        expected_output="96",
    )

    correctness_metric = GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the actual output correctly answers "
            "the user's input, using the expected output as the reference."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )

    assert_test(test_case, [correctness_metric])