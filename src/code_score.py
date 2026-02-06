import json
import traceback

from .utils import check_correctness as apps_check_correctness

def compute_score(completion, test_cases, continuous=False):
    # Extract solution more safely
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        solution = completion.split("```")[1].split("```")[0]
    else:
        solution = completion  # assume raw code

    try:
        if not isinstance(test_cases, dict):
            test_cases = json.loads(test_cases)

        # Validate structure
        if not all(k in test_cases for k in ["inputs", "outputs"]):
            raise ValueError("test_cases must contain 'inputs' and 'outputs'")

        # Try full test
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata_dict = dict(enumerate(metadata))
            metadata = metadata_dict.get(0, {})
            success = all(r is True for r in res)
            if success:
                return 1.0, [metadata]  # consistent return type
            else:
                return 0.0, [metadata]
        except Exception as e:
            print(f"Full test failed: {e}")  # don't silently pass

        # Per-case testing
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        test_cases_list = [{"inputs": [inp], "outputs": [out]} for inp, out in zip(inputs, outputs)]

        if continuous:
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                try:
                    res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=10, debug=False)
                    metadata_dict = dict(enumerate(metadata))
                    metadata = metadata_dict.get(0, {})
                    metadata.update({
                        "test_case": {
                            "input": str(test_case["inputs"][0]),
                            "output": str(test_case["outputs"][0]),
                            "res": str(res)
                        }
                    })
                    metadata_list.append(metadata)
                    res_list.extend(res)
                except Exception as e:
                    print(f"Test case {test_case_id} failed: {e}")
                    metadata_list.append({"error": str(e), "test_case": {}})
                    res_list.append(False)

                if test_case_id >= 9:
                    break

            res_count = len(res_list) or 1
            success = sum(r is True for r in res_list) / res_count
            return success

        else:
            # If not continuous and full test failed, return 0 with empty metadata?
            return 0.0

    except Exception:
        traceback.print_exc()
        return 0.0  # standardized error return