"""Tool-call correctness used by the server audit; format/length rewards are excluded."""

import json
import re
from collections import Counter


def _match_score(list1, list2):
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    count1 = Counter(list1)
    count2 = Counter(list2)
    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection
    return intersection / max_possible if max_possible > 0 else 0.0


def _compute_tool_call_reward(gt_tools, pd_tools):
    if gt_tools == pd_tools:
        return 1.0
    gt_names = [t["name"] for t in gt_tools]
    pd_names = [t["name"] for t in pd_tools]
    score = _match_score(gt_names, pd_names)
    local_max = 1.0
    used_pd = set()
    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        local_max += 1.0 + len(gt_params)
        best_score = 0.0
        best_idx = -1
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd or pd_tool["name"] != gt_name:
                continue
            pd_params = pd_tool["parameters"]
            param_score = _match_score(list(gt_params.keys()), list(pd_params.keys()))
            correctness = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)
            total = param_score + correctness
            if total > best_score:
                best_score = total
                best_idx = i
        if best_idx >= 0:
            used_pd.add(best_idx)
            score += best_score
    return score / local_max if local_max > 0 else 0.0


def compute_score(response: str, solution: str) -> float:
    """Score tool names and arguments; malformed or absent calls score zero."""
    ans = solution

    # Correctness score
    correctness_reward = 0.0
    if "<tool_call>" in ans:
        try:
            gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            gt_tools = [json.loads(t) for t in gt_tool_call.split("\n")]
            if "<tool_call>" in response and "</tool_call>" in response:
                pd_text = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                pd_tools = [json.loads(t) for t in pd_text.split("\n")]
                correctness_reward = _compute_tool_call_reward(gt_tools, pd_tools)
            else:
                correctness_reward = 0.0
        except Exception:
            correctness_reward = 0.0

    return correctness_reward
