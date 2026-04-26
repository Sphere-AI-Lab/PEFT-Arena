"""Math-style scorer used for both OpenR1 and MedThink RL data."""

from __future__ import annotations

import multiprocessing
import os
import queue
import re
import signal
import time
from math import isclose

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

try:
    from latex2sympy2 import latex2sympy
except ImportError:  # pragma: no cover - optional parser
    latex2sympy = None


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) <= 1:
        return new_str
    for substr in substrs[1:]:
        new_str += "\\frac"
        if substr and substr[0] == "{":
            new_str += substr
            continue
        if len(substr) < 2:
            return string
        a = substr[0]
        b = substr[1]
        if b != "{":
            post_substr = substr[2:] if len(substr) > 2 else ""
            new_str += "{" + a + "}{" + b + "}" + post_substr
        else:
            post_substr = substr[2:] if len(substr) > 2 else ""
            new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _fix_sqrt(string):
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    trimmed = re.sub(r"\\text{.*?}$", "", string).strip()
    if trimmed and trimmed != string:
        string = trimmed

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    string = string.replace("\\cdot", "")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string

    parts = string.split("=")
    if len(parts) == 2 and len(parts[0]) <= 2:
        string = parts[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def extract_answer_math(pred_str):
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if not ans:
            pred = ""
        elif ans[0] == "{":
            stack = 1
            extracted = []
            for char in ans[1:]:
                if char == "{":
                    stack += 1
                    extracted.append(char)
                elif char == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    extracted.append(char)
                else:
                    extracted.append(char)
            pred = "".join(extracted)
        else:
            pred = ans.split("$")[0].strip()
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    else:
        matches = re.findall(r"-?\d*\.?\d+", pred_str.replace(",", ""))
        pred = matches[-1] if matches else ""

    pred = pred.split("\n")[0]
    if pred.startswith(":"):
        pred = pred[1:]
    if pred.endswith("."):
        pred = pred[:-1]
    if pred.endswith("/"):
        pred = pred[:-1]
    return strip_string(pred)


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                return None
    return None


def is_digit(num):
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for matrix in matrix_str:
        matrix = matrix.strip("{}")
        pmatrix_list.append(r"\begin{pmatrix}" + matrix.replace(",", "\\") + r"\end{pmatrix}")
    return ", ".join(pmatrix_list)


def numeric_equal(prediction, reference):
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    parsers = [parse_latex, parse_expr]
    if latex2sympy is not None:
        parsers.append(latex2sympy)

    def _parse(value):
        for parser in parsers:
            try:
                return parser(value.replace("\\\\", "\\"))
            except Exception:
                try:
                    return parser(value)
                except Exception:
                    continue
        return value

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if abs(a.lhs - a.rhs).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    try:
        if a.shape == b.shape:
            rounded_a = a.applyfunc(lambda x: round(x, 3))
            rounded_b = b.applyfunc(lambda x: round(x, 3))
            if rounded_a.equals(rounded_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    output_queue.put(symbolic_equal(a, b))


def _func_wrapper(func, args, output_queue):
    try:
        func(*args, output_queue=output_queue)
    except Exception as exc:  # pragma: no cover - defensive
        output_queue.put(("ERROR", str(exc)))


def call_with_timeout(func, *args, timeout=3):
    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_func_wrapper, args=(func, args, output_queue))
    process.daemon = True
    process.start()

    start_time = time.time()
    while process.is_alive() and time.time() - start_time < timeout:
        process.join(0.1)
        try:
            if not output_queue.empty():
                result = output_queue.get(block=False)
                if process.is_alive():
                    process.terminate()
                process.join(0.1)
                if isinstance(result, tuple) and result[0] == "ERROR":
                    return False
                return result
        except queue.Empty:  # pragma: no cover - race dependent
            pass

    if process.is_alive():
        process.terminate()
        process.join(0.1)
        if process.is_alive():  # pragma: no cover - very rare
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:  # pragma: no cover - process exited between checks
                pass
            finally:
                process.join(0.1)
    return False


def math_equal(prediction, reference, include_percentage=True, is_close=True, timeout=False):
    if str(prediction) == str(reference):
        return True

    try:
        if is_digit(prediction) and is_digit(reference):
            prediction_value = parse_digits(prediction)
            reference_value = parse_digits(reference)
            candidates = (
                [reference_value / 100, reference_value, reference_value * 100]
                if include_percentage
                else [reference_value]
            )
            for candidate in candidates:
                try:
                    if is_close and numeric_equal(prediction_value, candidate):
                        return True
                    if not is_close and candidate == prediction_value:
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    pred_str, ref_str = prediction, reference
    if (
        (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("("))
        or (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("["))
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for token in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(token, "")
        pred_str = pred_str.replace(token, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if regex.match(r"(\(|\[).+(\)|\])", prediction) and regex.match(r"(\(|\[).+(\)|\])", reference):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))
        ):
            return True

    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip() for line in prediction[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip() for line in reference[len("\\begin{pmatrix}") : -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        if len(pred_lines) == len(ref_lines):
            matched = True
            for pred_line, ref_line in zip(pred_lines, ref_lines, strict=True):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) != len(ref_parts) or not all(
                    math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close, timeout)
                    for i in range(len(pred_parts))
                ):
                    matched = False
                    break
            if matched:
                return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and "=" not in reference:
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close, timeout):
            return True
    elif reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and "=" not in prediction:
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close, timeout):
            return True

    if timeout:
        return bool(call_with_timeout(symbolic_equal_process, prediction, reference))
    return symbolic_equal(prediction, reference)


def compute_score(solution_str, ground_truth):
    answer = extract_answer_math(solution_str)
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    correct = math_equal(answer, ground_truth, timeout=True)
    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": answer,
    }
