"""
test_utils.py - Unit tests for src/utils.py
=============================================
Tests safe_parse_json, validate_assessment, format_symptoms, validate_age.
Run with: python test_utils.py
"""

import sys
sys.path.insert(0, ".")

from src.utils import safe_parse_json, validate_assessment, format_symptoms, validate_age

passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1


print("=== safe_parse_json tests ===")

# 1. Valid JSON
data, err = safe_parse_json('{"summary": "test", "urgency_level": "LOW"}')
test("Valid JSON parsed", data is not None and err is None)

# 2. JSON with markdown fences
data, err = safe_parse_json('```json\n{"summary": "test"}\n```')
test("Markdown fences stripped", data is not None and err is None)

# 3. JSON with surrounding text
data, err = safe_parse_json('Here is the result: {"summary": "test"} Hope this helps!')
test("Surrounding text handled", data is not None and err is None)

# 4. Invalid JSON
data, err = safe_parse_json("not json at all")
test("Invalid JSON returns error", data is None and err is not None)

# 5. Empty input
data, err = safe_parse_json("")
test("Empty input returns error", data is None and err is not None)

# 6. JSON with code fence (no json label)
data, err = safe_parse_json('```\n{"summary": "test"}\n```')
test("Code fence without json label", data is not None and err is None)


print("\n=== validate_assessment tests ===")

# 7. Complete valid assessment
full = {
    "summary": "test",
    "possible_conditions": [{"name": "Cold", "reason": "symptoms match"}],
    "urgency_level": "LOW",
    "recommended_next_steps": ["rest"],
    "questions_for_doctor": ["q1"],
    "warning_signs": ["w1"],
}
result, warnings = validate_assessment(full.copy())
test("Complete assessment validates with no warnings", len(warnings) == 0)

# 8. Missing keys get defaults
partial = {"summary": "test"}
result, warnings = validate_assessment(partial.copy())
test("Missing keys filled with defaults", "urgency_level" in result and len(warnings) > 0)

# 9. Invalid urgency gets corrected
bad_urgency = full.copy()
bad_urgency["urgency_level"] = "INVALID"
result, warnings = validate_assessment(bad_urgency)
test("Invalid urgency corrected to MEDIUM", result["urgency_level"] == "MEDIUM")

# 10. Case-insensitive urgency
lower = full.copy()
lower["urgency_level"] = "high"
result, warnings = validate_assessment(lower)
test("Lowercase urgency normalised to HIGH", result["urgency_level"] == "HIGH")


print("\n=== format_symptoms tests ===")

# 11. Combined symptoms
s = format_symptoms(["Fever", "Cough"], "headache, nausea")
test("Combined symptoms", "Fever" in s and "headache" in s and "Cough" in s)

# 12. Empty symptoms
s = format_symptoms([], "")
test("Empty symptoms = None reported", s == "None reported")

# 13. Only free-text
s = format_symptoms([], "migraine, dizziness")
test("Free-text only", "migraine" in s and "dizziness" in s)

# 14. Only multiselect
s = format_symptoms(["Fever"], "")
test("Multiselect only", s == "Fever")


print("\n=== validate_age tests ===")

# 15. Valid age
ok, msg = validate_age("25")
test("Age 25 valid", ok and msg == "25")

# 16. Invalid age (letters)
ok, msg = validate_age("abc")
test("Age abc invalid", not ok)

# 17. Out of range
ok, msg = validate_age("200")
test("Age 200 out of range", not ok)

# 18. Empty age
ok, msg = validate_age("")
test("Empty age invalid", not ok)

# 19. Age zero
ok, msg = validate_age("0")
test("Age 0 valid (infant)", ok)

# 20. Negative age
ok, msg = validate_age("-5")
test("Negative age invalid", not ok)


print(f"\n{'=' * 40}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED!")
    sys.exit(1)
