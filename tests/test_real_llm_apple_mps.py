"""Opt-in real local-LLM smoke test for Apple Silicon.

Run explicitly with:

    XAI_RUN_REAL_LLM_TESTS=1 PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_real_llm_apple_mps.py

The default test suite skips this file because loading the 7B base model is
slow and requires local model weights.
"""

from __future__ import annotations

import os
import unittest

from xai_pipeline.core.api import handle_request
from xai_pipeline.planning.local_llm import check_local_llm_readiness


@unittest.skipUnless(os.environ.get("XAI_RUN_REAL_LLM_TESTS") == "1", "set XAI_RUN_REAL_LLM_TESTS=1 to run the real local LLM")
class RealLocalLLMAppleMPSTests(unittest.TestCase):
    def test_real_local_llm_proposes_compiler_accepted_plan_on_apple_mps(self):
        os.environ.update(
            {
                "XAI_ENABLE_LOCAL_LLM": "1",
                "XAI_PLANNING_MODE": "llm_required",
                "XAI_LLM_DEVICE": "mps",
                "XAI_LLM_TORCH_DTYPE": "float16",
                "XAI_LLM_DEVICE_MAP": "none",
                "XAI_LLM_JSON_PREFILL": "1",
                "XAI_LLM_JSON_EARLY_STOP": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "XAI_LLM_MAX_NEW_TOKENS": "32",
                "XAI_LLM_GENERATE_MAX_TIME_SECONDS": "60",
                "XAI_LLM_HARD_TIMEOUT_SECONDS": "0",
            }
        )
        readiness = check_local_llm_readiness()
        self.assertTrue(readiness["ready"], readiness)
        diagnostic = readiness.get("runtime_config", {}).get("mps_diagnostic") or {}
        self.assertTrue(diagnostic.get("mps_available"), diagnostic)

        response = handle_request(
            {"question": "A resistor R = 10 Ω has voltage U = 20 V. Find the current."},
            enable_llm=True,
            planning_mode="llm_required",
            timeout_seconds=-1,
        )
        llm_trace = response["front"]["trace"]["local_llm_solve_plan"]
        compiled_llm_trace = response["solve_plan"]["trace"]["llm_plan_trace"]
        self.assertTrue(llm_trace["used"], llm_trace)
        self.assertTrue(compiled_llm_trace["applied"], compiled_llm_trace)
        self.assertEqual(response["solve_plan"]["plan"]["source"], "local_llm")
        self.assertTrue(response["verifier"]["ok"], response)
        self.assertEqual(response["answer"], "2 A")


if __name__ == "__main__":
    unittest.main()
