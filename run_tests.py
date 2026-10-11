"""
run_tests.py — Automated test runner for PulseVision test suite.
Discovers and executes all unit tests across tests/ directory.
"""

import sys
import time
from pathlib import Path

# Add PulseVision root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import tests.test_heart_rate as t_hr
import tests.test_signal_processing as t_dsp
import tests.test_session_manager as t_sm
import tests.test_database as t_db
import tests.test_reports as t_rep


def run_all_tests():
    test_modules = [
        ("Heart Rate & Spectral Analysis", [
            ("test_bpm_estimation_known_frequency", t_hr.test_bpm_estimation_known_frequency),
            ("test_bpm_estimation_different_rates", t_hr.test_bpm_estimation_different_rates),
            ("test_insufficient_samples", t_hr.test_insufficient_samples),
            ("test_nan_and_inf_handling", t_hr.test_nan_and_inf_handling),
        ]),
        ("Digital Signal Processing (DSP)", [
            ("test_detrend_signal", t_dsp.test_detrend_signal),
            ("test_butter_bandpass_filter", t_dsp.test_butter_bandpass_filter),
            ("test_rppg_algorithms_output_shape", t_dsp.test_rppg_algorithms_output_shape),
        ]),
        ("Session Manager & Statistics", [
            ("test_session_state_transitions", t_sm.test_session_state_transitions),
            ("test_session_demographics_and_zones", t_sm.test_session_demographics_and_zones),
            ("test_session_statistics_calculation", t_sm.test_session_statistics_calculation),
            ("test_empty_session_statistics", t_sm.test_empty_session_statistics),
        ]),
        ("Database Persistence (SQLite)", [
            ("test_database_crud", t_db.test_database_crud),
        ]),
        ("Reports & PDF Export (ReportLab/CSV)", [
            ("test_csv_generation", t_rep.test_csv_generation),
            ("test_pdf_generation", t_rep.test_pdf_generation),
            ("test_empty_session_pdf_generation", t_rep.test_empty_session_pdf_generation),
        ]),
    ]

    print("=" * 70)
    print("PULSEVISION AUTOMATED UNIT & INTEGRATION TEST SUITE")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0
    t0 = time.time()

    for category, funcs in test_modules:
        print(f"\n[{category}]")
        for test_name, test_func in funcs:
            total_tests += 1
            try:
                test_func()
                print(f"  PASS: {test_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  FAIL: {test_name} -> {e}")
                import traceback
                traceback.print_exc()

    dt = time.time() - t0
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} PASSED in {dt:.2f}s")
    if passed_tests == total_tests:
        print("ALL TESTS PASSED WITH 100% SUCCESS!")
    else:
        print(f"WARNING: {total_tests - passed_tests} test(s) failed.")
    print("=" * 70)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
