"""Integration tests for the benchmark CLI scripts.

These exercise the commands as documented in the README, including the exit
codes CI depends on: the audit must fail loudly on a corrupt benchmark rather
than reporting a clean bill of health.
"""

from __future__ import annotations

import json

import pytest

from scripts import audit_benchmark, generate_benchmark as generate_script


class TestGenerateBenchmarkScript:
    def test_generates_and_writes_a_benchmark(self, tmp_path, capsys):
        exit_code = generate_script.main(["--cases", "6", "--seed", "42", "--out", str(tmp_path)])
        assert exit_code == 0

        output = capsys.readouterr().out
        assert "Wrote 6 synthetic cases" in output
        assert "Integrity checks: PASSED" in output

        files = list(tmp_path.rglob("cases_seed42_n6.json"))
        assert len(files) == 1
        assert len(json.loads(files[0].read_text(encoding="utf-8"))) == 6

    def test_writes_metadata_alongside_cases(self, tmp_path):
        generate_script.main(["--cases", "4", "--seed", "1", "--out", str(tmp_path)])
        metadata_files = list(tmp_path.rglob("metadata_seed1_n4.json"))
        assert len(metadata_files) == 1
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        assert metadata["case_count"] == 4
        assert metadata["contains_real_patient_data"] is False

    def test_refuses_to_overwrite_without_force(self, tmp_path, capsys):
        args = ["--cases", "3", "--seed", "2", "--out", str(tmp_path)]
        assert generate_script.main(args) == 0
        assert generate_script.main(args) == 1
        assert "already exists" in capsys.readouterr().err

    def test_force_overwrites(self, tmp_path):
        args = ["--cases", "3", "--seed", "2", "--out", str(tmp_path)]
        generate_script.main(args)
        assert generate_script.main([*args, "--force"]) == 0

    def test_same_seed_produces_identical_files(self, tmp_path):
        first_dir = tmp_path / "a"
        second_dir = tmp_path / "b"
        generate_script.main(["--cases", "5", "--seed", "42", "--out", str(first_dir)])
        generate_script.main(["--cases", "5", "--seed", "42", "--out", str(second_dir)])

        first = next(first_dir.rglob("cases_*.json")).read_text(encoding="utf-8")
        second = next(second_dir.rglob("cases_*.json")).read_text(encoding="utf-8")
        assert first == second

    def test_difficulty_flag_is_honoured(self, tmp_path):
        generate_script.main(
            ["--cases", "4", "--seed", "3", "--difficulty", "ADVERSARIAL", "--out", str(tmp_path)]
        )
        cases = json.loads(next(tmp_path.rglob("cases_*.json")).read_text(encoding="utf-8"))
        assert all(case["difficulty"] == "ADVERSARIAL" for case in cases)

    def test_categories_flag_restricts_the_pool(self, tmp_path):
        generate_script.main(
            [
                "--cases", "4", "--seed", "3", "--difficulty", "MODERATE",
                "--categories", "negation", "abnormal_labs", "--out", str(tmp_path),
            ]
        )
        cases = json.loads(next(tmp_path.rglob("cases_*.json")).read_text(encoding="utf-8"))
        for case in cases:
            assert set(case["categories"]) <= {"negation", "abnormal_labs"}

    def test_curated_flag_regenerates_the_seed_set(self, tmp_path, capsys):
        assert generate_script.main(["--curated", "--out", str(tmp_path)]) == 0
        assert "10 curated synthetic cases" in capsys.readouterr().out
        assert (tmp_path / "benchmark_v1.json").exists()

    def test_parses_documented_arguments(self):
        args = generate_script.parse_args(["--cases", "100", "--seed", "42"])
        assert args.cases == 100
        assert args.seed == 42

    def test_refuses_to_write_a_benchmark_that_fails_integrity(self, tmp_path, capsys, monkeypatch):
        """The safety gate: a generation run that produces a corrupt benchmark
        must fail loudly rather than publishing bad ground truth."""
        from app.benchmark.integrity import IntegrityIssue

        monkeypatch.setattr(
            generate_script, "validate_benchmark",
            lambda cases, metadata=None: [
                IntegrityIssue("fake_check", c.case_id, "injected failure") for c in cases
            ],
        )
        exit_code = generate_script.main(["--cases", "3", "--seed", "1", "--out", str(tmp_path)])
        assert exit_code == 1

        captured = capsys.readouterr().err
        assert "Refusing to write" in captured
        assert not list(tmp_path.rglob("cases_*.json")), "no file should have been written"

    def test_skip_validation_writes_despite_failures(self, tmp_path, monkeypatch):
        from app.benchmark.integrity import IntegrityIssue

        monkeypatch.setattr(
            generate_script, "validate_benchmark",
            lambda cases, metadata=None: [IntegrityIssue("fake_check", "-", "injected")],
        )
        exit_code = generate_script.main(
            ["--cases", "3", "--seed", "1", "--out", str(tmp_path), "--skip-validation"]
        )
        assert exit_code == 0
        assert list(tmp_path.rglob("cases_*.json"))

    def test_truncates_a_long_issue_list(self, tmp_path, capsys, monkeypatch):
        from app.benchmark.integrity import IntegrityIssue

        monkeypatch.setattr(
            generate_script, "validate_benchmark",
            lambda cases, metadata=None: [
                IntegrityIssue("fake_check", str(i), "injected") for i in range(30)
            ],
        )
        generate_script.main(["--cases", "2", "--seed", "1", "--out", str(tmp_path)])
        assert "and 10 more" in capsys.readouterr().err


class TestAuditBenchmarkScript:
    @pytest.fixture
    def benchmark_dir(self, tmp_path):
        generate_script.main(["--cases", "8", "--seed", "42", "--out", str(tmp_path)])
        return tmp_path

    def test_reports_a_clean_benchmark_and_exits_zero(self, benchmark_dir, capsys):
        exit_code = audit_benchmark.main(["--base-dir", str(benchmark_dir)])
        output = capsys.readouterr().out
        assert exit_code == 0
        assert "All integrity checks PASSED" in output
        assert "Cases by difficulty" in output
        assert "Annotated fact counts" in output

    def test_report_includes_the_required_statistics(self, benchmark_dir, capsys):
        audit_benchmark.main(["--base-dir", str(benchmark_dir)])
        output = capsys.readouterr().out
        for expected in (
            "Total cases", "Cases by difficulty", "Cases by failure-mode category",
            "note length (characters)", "annotated facts per case",
            "critical_facts", "medication_facts", "temporal_facts", "negation_facts",
        ):
            assert expected in output, expected

    def test_exits_non_zero_on_a_corrupt_benchmark(self, benchmark_dir, capsys):
        cases_path = next(benchmark_dir.rglob("cases_*.json"))
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        cases[0]["critical_facts"][0]["source_ref"] = "DOES-NOT-EXIST"
        cases_path.write_text(json.dumps(cases), encoding="utf-8")

        exit_code = audit_benchmark.main(["--base-dir", str(benchmark_dir)])
        assert exit_code == 1
        assert "DOES-NOT-EXIST" in capsys.readouterr().out

    def test_exits_non_zero_when_pii_is_present(self, benchmark_dir, capsys):
        cases_path = next(benchmark_dir.rglob("cases_*.json"))
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        cases[0]["clinical_note"] += "\nReach the patient at patient@example.com"
        cases_path.write_text(json.dumps(cases), encoding="utf-8")

        assert audit_benchmark.main(["--base-dir", str(benchmark_dir)]) == 1
        assert "email" in capsys.readouterr().out

    def test_detects_metadata_drift(self, benchmark_dir, capsys):
        metadata_path = next(benchmark_dir.rglob("metadata_*.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["counts"]["critical_facts"] += 99
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        assert audit_benchmark.main(["--base-dir", str(benchmark_dir)]) == 1
        assert "does not match" in capsys.readouterr().out

    def test_reports_a_missing_benchmark_clearly(self, tmp_path, capsys):
        assert audit_benchmark.main(["--base-dir", str(tmp_path)]) == 1
        assert "No generated benchmark found" in capsys.readouterr().err

    def test_accepts_an_explicit_cases_file(self, benchmark_dir, capsys):
        cases_path = next(benchmark_dir.rglob("cases_*.json"))
        assert audit_benchmark.main(["--cases-file", str(cases_path)]) == 0
        assert "All integrity checks PASSED" in capsys.readouterr().out

    def test_rejects_an_empty_cases_file(self, tmp_path, capsys):
        empty = tmp_path / "empty.json"
        empty.write_text("[]", encoding="utf-8")
        assert audit_benchmark.main(["--cases-file", str(empty)]) == 1
        assert "contains no cases" in capsys.readouterr().err

    def test_reports_a_missing_explicit_file(self, tmp_path, capsys):
        assert audit_benchmark.main(["--cases-file", str(tmp_path / "nope.json")]) == 1
        assert "not found" in capsys.readouterr().err

    def test_reports_the_master_seed_not_a_per_case_seed(self, benchmark_dir, capsys):
        """The reported seed must be one the operator can pass back to --seed."""
        audit_benchmark.main(["--base-dir", str(benchmark_dir)])
        output = capsys.readouterr().out
        assert "Seed              : 42" in output
        assert "42:0" not in output

    def test_master_seed_extraction(self):
        from app.benchmark import generate_benchmark as generate_cases

        cases = generate_cases(case_count=2, seed=77)
        assert audit_benchmark.master_seed(cases) == "77"
        assert audit_benchmark.master_seed([]) == "unknown"

    def test_finds_the_newest_benchmark_in_a_directory(self, benchmark_dir):
        found = audit_benchmark.find_latest_benchmark(benchmark_dir)
        assert found is not None
        cases_path, metadata_path = found
        assert cases_path.exists()
        assert metadata_path is not None and metadata_path.exists()
