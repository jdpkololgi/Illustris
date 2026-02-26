import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


ENTRYPOINTS = [
    "workflows/abacus_tweb/abacus_process_particles2.py",
    "workflows/jraph/jraph_pipeline.py",
    "workflows/sbi/jraph_sbi_flowjax.py",
    "workflows/sbi/experimental/jraph_sbi_two_stage.py",
    "workflows/gcn_paper/gcn_pipeline.py",
    "jraph_sbi_pipeline.py",
    "jraph_sbi_flowjax_two_stage.py",
]


class TestEntrypointHelp(unittest.TestCase):
    def test_help_entrypoints(self) -> None:
        for rel_path in ENTRYPOINTS:
            script = REPO_ROOT / rel_path
            self.assertTrue(script.exists(), f"Missing entrypoint: {rel_path}")
            proc = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"`--help` failed for {rel_path}:\n{proc.stdout[:600]}",
            )


if __name__ == "__main__":
    unittest.main()
