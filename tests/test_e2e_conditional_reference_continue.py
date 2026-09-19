import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from workflows.sbi.e2e_conditional_reference import train_step
from workflows.sbi.e2e_direct_vdm import ConditionalVDM
from workflows.sbi.e2e_conditional_reference_continue import (
    assert_state_equal,digest,fit_items,restore_training_state,validate_extension,validate_state,verify_snapshot,
)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        repo=Path(__file__).resolve().parents[1]
        cls.base=json.loads((repo/"configs/e2e_conditional_reference_v1.json").read_text())
        cls.ext=json.loads((repo/"configs/e2e_conditional_reference_continuation_v1.json").read_text())

    def test_frozen_parent_config(self):
        path=Path(__file__).resolve().parents[1]/"configs/e2e_conditional_reference_v1.json"
        self.assertEqual(digest(path),self.ext["parent_config_sha256"])
        validate_extension(self.ext,self.base)

    def test_disjoint_worker_assignment(self):
        items=fit_items(self.base)
        self.assertEqual(len(items),12)
        partitions=[items[r::4] for r in range(4)]
        self.assertEqual([len(x) for x in partitions],[3,3,3,3])
        self.assertEqual(len({x["name"] for p in partitions for x in p}),12)

    def test_reject_bad_ladder(self):
        for changes in ({"start_update":0},{"updates":4096},{"workers":5},
                        {"deadline_seconds":5401},{"checkpoints":[4096,8192,8192,65536]},
                        {"nfe_by_objective":{"vdm":[256,512],"cfm":[128,256]}}):
            with self.assertRaises(ValueError):
                validate_extension(self.ext|changes,self.base)

    def test_parent_state_fail_closed(self):
        saved=dict(sources={"a":"b"},update=4096,model={},optimizer={},generator=[],cpu_rng=[],cuda_rng=[])
        validate_state(saved,{"a":"b"},4096)
        with self.assertRaises(ValueError):
            validate_state(saved,{"a":"changed"},4096)
        with self.assertRaises(ValueError):
            validate_state(saved,{"a":"b"},8192)

    def test_serialized_optimizer_rng_continuation(self):
        for objective in ("vdm","cfm"):
            torch.manual_seed(102)
            model=ConditionalVDM(3,8,1,False)
            optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
            rng=torch.Generator().manual_seed(312)
            x=torch.randn(2,1,4,4,4,generator=rng)
            c=torch.randn(2,3,4,4,4,generator=rng)
            train_step(model,optimizer,x,c,objective,rng)
            saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                                     generator=rng.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=[]))
            expected=train_step(model,optimizer,x,c,objective,rng)
            target=copy.deepcopy(model.state_dict())
            with tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)/"checkpoint.pt"
                torch.save(saved,path)
                loaded=torch.load(path,weights_only=False)
                restore_training_state(model,optimizer,rng,loaded)
                actual=train_step(model,optimizer,x,c,objective,rng)
                self.assertEqual(expected,actual)
                for key in target:
                    torch.testing.assert_close(target[key],model.state_dict()[key],rtol=0,atol=0)
                # The replay control must not mutate its loaded checkpoint.
                for key,value in saved["optimizer"]["state"].items():
                    for name,tensor in value.items():
                        torch.testing.assert_close(tensor,loaded["optimizer"]["state"][key][name],rtol=0,atol=0)

    def test_snapshot_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"source").mkdir()
            path=root/"source/test.py"
            path.write_text("original\n")
            manifest=dict(sources={"test.py":digest(path)},items=[])
            verify_snapshot(root,manifest)
            path.write_text("changed\n")
            with self.assertRaises(ValueError):
                verify_snapshot(root,manifest)

    def test_exact_restored_state_check(self):
        state={"tensor":torch.arange(4),"groups":[{"lr":.0003}]}
        assert_state_equal(copy.deepcopy(state),state)
        altered=copy.deepcopy(state); altered["tensor"][0]=1
        with self.assertRaises(AssertionError):
            assert_state_equal(altered,state)


if __name__=="__main__":
    unittest.main()
