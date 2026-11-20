"""Native LitePT-S in the actual corrected P12-A sampling, loss and selection loop."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import numpy as np
import torch

from workflows.abacus_tweb import lowz_litept as pt
from workflows.abacus_tweb.lowz_point_loader import PointPhase, ROOT, roles, read
from workflows.abacus_tweb.p10_training_contract import P10PhaseBalancedLoader

INDEX = ROOT.parents[1] / "lowz_points_61662493"
UP = ROOT.parents[1] / "litept_source_436d048"


@dataclass
class PointView:
    field: object
    points: PointPhase

    @property
    def authoritative_parent_id(self):
        return self.field.authoritative_parent_id


class Loader(P10PhaseBalancedLoader):
    def field_adapter(self, phase):
        field = super().field_adapter(phase)
        if not hasattr(self, "_point_views"):
            self._point_views = {}
        if phase not in self._point_views:
            points = PointPhase(self.root, INDEX, phase)
            self._point_views[phase] = SimpleNamespace(extract=lambda *a, **k:
                PointView(field.extract(*a, **k), points))
        return self._point_views[phase]


def pack(view, normalization, device):
    _, xyz, features, loss_ix = pt.observed_inputs(view.points, view.field, normalization)
    d, inverse = pt.voxels(xyz, features)
    # Pack the small patch-local integer coordinates losslessly into float32
    # so the unmodified ordered prefetcher can move just two tensors.
    if d["grid_coord"].max() >= 2**24:
        raise ValueError("voxel index cannot be packed losslessly")
    v = np.concatenate([d["coord"], d["grid_coord"].astype(np.float32), d["feat"]], axis=1)
    return torch.from_numpy(v).to(device), torch.from_numpy(inverse[loss_ix]).long().to(device)


class PackedRegressor(pt.LitePTRegressor):
    def forward(self, packed, points):
        d = dict(coord=packed[:, :3].contiguous(), grid_coord=packed[:, 3:6].int().contiguous(),
                 feat=packed[:, 6:].contiguous(), offset=points.new_tensor([len(packed)]))
        return super().forward(d, points)


@torch.no_grad()
def predict(model, adapter, cores, normalization, device):
    model.eval()
    parents, predictions = [], []
    for core in cores:
        patch = adapter.extract(int(core), pt.grid.HALO_VOXELS, pt.grid.CHANNELS,
                                alignment_voxels=pt.grid.ALIGNMENT_VOXELS)
        pred = model(*pack(patch, normalization, device)).cpu().numpy()
        if pred.shape != (len(patch.authoritative_parent_id), 3) or not np.isfinite(pred).all():
            raise ValueError(f"invalid point prediction for core {core}")
        parents.append(patch.authoritative_parent_id)
        predictions.append(pred)
    return np.concatenate(parents), np.concatenate(predictions), 0


def main():
    from workflows.abacus_tweb import p10_train_arm_a as trainer
    args = trainer.parse_args()
    receipt = read(INDEX / "INDEX_READY.json")
    if ("SLURM_STEP_ID" not in os.environ or not receipt["pass"]
            or tuple(receipt["phases"]) != roles(ROOT) or args.contract_root.resolve() != ROOT
            or args.model != "unet" or not args.run_name.startswith("litept_s_")
            or args.p11_dense_view or args.multitracer_view is not None):
        raise ValueError("require registered LitePT low-z contract and compute step")
    subprocess.run(["git", "-C", str(UP), "diff", "--quiet", "HEAD"], check=True)
    src = trainer.source_contract
    def binding():
        rel = ("lowz_litept_train.py", "lowz_litept.py", "lowz_point_loader.py")
        return {**src(), **{f"workflows/abacus_tweb/{p}": trainer.sha256(Path(__file__).with_name(p)) for p in rel},
                str(INDEX / "INDEX_READY.json"): trainer.sha256(INDEX / "INDEX_READY.json"),
                str(UP / "litept/model.py"): trainer.sha256(UP / "litept/model.py")}
    trainer.source_contract = binding
    trainer.P10PhaseBalancedLoader = Loader
    # Process-local interface: the original grid model_inputs remains available
    # to compute the exact three normalized observational channels.
    trainer.unet_impl = SimpleNamespace(UPatch=lambda **kw: PackedRegressor(UP),
        model_inputs=pack, predict_fold=predict, CHANNELS=pt.grid.CHANNELS,
        HALO_VOXELS=pt.grid.HALO_VOXELS, ALIGNMENT_VOXELS=pt.grid.ALIGNMENT_VOXELS)
    trainer.main()


if __name__ == "__main__":
    main()
