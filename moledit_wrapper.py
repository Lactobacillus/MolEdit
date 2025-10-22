#!/usr/bin/env python3
"""
MolEdit CLI v2: masked coordinate repaint + element mutation + atom addition

핵심 아이디어
- 레포의 구조 확산(inference) 파이프라인을 그대로 사용하되, constituent(원소종/수소수/혼성화) 벡터를
  마스크에 맞춰 샘플마다 변형한 뒤 preprocess_data -> inference 로 전달.
- 좌표는 repaint_dict 로 고정/비고정을 제어.
- 원자 추가는 constituent 벡터와 초기 좌표를 확장해 모델이 구조를 정련하도록 함.

실험적 옵션으로 constituents transformer를 사용할 수 있는 훅을 두었지만,
기본은 간단하고 예측가능한 규칙기반 변형(허용 원소 집합에서 치환/추가)으로 동작.

사용 예
python modeledit_cli_v2.py \
  --input-mol input.mol \
  --mask-mol mask.mol \
  --out-dir runs/run2 \
  --num-samples 64 \
  --edit-mode both \
  --allowed-elements C,N,O,F,Cl,Br,I \
  --element-change-prob 0.5 \
  --add-atoms 2 \
  --add-radius-scale 0.6 \
  --method DPM_3 --steps 20 --output-format xyz

주의
- --mask-mol 에서 Og 로 표시한 원자는 기본적으로 '바뀌어도 되는 원자'로 해석(editable).
  반대 의미라면 --mask-means fixed 지정.
- constituents transformer 사용은 --use-transformer 로 활성화. 체크포인트가 있어야 함.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "False")
import math
import json
import argparse
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from rdkit import Chem
from rdkit.Chem import AllChem

# MolEdit repo modules
from cybertron.common.config_load import load_config
from train.train import MolEditScoreNet
from cybertron.model.molct_plus import MolCT_Plus
from cybertron.readout import GFNReadout
from train.utils import set_dropout_rate_config
from jax.sharding import PositionalSharding

from inference.inference import DPM_3_inference, Langevin_inference, DPM_pp_2S_inference
from inference.utils import preprocess_data
from graph_assembler.graph_assembler import assemble_mol_graph

# Optional: constituents transformer
try:
    from transformer.model import Transformer
    from config.transformer_config import transformer_config
    HAVE_TRANSFORMER = True
except Exception:
    HAVE_TRANSFORMER = False

# -------------------------
# Utilities
# -------------------------

PT = Chem.GetPeriodicTable()
COMMON_ELEMS = ['H','C','N','O','F','P','S','Cl','Br','I','Si']

def elem_to_Z(sym: str) -> int:
    try:
        return int(PT.GetAtomicNumber(sym))
    except Exception:
        raise ValueError(f"Unknown element symbol: {sym}")

def Z_to_elem(Z: int) -> str:
    return Chem.GetPeriodicTable().GetElementSymbol(int(Z))

def read_mol_with_coords(path: Path):
    ext = path.suffix.lower()
    if ext in [".mol", ".sdf"]:
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
    elif ext == ".xyz":
        with open(path, "r") as f:
            lines = [l.strip() for l in f.readlines()]
        n = int(lines[0])
        elems = []
        coords = []
        for i in range(n):
            parts = lines[2+i].split()
            elems.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        mol = Chem.RWMol()
        for e in elems:
            a = Chem.Atom(e)
            mol.AddAtom(a)
        mol = mol.GetMol()
        conf = Chem.Conformer(len(elems))
        for i, xyz in enumerate(coords):
            conf.SetAtomPosition(i, xyz)
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
    else:
        raise ValueError(f"Unsupported input extension: {ext}")
    if mol is None:
        raise ValueError(f"Failed to read molecule from {path}")
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    return mol, coords

def parse_mask_from_mol(mask_mol: Chem.Mol, mask_means: str, base_natoms: int):
    assert mask_means in ("editable", "fixed")
    editable = np.zeros(base_natoms, dtype=bool)

    # RDKit AtomIterator는 슬라이싱이 안 되므로 list로 변환
    atoms = list(mask_mol.GetAtoms()) if mask_mol is not None else []
    limit = min(base_natoms, len(atoms))
    for i in range(limit):
        a = atoms[i]
        if a.GetSymbol() == "Og":
            editable[i] = True

    # mask 의미 해석
    if mask_means == "editable":
        fixed_mask = ~editable
    else:
        fixed_mask = editable
    return fixed_mask

def rdkit_to_constituents_and_structure(mol: Chem.Mol):
    mol = Chem.RemoveAllHs(mol)
    atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.uint8)
    hydrogen_numbers = np.array([a.GetTotalNumHs() for a in mol.GetAtoms()], dtype=np.uint8)
    hybridizations = np.array([int(a.GetHybridization()) for a in mol.GetAtoms()], dtype=np.uint8)
    bond_ids = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    bond_types = [int(b.GetBondType()) for b in mol.GetBonds()]
    topology = {i: {} for i in range(len(atomic_numbers))}
    for (i,j), t in zip(bond_ids, bond_types):
        topology[i][j] = topology[j][i] = t
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    return dict(atomic_numbers=atomic_numbers,
                hydrogen_numbers=hydrogen_numbers,
                hybridizations=hybridizations,
                bonds=topology), coords

def write_xyz(path: Path, atomic_numbers, coords):
    with open(path, "w") as f:
        f.write(f"{len(atomic_numbers)}\n")
        f.write("generated by modeledit_cli_v2\n")
        for Z, (x,y,z) in zip(atomic_numbers, coords):
            f.write(f"{Z_to_elem(int(Z))} {x:.6f} {y:.6f} {z:.6f}\n")

def try_write_mol(path: Path, atomic_numbers, hydrogen_numbers, coords):
    success, xmol, smiles = assemble_mol_graph(np.array(atomic_numbers, dtype=np.uint8),
                                               np.array(hydrogen_numbers, dtype=np.uint8),
                                               np.array(coords, dtype=np.float32))
    if not success:
        return False, ""
    atoms = xmol.atoms[::1]
    h_idx = sorted([idx for idx, a in enumerate(atoms) if 'H' in a])[::-1]
    for idx in h_idx:
        xmol.delete_atom(idx)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, ""
    mol = Chem.AddHs(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    if len(coords) < mol.GetNumAtoms():
        return False, smiles
    for i in range(min(mol.GetNumAtoms(), len(coords))):
        conf.SetAtomPosition(i, coords[i])
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()
    return True, smiles

def build_repaint_dict(coords, fixed_mask, max_atoms):
    structure = np.zeros((max_atoms, 3), dtype=np.float32)
    mask = np.zeros((max_atoms,), dtype=bool)
    n = min(len(coords), max_atoms)
    structure[:n] = coords[:n]
    mask[:n] = fixed_mask[:n]
    return {"structure": structure, "mask": mask}

# -------------------------
# Constituents mutation utilities
# -------------------------

def mutate_constituents(atomic_numbers: np.ndarray,
                        hydrogen_numbers: np.ndarray,
                        hybridizations: np.ndarray,
                        fixed_mask: np.ndarray,
                        allowed_Z: List[int],
                        change_prob: float,
                        add_atoms: int,
                        add_radius: float,
                        coords: np.ndarray,
                        rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    규칙 기반 변형
    - fixed_mask=True 인 원자는 그대로 유지
    - fixed_mask=False 인 원자 중 change_prob 로 원소 치환
    - add_atoms 만큼 새로운 원자 추가, 초기 좌표는 중심 근방에 무작위 배치
    - 수소 수와 혼성화는 보수적으로 유지/가정
    """
    an = atomic_numbers.copy()
    hn = hydrogen_numbers.copy()
    hy = hybridizations.copy()
    xyz = coords.copy()

    editable_idx = np.where(~fixed_mask)[0]
    for i in editable_idx:
        if rng.rand() < change_prob:
            newZ = rng.choice(allowed_Z)
            an[i] = newZ
            # 간단히 수소수를 0으로 초기화, 혼성화는 원래 값 유지
            hn[i] = 0

    n_add = int(max(0, add_atoms))
    if n_add > 0:
        center = xyz.mean(axis=0, keepdims=True)
        rg = float(np.sqrt(np.mean(np.sum((xyz-center)**2, axis=1))))
        scale = add_radius if add_radius is not None else 0.5
        for _ in range(n_add):
            newZ = int(rng.choice(allowed_Z))
            an = np.append(an, newZ).astype(np.uint8)
            hn = np.append(hn, 0).astype(np.uint8)
            hy = np.append(hy, 3).astype(np.uint8)  # rdkit enums: approximate sp2
            disp = rng.normal(size=(3,)).astype(np.float32)
            disp = disp / (np.linalg.norm(disp)+1e-8) * (rg * scale + 0.5*rng.rand())
            newpos = center[0] + disp
            xyz = np.vstack([xyz, newpos]).astype(np.float32)

    return an, hn, hy, xyz

# -------------------------
# Model init
# -------------------------

def init_structure_model(params_dir: Path, ndevices: int, dropout: float = 0.0):
    encoder_config = load_config("config/molct_plus.yaml")
    gfn_config = load_config("config/gfn.yaml")
    gfn_config.settings.n_interactions = 4

    modules = {
        "encoder": {"module": MolCT_Plus, "args": {"config": encoder_config}},
        "gfn": {"module": GFNReadout, "args": {"config": gfn_config}},
    }

    ckpts = [
        params_dir / "structure_model" / "moledit_params_track1.pkl",
        params_dir / "structure_model" / "moledit_params_track2.pkl",
        params_dir / "structure_model" / "moledit_params_track3.pkl",
    ]
    noise_thresholds = [0.35, 1.95]

    import pickle as pkl
    params = []
    for p in ckpts:
        with open(p, "rb") as f:
            params.append(jax.tree_util.tree_map(lambda x: jnp.array(x), pkl.load(f)))

    if ndevices > 1:
        global_sharding = PositionalSharding(jax.devices()).reshape(ndevices, 1)
        params = jax.device_put(params, global_sharding.replicate())

    for k, v in modules.items():
        modules[k]['args']['config'] = set_dropout_rate_config(modules[k]['args']['config'], dropout)
        modules[k]["module"] = v["module"](**v["args"])
        modules[k]["callable_fn"] = []
        for param in params:
            partial_params = {"params": param["params"]['score_net'].pop(k)}
            modules[k]["callable_fn"].append(partial(modules[k]["module"].apply, partial_params))

    moledit_scorenets = [MolEditScoreNet(
        encoder=modules['encoder']['callable_fn'][k],
        gfn=modules['gfn']['callable_fn'][k],
    ) for k in range(len(ckpts))]

    def score_forward_fn(atom_feat, bond_feat, x, atom_mask, sigma, rg, gamma=1.0):
        cond_list = [sigma < noise_thresholds[0],] + \
                    [jnp.logical_and(sigma >= noise_thresholds[i], sigma < noise_thresholds[i+1]) for i in range(0, len(noise_thresholds) - 1)] + \
                    [sigma >= noise_thresholds[-1],]
        value_list = [net.apply({}, atom_feat, bond_feat, x, atom_mask, sigma, rg)[-1] for net in moledit_scorenets]
        value_unc_list = [net.apply({}, atom_feat, jnp.zeros_like(bond_feat), x, atom_mask, sigma, rg)[-1] for net in moledit_scorenets]
        value = gamma * jnp.array(value_list, jnp.float32) + (1.0 - gamma) * jnp.array(value_unc_list, jnp.float32)
        return jnp.sum(jnp.array(cond_list, dtype=jnp.float32)[..., None, None] * value, axis=0)

    score_forward_fn_jvj = jax.jit(jax.vmap(jax.jit(score_forward_fn)))
    return score_forward_fn_jvj

def pick_inference(method: str, score_fn, n_steps: int, shard_inputs: bool):
    if method == "DPM_3":
        return partial(DPM_3_inference, score_fn=score_fn, n_steps=n_steps, shard_inputs=shard_inputs)
    if method == "DPM_pp_2S":
        return partial(DPM_pp_2S_inference, score_fn=score_fn, n_steps=n_steps, shard_inputs=shard_inputs)
    if method == "Langevin":
        return partial(Langevin_inference, score_fn=score_fn, n_steps=n_steps, shard_inputs=shard_inputs)
    raise ValueError(f"Unknown method {method}")

def to_heavy_mask(mol: Chem.Mol, fixed_mask_full: np.ndarray) -> np.ndarray:
    heavy_indices = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() != 1]
    # 마스크가 더 짧거나 길 수 있으니 길이 클램핑
    heavy_indices = [i for i in heavy_indices if i < len(fixed_mask_full)]
    return fixed_mask_full[heavy_indices]

def write_with_hydrogens(out_path_mol, out_path_xyz, smiles, heavy_coords):
    """
    smiles: assemble_mol_graph 등으로 얻은 heavy-only SMILES
    heavy_coords: (N_heavy, 3) numpy array
    """
    # 1) heavy-only Mol 만들기
    mol_noH = Chem.MolFromSmiles(smiles)
    if mol_noH is None:
        raise ValueError("SMILES -> Mol 변환 실패")

    mol_noH = Chem.AddHs(mol_noH, addCoords=False)  # 좌표 넣기 전에 H를 붙이지 않음
    # 위 줄은 사실 필요 없음. 입력은 heavy-only로 두고 좌표부터 넣을 것
    mol_noH = Chem.RemoveHs(mol_noH)

    # 2) heavy 좌표 컨포머 추가
    conf = Chem.Conformer(mol_noH.GetNumAtoms())
    if mol_noH.GetNumAtoms() != heavy_coords.shape[0]:
        raise ValueError("heavy atom 개수와 좌표 개수가 다릅니다")
    for i in range(mol_noH.GetNumAtoms()):
        x, y, z = map(float, heavy_coords[i])
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    mol_noH.RemoveAllConformers()
    mol_noH.AddConformer(conf, assignId=True)

    # 3) H 추가하면서 좌표 자동 배치
    molH = Chem.AddHs(mol_noH, addCoords=True)

    # 4) mol/SDF 저장
    w = Chem.SDWriter(str(out_path_mol))
    w.write(molH)
    w.close()

    # 5) xyz 저장
    confH = molH.GetConformer()
    with open(out_path_xyz, "w") as f:
        f.write(f"{molH.GetNumAtoms()}\n")
        f.write("generated by MolEdit wrapper with RDKit AddHs(addCoords=True)\n")
        for idx, atom in enumerate(molH.GetAtoms()):
            pos = confH.GetAtomPosition(idx)
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="MolEdit masked editor with element mutation & atom addition")
    ap.add_argument("--input-mol", required=True, type=Path)
    ap.add_argument("--mask-mol", required=True, type=Path, help="Og 로 표시")
    ap.add_argument("--mask-means", choices=["editable", "fixed"], default="editable")
    ap.add_argument("--params-dir", type=Path, default=Path("params/ZINC_3m"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--natoms", type=int, default=64)
    ap.add_argument("--method", choices=["DPM_3", "DPM_pp_2S", "Langevin"], default="DPM_3")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--output-format", choices=["xyz", "mol", "both"], default="xyz")
    ap.add_argument("--ndevices", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=1.0)

    # element mutation / addition
    ap.add_argument("--edit-mode", choices=["coords","elements","both"], default="coords",
                    help="coords: 좌표만, elements: 원소/추가만, both: 둘 다")
    ap.add_argument("--allowed-elements", type=str, default=",".join(COMMON_ELEMS),
                    help="치환/추가에 사용할 원소 리스트. 예: C,N,O,F,Cl,Br,I")
    ap.add_argument("--element-change-prob", type=float, default=0.0,
                    help="editable 원자에 대해 원소를 바꿀 확률")
    ap.add_argument("--add-atoms", type=int, default=0, help="샘플마다 추가할 새로운 원자 수")
    ap.add_argument("--add-radius-scale", type=float, default=0.5,
                    help="초기 좌표를 배치할 반지름 스케일(입력 rg 배수)")
    # experimental transformer usage
    ap.add_argument("--use-transformer", action="store_true",
                    help="constituents transformer로 조성 재샘플링 시도 (실험적)")

    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_mol, base_coords = read_mol_with_coords(args.input_mol)
    mask_mol, _ = read_mol_with_coords(args.mask_mol)
    # fixed_mask = parse_mask_from_mol(mask_mol, args.mask_means, base_mol.GetNumAtoms())
    fixed_mask_full = parse_mask_from_mol(mask_mol, args.mask_means, base_mol.GetNumAtoms())
    fixed_mask_heavy = to_heavy_mask(base_mol, fixed_mask_full)
    consts, coords = rdkit_to_constituents_and_structure(base_mol)

    rg = float(np.sqrt(np.mean(np.sum((coords - coords.mean(0))**2, axis=1))))
    allowed_Z = [elem_to_Z(s) for s in args.allowed_elements.split(",")]

    # model init
    ndevices = len(jax.devices()) if args.ndevices is None else int(args.ndevices)
    score_fn = init_structure_model(args.params_dir, ndevices=ndevices, dropout=0.0)
    inference = pick_inference(args.method, score_fn, n_steps=args.steps, shard_inputs=(ndevices>1))

    # optional: transformer load (not strictly required)
    transformer = None
    tr_params = None
    vocab = None
    if args.use_transformer:
        if not HAVE_TRANSFORMER:
            print("Warning: transformer modules not importable. Falling back to rule-based mutation.", flush=True)
        else:
            import pickle as pkl
            try:
                with open(args.params_dir / "constituents_model" / "constituents_vocab.pkl", "rb") as f:
                    vocab = pkl.load(f)
                with open(args.params_dir / "constituents_model" / "moledit_params.pkl", "rb") as f:
                    tr_params = jax.tree_util.tree_map(lambda x: jnp.array(x), pkl.load(f))
                transformer = Transformer(transformer_config)
            except Exception as e:
                print(f"Warning: failed to load transformer checkpoints: {e}. Using rule-based mutation.", flush=True)
                transformer = None

    samples_written = 0
    for i in range(args.num_samples):
        an, hn, hy, xyz = consts["atomic_numbers"], consts["hydrogen_numbers"], consts["hybridizations"], coords

        if args.edit_mode in ("elements", "both"):
            if transformer is not None:
                # placeholder: keep current constituents; advanced constrained sampling could be added here.
                pass
            an, hn, hy, xyz = mutate_constituents(
                an, hn, hy, fixed_mask_heavy, allowed_Z,
                change_prob=args.element_change_prob,
                add_atoms=args.add_atoms,
                add_radius=args.add_radius_scale,
                coords=xyz,
                rng=rng,
            )

        # pad to natoms for model input
        raw_info = {
            "atomic_numbers": an,
            "hydrogen_numbers": hn,
            "hybridizations": hy,
            "radius_of_gyrations": [rg, rg],
            "bonds": consts["bonds"],  # 새 원자에 대해서는 비결합으로 두고, 어셈블러가 사후 재구성
        }
        inp = preprocess_data(raw_info, args.natoms)
        # batch of 1, but we can still use vmap-friendly shapes
        inp = {k: np.repeat(v[None, ...], 1, axis=0) for k, v in inp.items()}

        # repaint mask: 좌표 고정은 입력 분자의 기존 원자 인덱스까지만 의미가 있음
        # 새로 추가한 원자들은 mask=False 로 두어 자유롭게 이동
        n_orig = len(consts["atomic_numbers"])
        fixed_mask_now = np.concatenate([fixed_mask_heavy, np.zeros((len(an)-n_orig,), dtype=bool)], axis=0)
        repaint = build_repaint_dict(xyz, fixed_mask_now, args.natoms)
        repaint = {k: np.repeat(v[None, ...], 1, axis=0) for k, v in repaint.items()}

        # jax arrays
        inp = jax.tree_map(lambda x: jnp.array(x), inp)
        repaint = jax.tree_map(lambda x: jnp.array(x), repaint)

        # run
        rng_key = jax.random.PRNGKey(rng.randint(0, 2**31-1))
        structures, trajectories, _ = inference(inp, rng_key, repaint_dict=repaint)

        xyz_out = np.array(structures)[0][:len(an)]
        out_base = args.out_dir / f"sample_{i:04d}"
        fmt = args.output_format
        if fmt in ("xyz","both"):
            write_xyz(out_base.with_suffix(".xyz"), an, xyz_out)
        if fmt in ("mol","both"):
            ok, smi = try_write_mol(out_base.with_suffix(".mol"), an, hn, xyz_out)
            meta = {"smiles": smi, "mol_ok": bool(ok)}
            with open(out_base.with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)
        samples_written += 1

    print(f"Wrote {samples_written} samples to {args.out_dir}")

if __name__ == "__main__":
    main()
