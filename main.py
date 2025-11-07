# pip install rdkit-pypi grakel fcd_torch numpy scipy scikit-learn
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

from rdkit import Chem

from grakel import Graph
# https://ysig.github.io/GraKeL/0.1a8/kernels/neighborhood_subgraph_pairwise_distance.html
from grakel import NeighborhoodSubgraphPairwiseDistance as NSPD

# https://github.com/insilicomedicine/fcd_torch
from fcd_torch import FCD

# Parse a SMILES into a molecule class, rdkit.Chem.rdchem.Mol. 
def parse_smiles_list(smiles: Iterable[str], sanitize: bool = True):
# sanitize = True checks for common issues like closed rings, correct valency, and chemical validity
# True by default
    out = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(s, sanitize=sanitize)
        except Exception:
            m = None  
        out.append(m)
    return out

# Returns the canonical SMILES string for a molecule
def canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None
    
# check validity, uniqueness, and novelty
# Input: lists of generated smiles and training smiles
# output: validity, uniqueness, and novelty of generated smiles
# from Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations:
# Validity is the fraction of the generated molecules that do not violate the chemical valency rule. 
# Uniqueness is the fraction of the valid molecules that are unique. 
# Novelty is the fraction of the valid molecules that are not included in the training set.
def validity_uniqueness_novelty(
    gen_smiles: Iterable[str],
    train_smiles: Iterable[str] # list or iterable of all training SMILES strings
) -> Dict[str, Any]:
    gen_mols = parse_smiles_list(gen_smiles)
    valid_cans: List[str] = []
    for m in gen_mols:
        can = canonical_smiles(m)
        if can is not None:
            valid_cans.append(can)
            
    validity = 0.0 if len(gen_smiles) == 0 else len(valid_cans) / len(list(gen_smiles))

    uniqueness = 0.0 if not valid_cans else len(set(valid_cans)) / len(valid_cans)

    train_mols = parse_smiles_list(train_smiles, sanitize=True)
    train_can = set([c for c in (canonical_smiles(m) for m in train_mols) if c is not None])

    if valid_cans:
        novel_cnt = sum(1 for s in valid_cans if s not in train_can)
        novelty = novel_cnt / len(valid_cans)
    else:
        novelty = 0.0

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "valid_canonical": valid_cans, 
    }
    
# NSPDK MMD: Neighborhood Subgraph Pairwise Distance Kernel 
# The lower the better
@dataclass
class GraphData:
    nodes: List[int]
    edges: List[Tuple[int, int]]
    node_labels: Dict[int, Any]
    edge_labels: Dict[Tuple[int, int], Any] 
    
def mol_to_graphdata(m: Chem.Mol) -> GraphData:
    nodes = list(range(m.GetNumAtoms()))
    edges, node_labels, edge_labels = [], {}, {}
    for a in m.GetAtoms():
        node_labels[a.GetIdx()] = a.GetAtomicNum()
    for b in m.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u == v: 
            continue
        bond_order = int(b.GetBondTypeAsDouble())
        aromatic = int(b.GetIsAromatic())
        lab = (bond_order, aromatic)
        edges.append((u, v))
        edge_labels[(u, v)] = lab
        edge_labels[(v, u)] = lab
    return GraphData(nodes, edges, node_labels, edge_labels)

def graphs_to_grakel(graphs: List[GraphData]) -> List[Graph]:
    return [
        Graph(
            initialization_object=g.edges,         
            node_labels=g.node_labels,              
            edge_labels=g.edge_labels,              
            graph_format="dictionary"              
        )
        for g in graphs
    ]

def mmd2_from_gram(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    n, m = Kxx.shape[0], Kyy.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    Kxx = Kxx.copy(); Kyy = Kyy.copy()
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_x = Kxx.sum() / (n * (n - 1))
    term_y = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()
    return float(term_x + term_y - 2 * term_xy)

def nspd_mmd_from_smiles(
    ref_smiles: Iterable[str],
    gen_smiles: Iterable[str],
    radius: int = 2,
    distance: int = 4,
    n_jobs: int = 1,
    normalize: bool = False
) -> Dict[str, float]:
    ref_mols = [m for m in parse_smiles_list(ref_smiles) if m is not None]
    gen_mols = [m for m in parse_smiles_list(gen_smiles) if m is not None]

    Gx = graphs_to_grakel([mol_to_graphdata(m) for m in ref_mols])
    Gy = graphs_to_grakel([mol_to_graphdata(m) for m in gen_mols])

    kernel = NSPD(r=radius, d=distance, n_jobs=n_jobs, normalize=normalize)
    Gall = Gx + Gy
    K = kernel.fit_transform(Gall)

    n = len(Gx)
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    mmd2 = mmd2_from_gram(Kxx, Kyy, Kxy)
    return {
        "NSPD_MMD2": mmd2,
        "NSPD_MMD": float(np.sqrt(mmd2)) if (mmd2 >= 0) else float("nan")
    }
    
# FCD : Fréchet ChemNet Distance
# quantifies the chemical and biological similarity between a set of 
# generated molecules and a set of real, reference molecules
def fcd_value(
    smiles_ref: Iterable[str],
    smiles_gen: Iterable[str],
    device: str = "cpu",
    n_jobs: int = 4,
    batch_size: int = 512
) -> float:
    calc = FCD(device=device, n_jobs=n_jobs, batch_size=batch_size)
    return float(calc(ref=list(smiles_ref), gen=list(smiles_gen)))

def fcd_value_with_precalc(
    smiles_ref: Iterable[str],
    smiles_gen: Iterable[str],
    device: str = "cpu",
    n_jobs: int = 4,
    batch_size: int = 512
) -> Dict[str, Any]:
    calc = FCD(device=device, n_jobs=n_jobs, batch_size=batch_size)
    pref = calc.precalc(list(smiles_ref))
    val = float(calc(pref=pref, gen=list(smiles_gen)))
    return {"FCD": val, "pref": pref}

# example 
if __name__ == "__main__":
    gen = ["CCO", "c1ccccc1", "N(N)N", "C1CC1"]
    train = ["CCO", "CCN", "CCC", "c1ccccc1O"]
    # print(validity_uniqueness_novelty(gen, train))
    print(nspd_mmd_from_smiles(gen, train))
    # print(fcd_value(gen, train))