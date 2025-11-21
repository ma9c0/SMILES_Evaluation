# pip install rdkit-pypi grakel fcd_torch numpy scipy scikit-learn
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

from rdkit import Chem

import networkx as nx
from sklearn.metrics.pairwise import pairwise_kernels

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
def mol_to_nx(mol: Chem.Mol) -> nx.Graph:
    """Convert an RDKit molecule to a NetwrokX graph for vectorizing.

    Args:
        mol (Chem.Mol)

    Returns:
        nx.Graph:
    """
    
    G = nx.Graph()
    
    for a in mol.GetAtoms():
        idx, sym = a.GetIdx(), a.GetSymbol()
        chg = a.GetFormalCharge()
        aro = int(a.GetIsAromatic())
        hcnt = a.GetTotalNumHs()
        nlabel = f'{sym}|chg={chg}|aro={aro}|H={hcnt}'
        G.add_node(idx, label=nlabel)
        
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u==v:
            continue
        order = int(b.GetBondTypeAsDouble())
        aro = int(b.GetIsAromatic())
        elabel = f'{order}:{aro}'
        G.add_edge(u, v, label=elabel)
    
    return G

### Vectorization
_nbits = 16
_bitmask_ = (1 << _nbits) - 1 
from scipy.sparse import csr_matrix

def fast_hash_2(dat_1, dat_2, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2)) & bitmask) + 1


def fast_hash_3(dat_1, dat_2, dat_3, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2, dat_3)) & bitmask) + 1


def fast_hash_4(dat_1, dat_2, dat_3, dat_4, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2, dat_3, dat_4)) & bitmask) + 1

def fast_hash_multiset(vals, bitmask=_bitmask_):
    return (hash(tuple(sorted(vals))) & bitmask) + 1

from collections import deque, defaultdict

def _all_pairs_shortest_paths_upto_0(G: nx.Graph, D: int):
    shells = {}
    for src in G.nodes:
        dist_map = {0: {src}}
        visited = {src}
        q = deque([(src, 0)])
        while q:
            u, d = q.popleft()
            if d == 0:
                continue
            nd = d + 1
            for v in G.neighbors(u):
                if v in visited:
                    continue
                visited.add(v)
                q.append((v, nd))
                dist_map.setdefault(nd, set()).add(v)
        shells[src] = dist_map
    return shells

def _edge_label(eattr):
    lab = eattr.get("label", None)
    if isinstance(lab, tuple):
        return (hash(lab) & _bitmask_) + 1
    return (hash(lab) & _bitmask_) + 1

def _incident_bond_signature(G: nx.Graph, u: int):
    labs = []
    for v in G.neighbors(u):
        labs.append(_edge_label(G.edges[u, v]))
    return fast_hash_multiset(labs)

def _neighborhood_hashes(G: nx.Graph, R: int, D: int, label_key: str = "label"):
    deg = dict(G.degree())
    ebond = {}
    labhash = {}
    for u in G.nodes():
        ebond[u] = _incident_bond_signature(G, u)
        lab = G.nodes[u].get(label_key, None)
        labhash[u] = (hash(lab) & _bitmask_) + 1

    # shells up to max(R, D)
    Dmax = max(R, D)
    shells = _all_pairs_shortest_paths_upto_0(G, Dmax)

    # cumulative neighborhood hashes <= r (include nodes at all distances <= r)
    neigh_hash = defaultdict(dict)
    for u in G.nodes():
        acc = []
        for r in range(0, R + 1):
            for v in shells[u].get(r, []):
                code_v = fast_hash_3(labhash[v], deg.get(v, 0), ebond[v], _bitmask_)
                acc.append(code_v)
            neigh_hash[u][r] = fast_hash_multiset(acc)
    return neigh_hash, shells

def _feature_dict_for_graph(G: nx.Graph, R: int, D : int) -> dict:
    if G.number_of_nodes() == 0:
        return {}

    neigh_hash, shells = _neighborhood_hashes(G, R, D)

    grouped = defaultdict(lambda: defaultdict(float)) 

    nodes = list(G.nodes())

    for u in nodes:
        for r in range(0, R + 1):
            hu = neigh_hash[u][r]
            root_feat = fast_hash_2(hu, r, _bitmask_) 
            grouped[(r, 0)][root_feat] += 1.0

    for u in nodes:
        for d in range(1, D + 1):
            at_d = shells[u].get(d, set())
            if not at_d:
                continue
            for v in at_d:
                for r in range(0, R + 1):
                    hu = neigh_hash[u][r]
                    hv = neigh_hash[v][r]
                    a, b = (hu, hv) if hu <= hv else (hv, hu)
                    feat = fast_hash_4(a, b, r, d, _bitmask_)
                    half = fast_hash_3(hv, r, d, _bitmask_)
                    grouped[(r, d)][feat] += 1.0
                    grouped[(r, d)][half] += 1.0

    feat_vec = {}
    for rd, bag in grouped.items():
        norm2 = sum(v * v for v in bag.values())
        if norm2 <= 0:
            continue
        inv = 1.0 / (norm2 ** 0.5)
        for fid, val in bag.items():
            feat_vec[fid] = feat_vec.get(fid, 0.0) + val * inv

    # global L2 normalization
    total2 = sum(v * v for v in feat_vec.values())
    if total2 > 0:
        inv_total = 1.0 / (total2 ** 0.5)
        for fid in list(feat_vec.keys()):
            feat_vec[fid] *= inv_total

    if not feat_vec:
        return {1: 1.0}
    return feat_vec

_EDEN_FEATURE_SIZE = _bitmask_ + 2

def dicts_to_csr(rows: list):
    if not rows:
        return csr_matrix((0, _EDEN_FEATURE_SIZE), dtype=np.float64)
    data, ind, indptr = [], [], [0]
    for row in rows:
        if not row:
            ind.append(1); data.append(1.0)
        else:
            for k, v in row.items():
                ind.append(int(k))
                data.append(float(v))
        indptr.append(len(ind))
    return csr_matrix((np.asarray(data, dtype=np.float64),
                       np.asarray(ind, dtype=np.int64),
                       np.asarray(indptr, dtype=np.int64)),
                      shape=(len(rows), _EDEN_FEATURE_SIZE))

def vectorize(graphs, complexity: int = 4, discrete: bool = True):
    R = int(complexity)
    D = int(complexity)
    rows = []
    for g in graphs:
        if not isinstance(g, nx.Graph):
            raise TypeError("vectorize expects NetworkX Graphs with node attribute 'label'.")
        if g.number_of_nodes() == 0:
            rows.append({0: 0.0})
            continue
        rows.append(_feature_dict_for_graph(g, R=R, D=D))
    return dicts_to_csr(rows)

def _vectorize_mols_nspdk(mols: List[Chem.Mol], complexity: int = 4, discrete: bool = True):
    """Vectorize a list of RDKit mols

    Args:
        mols (List[Chem.Mol]): _description_
        complexity (int, optional): _description_. Defaults to 4.
        discrete (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    graphs = [mol_to_nx(m) for m in mols if m is not None and m.GetNumAtoms() > 0]
    if len(graphs) == 0:
        return np.zeros((0,1), dtype=np.float(32))
    X = vectorize(graphs, complexity = complexity, discrete= discrete)
    
    return X

def __mmd2_from_feature_linear(X, Y, n_jobs: int = 1,) -> float:
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return float('nan')
    Kxx = pairwise_kernels(X, None, metric='linear', n_jobs=n_jobs)
    Kyy = pairwise_kernels(Y, None, metric='linear', n_jobs=n_jobs)
    Kxy = pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)
    
    Kxx = Kxx.copy(); Kyy = Kyy.copy()
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_x  = Kxx.sum() / (n * (n - 1))
    term_y  = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()
    return float(term_x + term_y - 2.0 * term_xy)

def nspdk_mmd_from_smiles(
    ref_smiles: Iterable[str],
    gen_smiles: Iterable[str],
    *,
    complexity: int = 4,
    n_jobs: int = 1,
    discrete: bool = True,
) -> Dict[str, float]:
    """
    Compute NSPDK MMD between two SMILES sets using eden NSPDK features + linear kernel.
    Returns both MMD^2 and MMD for convenience.
    """
    ref_mols = [m for m in parse_smiles_list(ref_smiles) if m is not None]
    gen_mols = [m for m in parse_smiles_list(gen_smiles) if m is not None]

    X = _vectorize_mols_nspdk(ref_mols, complexity=complexity, discrete=discrete)
    Y = _vectorize_mols_nspdk(gen_mols, complexity=complexity, discrete=discrete)
    print(X.shape, Y.shape, X.nnz, Y.nnz)
    mmd2 = __mmd2_from_feature_linear(X, Y, n_jobs=n_jobs)
    return {
        "NSPDK_MMD2": mmd2,
        "NSPDK_MMD": float(np.sqrt(mmd2)) if (mmd2 >= 0) else float("nan"),
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
    train = [
        "CCCC",        # butane
        "CCO",         # ethanol
        "CCCO",        # propanol
        "CCCCO",       # butanol
    ]

    gen = [
        "c1ccccc1",    # benzene
        "c1ccncc1",    # pyridine
        "CC(=O)O",     # acetic acid
        "CC(=O)OC",    # methyl acetate
    ]

    # print(validity_uniqueness_novelty(gen, train))
    print(nspdk_mmd_from_smiles(gen, train))
    # print(fcd_value(gen, train))