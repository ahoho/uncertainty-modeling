# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Models of Annotation
======================================

In this example, we run MCMC for various crowdsourced annotation models in [1].

All models have discrete latent variables. Under the hood, we enumerate over
(marginalize out) those discrete latent sites in inference. Those models have different
complexity so they are great references for those who are new to Pyro/NumPyro
enumeration mechanism. We recommend readers compare the implementations with the
corresponding plate diagrams in [1] to see how concise a Pyro/NumPyro program is.

The interested readers can also refer to [3] for more explanation about enumeration.

The data is taken from Table 1 of reference [2].

Currently, this example does not include postprocessing steps to deal with "Label
Switching" issue (mentioned in section 6.2 of [1]).

**References:**

    1. Paun, S., Carpenter, B., Chamberlain, J., Hovy, D., Kruschwitz, U.,
       and Poesio, M. (2018). "Comparing bayesian models of annotation"
       (https://www.aclweb.org/anthology/Q18-1040/)
    2. Dawid, A. P., and Skene, A. M. (1979).
       "Maximum likelihood estimation of observer error‐rates using the EM algorithm"
    3. "Inference with Discrete Latent Variables"
       (http://pyro.ai/examples/enumeration.html)

"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import numpy as np
from tqdm import tqdm

from jax import nn, random, vmap, clear_caches
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import LocScaleReparam
from numpyro.ops.indexing import Vindex


def load_jsonl(file_path: str) -> List[dict]:
    """
    Load a JSONL file into a list of dictionaries.
    
    Parameters:
    -----------
    file_path : str
        Path to the JSONL file
    
    Returns:
    --------
    List[dict]: List of dictionaries
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(eval(line))
    return data


def create_annotator_mapping(data: List[dict]) -> Dict[Tuple[int, int], int]:
    """
    Create a consistent mapping from (annotator_id, position) to matrix position.
    This is used to ensure consistent indexing across train and test sets.
    
    Parameters:
    -----------
    data : List[dict]
        The full dataset before splitting
        
    Returns:
    --------
    Dict[Tuple[int, int], int]: Mapping from (annotator_id, position) to matrix position
    """
    # First pass: collect all annotator-position combinations
    annotator_positions = defaultdict(set)  # annotator_id -> set of positions they appear in
    
    for item in data:
        annotators = item['annotators']
        for pos, ann in enumerate(annotators):
            annotator_positions[ann].add(pos)
    
    # Create consistent position mapping for each annotator
    annotator_to_positions: Dict[Tuple[int, int], int] = {}
    current_position = 0
    
    # Sort annotators for deterministic output
    for annotator in sorted(annotator_positions.keys()):
        positions = sorted(annotator_positions[annotator])
        for pos in positions:
            annotator_to_positions[(annotator, pos)] = current_position
            current_position += 1
    
    return annotator_to_positions


def process_annotations(data: List[dict], annotator_mapping: Optional[Dict[Tuple[int, int], int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process annotation data into the required format, handling variable numbers of annotators.
    
    Parameters:
    -----------
    data : List[dict]
        Annotation data
    annotator_mapping : Dict[Tuple[int, int], int], optional
        Pre-defined mapping from (annotator_id, position) to matrix position.
        If None, a new mapping will be created.
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - positions: array of annotator indices (0-based)
        - annotations: 2D array of class labels (0-based) with placeholder zeros for missing data
        - masks: boolean 2D array indicating which entries in annotations are valid
    """
    # Create mapping if not provided
    if annotator_mapping is None:
        annotator_mapping = create_annotator_mapping(data)
    
    # Determine the total number of matrix positions
    total_positions = max(annotator_mapping.values()) + 1
    
    # Create the positions and annotations arrays
    positions = np.zeros(total_positions, dtype=int)
    annotations = np.zeros((len(data), total_positions), dtype=int)  # Use 0 as placeholder
    masks = np.zeros((len(data), total_positions), dtype=bool)       # Mask to indicate valid entries
    
    # Fill in sparse matrix with available annotations
    for item_idx, item in enumerate(data):
        for pos, (annotator, label) in enumerate(zip(item['annotators'], item['labels'])):
            if (annotator, pos) in annotator_mapping:
                matrix_pos = annotator_mapping[(annotator, pos)]
                annotations[item_idx, matrix_pos] = label
                masks[item_idx, matrix_pos] = True
                positions[matrix_pos] = annotator
    
    return positions, annotations, masks


def multinomial(annotations):
    """
    This model corresponds to the plate diagram in Figure 1 of reference [1].
    """
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample("zeta", dist.Dirichlet(jnp.ones(num_classes)))

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi), infer={"enumerate": "parallel"})

        with numpyro.plate("position", num_positions):
            numpyro.sample("y", dist.Categorical(zeta[c]), obs=annotations)


def dawid_skene(positions, annotations):
    """
    This model corresponds to the plate diagram in Figure 2 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            beta = numpyro.sample("beta", dist.Dirichlet(jnp.ones(num_classes)))

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi), infer={"enumerate": "parallel"})

        # here we use Vindex to allow broadcasting for the second index `c`
        # ref: http://num.pyro.ai/en/latest/utilities.html#numpyro.contrib.indexing.vindex
        with numpyro.plate("position", num_positions):
            numpyro.sample(
                "y", dist.Categorical(Vindex(beta)[positions, c, :]), obs=annotations
            )


def mace(positions, annotations):
    """
    This model corresponds to the plate diagram in Figure 3 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("annotator", num_annotators):
        epsilon = numpyro.sample("epsilon", dist.Dirichlet(jnp.full(num_classes, 10)))
        theta = numpyro.sample("theta", dist.Beta(0.5, 0.5))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample(
            "c",
            dist.DiscreteUniform(0, num_classes - 1),
            infer={"enumerate": "parallel"},
        )

        with numpyro.plate("position", num_positions):
            s = numpyro.sample(
                "s",
                dist.Bernoulli(1 - theta[positions]),
                infer={"enumerate": "parallel"},
            )
            probs = jnp.where(
                s[..., None] == 0, nn.one_hot(c, num_classes), epsilon[positions]
            )
            numpyro.sample("y", dist.Categorical(probs), obs=annotations)


def hierarchical_dawid_skene(positions, annotations):
    """
    This model corresponds to the plate diagram in Figure 4 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        # NB: we define `beta` as the `logits` of `y` likelihood; but `logits` is
        # invariant up to a constant, so we'll follow [1]: fix the last term of `beta`
        # to 0 and only define hyperpriors for the first `num_classes - 1` terms.
        zeta = numpyro.sample(
            "zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        omega = numpyro.sample(
            "Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            # non-centered parameterization
            with handlers.reparam(config={"beta": LocScaleReparam(0)}):
                beta = numpyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
            # pad 0 to the last item
            beta = jnp.pad(beta, [(0, 0)] * (jnp.ndim(beta) - 1) + [(0, 1)])

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi), infer={"enumerate": "parallel"})

        with numpyro.plate("position", num_positions):
            logits = Vindex(beta)[positions, c, :]
            numpyro.sample("y", dist.Categorical(logits=logits), obs=annotations)


def item_difficulty(annotations):
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        eta = numpyro.sample(
            "eta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        chi = numpyro.sample(
            "Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi), infer={"enumerate": "parallel"})

        with handlers.reparam(config={"theta": LocScaleReparam(0)}):
            theta = numpyro.sample("theta", dist.Normal(eta[c], chi[c]).to_event(1))
            theta = jnp.pad(theta, [(0, 0)] * (jnp.ndim(theta) - 1) + [(0, 1)])

        with numpyro.plate("position", annotations.shape[-1]):
            numpyro.sample("y", dist.Categorical(logits=theta), obs=annotations)


def logistic_random_effects(positions, annotations):
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_annotators = int(np.max(positions)) + 1
    num_classes = int(np.max(annotations)) + 1
    num_items, num_positions = annotations.shape

    with numpyro.plate("class", num_classes):
        zeta = numpyro.sample(
            "zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1)
        )
        omega = numpyro.sample(
            "Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )
        chi = numpyro.sample(
            "Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1)
        )

    with numpyro.plate("annotator", num_annotators, dim=-2):
        with numpyro.plate("class", num_classes):
            with handlers.reparam(config={"beta": LocScaleReparam(0)}):
                beta = numpyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
                beta = jnp.pad(beta, [(0, 0)] * (jnp.ndim(beta) - 1) + [(0, 1)])

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(num_classes)))

    with numpyro.plate("item", num_items, dim=-2):
        c = numpyro.sample("c", dist.Categorical(pi), infer={"enumerate": "parallel"})

        with handlers.reparam(config={"theta": LocScaleReparam(0)}):
            theta = numpyro.sample("theta", dist.Normal(0, chi[c]).to_event(1))
            theta = jnp.pad(theta, [(0, 0)] * (jnp.ndim(theta) - 1) + [(0, 1)])

        with numpyro.plate("position", num_positions):
            logits = Vindex(beta)[positions, c, :] - theta
            numpyro.sample("y", dist.Categorical(logits=logits), obs=annotations)


def chunk_predictive(model, posterior_samples, rng_key, data, chunk_size=10):
    num_samples = len(posterior_samples["pi"])
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    combined_discrete_samples = []
    
    for i in tqdm(range(num_chunks), desc="Predicting..."):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, num_samples)
        chunk_samples = {k: v[chunk_start:chunk_end] for k, v in posterior_samples.items()}
        predictive = Predictive(model, chunk_samples, infer_discrete=True)
        gc.collect()
        if i % 10 == 0:
            clear_caches()
        chunk_discrete_samples = predictive(random.fold_in(rng_key, i), *data)
        chunk_samples = {k: np.array(v) for k, v in chunk_discrete_samples.items() if k != "y"}
        if i == 0:
            combined_discrete_samples = chunk_samples
        else:
            combined_discrete_samples = {
                k: np.concatenate([combined_discrete_samples[k], v], axis=0)
                for k, v in chunk_samples.items()
            }
    return combined_discrete_samples


NAME_TO_MODEL = {
    "mn": multinomial,
    "ds": dawid_skene,
    "mace": mace,
    "hds": hierarchical_dawid_skene,
    "id": item_difficulty,
    "lre": logistic_random_effects,
}


def main(args):
    # load data
    data = load_jsonl(args.input_file)

    annotators, annotations, _ = process_annotations(data)
    model = NAME_TO_MODEL[args.model]
    data = (
        (annotations,)
        if model in [multinomial, item_difficulty]
        else (annotators, annotations)
    )

    mcmc = MCMC(
        NUTS(model),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(random.PRNGKey(0), *data)
    mcmc.print_summary()

    posterior_samples = mcmc.get_samples()
    discrete_samples = chunk_predictive(model, posterior_samples, random.PRNGKey(1), data)
    
    # item_class = vmap(lambda x: jnp.bincount(x, length=4), in_axes=1)(
    #     discrete_samples["c"].squeeze(-1)
    # )
    # print("Histogram of the predicted class of each item:")
    # row_format = "{:>10}" * 5
    # print(row_format.format("", *["c={}".format(i) for i in range(4)]))
    # for i, row in enumerate(item_class):
    #     print(row_format.format(f"item[{i}]", *row))

    # save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # move posterior samples to cpu
    np.savez(
        os.path.join(args.output_dir, f"{args.model}_posterior_samples.npz"),
        **posterior_samples,
    )
    np.savez(
        os.path.join(args.output_dir, f"{args.model}_discrete_samples.npz"),
        **discrete_samples,
    )


# %%
# .. note::
#     In the above inference code, we marginalized the discrete latent variables `c`
#     hence `mcmc.get_samples(...)` does not include samples of `c`. We then utilize
#     `Predictive(..., infer_discrete=True)` to get posterior samples for `c`, which
#     is stored in `discrete_samples`. To merge those discrete samples into the `mcmc`
#     instance, we can use the following pattern::
#
#         chain_discrete_samples = jax.tree.map(
#             lambda x: x.reshape((args.num_chains, args.num_samples) + x.shape[1:]),
#             discrete_samples)
#         mcmc.get_samples().update(discrete_samples)
#         mcmc.get_samples(group_by_chain=True).update(chain_discrete_samples)
#
#     This is useful when we want to pass the `mcmc` instance to `arviz` through
#     `arviz.from_numpyro(mcmc)`.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Models of Annotation")
    parser.add_argument("--input_file", default="ghc_train.jsonl", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("-n", "--num_samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num_warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num_chains", nargs="?", default=1, type=int)
    parser.add_argument(
        "--model",
        nargs="?",
        default="ds",
        help='one of "mn" (multinomial), "ds" (dawid_skene), "mace",'
        ' "hds" (hierarchical_dawid_skene),'
        ' "id" (item_difficulty), "lre" (logistic_random_effects)',
    )
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
