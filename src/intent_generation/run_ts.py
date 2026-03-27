# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Main script for tree search implementation
"""

import json
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig

from src.intent_generation.tree_search import run_tree_search
from src.intent_generation.beam_search import run_beam_search
from src.utils import RankedLogger, print_config_tree

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> None:
    """
    Run tree search with data loading
    
    Args:
        cfg: Configuration
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # Load checkpoint if provided
    if cfg.get("ckpt_path"):
        log.info(f"Loading checkpoint from: {cfg.ckpt_path}")
        model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"], strict=False)

    # Prepare output directory
    output_dir = Path(cfg.paths.output_dir) / "tree_search_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "tree_search_results.json"
    
    # Prepare data
    datamodule.prepare_data()
    datamodule.setup("validate")
    validate_dataloader = datamodule.val_dataloader()

    # Results storage
    all_results = []

    # Run tree search on test data
    log.info("Running search on validate data...")
    for batch_idx, data in enumerate(validate_dataloader):
        if batch_idx >= cfg.get("max_batches", 10):
            break
        
        log.info(f"Processing batch {batch_idx+1}")
        
        # Tokenize data
        tokenized_map, tokenized_agent = model.token_processor(data)
        
        # Run search based on search type
        search_type = cfg.tree_search.get("search_type", "tree")
        
        if search_type == "tree":
            search_config = cfg.tree_search.tree_search
            result = run_tree_search(model, tokenized_map, tokenized_agent, search_config)
        elif search_type == "beam":
            search_config = cfg.tree_search.beam_search
            result = run_beam_search(model, tokenized_map, tokenized_agent, search_config)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Extract scenario and agent information
        scenario_ids = data.get("scenario_id", [f"batch_{batch_idx}_scenario_{i}" for i in range(len(data["agent"]["id"]))])
        agent_ids = data["agent"]["id"].tolist()
        
        # Store results
        for i, (scenario_id, agent_id) in enumerate(zip(scenario_ids, agent_ids)):
            agent_result = {
                "scenario_id": scenario_id,
                "agent_id": agent_id,
                "pred_traj_10hz": result["pred_traj_10hz"][i].tolist(),
                "pred_head_10hz": result["pred_head_10hz"][i].tolist(),
                "pred_pos": result["pred_pos"][i].tolist(),
                "pred_head": result["pred_head"][i].tolist()
            }
            all_results.append(agent_result)
        
        # Log results
        log.info(f"{search_type.capitalize()} search completed for batch {batch_idx+1}")
        log.info(f"Predicted trajectory shape: {result['pred_traj_10hz'].shape}")
        log.info(f"Processed {len(agent_ids)} agents in this batch")
    
    # Save results to file
    log.info(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    log.info(f"Saved {len(all_results)} agent trajectories to {output_file}")


@hydra.main(config_path="../../configs/", config_name="run_ts.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for tree search
    
    Args:
        cfg: Configuration
    """
    torch.set_printoptions(precision=3)

    # Skip printing config tree to avoid interpolation errors
    # log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    # print_config_tree(cfg, resolve=True, save_to_file=True)

    run(cfg)  # run tree search

    log.info(f"Output dir: {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
    log.info("run_ts.py DONE!!!")