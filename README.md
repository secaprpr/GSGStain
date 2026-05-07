### GSGStain Virtual IHC Training

This fork also includes a CUT-based GSGStain implementation for weakly paired H&E-to-IHC virtual staining. Prepare consecutive-section images with H&E slides in `trainA` and reference IHC slides in `trainB`; sorted files are treated as weak pairs by default.

```bash
python train.py --model gsgstain --gsg_stage train_gsrm --dataroot ./datasets/he2ihc --name he2ihc_GSRM
python train.py --model gsgstain --gsg_stage train_generator --dataroot ./datasets/he2ihc --name he2ihc_GSGStain --gsg_pretrained_R_name he2ihc_GSRM
```

The first stage trains only the graph semantic rectification module with HPC and IRC losses. The second stage freezes the pretrained GSRM and trains the CUT generator with adversarial, PatchNCE, graph semantic consistency, and dual-branch ranking losses. GSRM uses PyG `PNAConv` by default through `--gsrm_conv pna`, matching the paper's PNA-style graph reasoning more closely. The previous built-in mean/max/min/std implementation remains available for debugging or environments without PyG via `--gsrm_conv custom`. The default graph builder uses differentiable grid nodes so the model can train without mandatory Cellpose/UNI2-h preprocessing; precomputed cell-level features can be integrated by replacing the graph feature construction methods. A `--gsg_stage joint` mode is also available for ablations.

If precomputed cell graphs are available, place a cache at `./datasets/he2ihc/processedcut/train.pt`, or put per-image files under `./datasets/he2ihc/processedcut/train/*.pt` with filenames matching `trainA`. Each graph should provide `x_he`, `x_ihc`, and either normalized `coords` or pixel-space `centroids` plus `image_size`; the dataset will pad them to `--graph_max_nodes` and GSRM will use those real cell nodes instead of the grid fallback.

To build PyG cell graphs inside this project, run:

```bash
python build_graphs.py --dataroot ./datasets/he2ihc --phase train
```

This writes a single PyG graph cache to `./datasets/he2ihc/processedcut/train.pt`. The default morphology encoder is `resnet50`, which matches the project dependencies and avoids extra UNI2-h setup, and IHC priors are extracted from corresponding centroid-centered patches by default. Use `--ihc_region mask` for the stricter old mask-based extraction. Use `--morphology_encoder uni2 --graph_he_dim 1536` for paper-like experiments, or `--morphology_encoder rgb_stats --graph_he_dim 6` for fast debugging. Use `--graph_save_mode files` to instead write one `torch_geometric.data.Data` file per image under `processedcut/train/`. The graph script requires `cellpose` and `torch_geometric`; `--morphology_encoder vit` additionally requires `transformers`, and `--morphology_encoder uni2` requires `timm` plus access to the UNI2-h weights.

For the main reproduction setting, keep the default `--graph_construction_method knn --knn_k 8`. `--graph_construction_method delaunay` is intended only for extension experiments.
