import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import Data
from .stain_utils import dab_od_from_rgb, hed_from_rgb


class GraphConstructor(torch.nn.Module):
    """Build cell-centric H&E-IHC PyG graphs for GSGStain.

    The constructor follows the paper's graph construction recipe: Cellpose
    nodes on H&E, contextual H&E node features, handcrafted DAB/OD IHC semantic
    features, kNN/Delaunay topology, and morphology-similarity edge weights.
    The saved graph is a torch_geometric.data.Data object.
    """

    @dataclass
    class Config:
        seg_model_type: str = 'cyto'
        morphology_encoder: str = 'resnet50'
        feature_extractor_name: str = 'google/vit-base-patch16-224'
        graph_construction_method: str = 'knn'
        knn_k: int = 8
        patch_size: int = 96
        ihc_region: str = 'patch'
        max_edge_distance: Optional[float] = None
        positive_threshold: float = 0.15
        zscore_ihc: bool = False
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument('--seg_model_type', type=str, default=cls.Config.seg_model_type)
        parser.add_argument('--morphology_encoder', type=str, default=cls.Config.morphology_encoder,
                            choices=['rgb_stats', 'resnet50', 'vit', 'uni2'])
        parser.add_argument('--feature_extractor_name', type=str, default=cls.Config.feature_extractor_name)
        parser.add_argument('--graph_construction_method', type=str, default=cls.Config.graph_construction_method,
                            choices=['knn', 'delaunay'])
        parser.add_argument('--knn_k', type=int, default=cls.Config.knn_k)
        parser.add_argument('--patch_size', type=int, default=cls.Config.patch_size)
        parser.add_argument('--ihc_region', type=str, default=cls.Config.ihc_region, choices=['patch', 'mask'])
        parser.add_argument('--max_edge_distance', type=float, default=cls.Config.max_edge_distance)
        parser.add_argument('--positive_threshold', type=float, default=cls.Config.positive_threshold)
        parser.add_argument('--zscore_ihc', action='store_true')
        parser.add_argument('--device', type=str, default=cls.Config.device)
        return parser

    @classmethod
    def from_cli_args(cls, args):
        return cls(
            seg_model_type=args.seg_model_type,
            morphology_encoder=args.morphology_encoder,
            feature_extractor_name=args.feature_extractor_name,
            graph_construction_method=args.graph_construction_method,
            knn_k=args.knn_k,
            patch_size=args.patch_size,
            ihc_region=args.ihc_region,
            max_edge_distance=args.max_edge_distance,
            positive_threshold=args.positive_threshold,
            zscore_ihc=args.zscore_ihc,
            device=args.device,
        )

    def __init__(self,
                 seg_model_type='cyto',
                 morphology_encoder='resnet50',
                 feature_extractor_name='google/vit-base-patch16-224',
                 graph_construction_method='knn',
                 knn_k=8,
                 patch_size=96,
                 ihc_region='patch',
                 max_edge_distance=None,
                 positive_threshold=0.15,
                 zscore_ihc=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.morphology_encoder_type = morphology_encoder.lower()
        self.graph_construction_method = graph_construction_method
        self.knn_k = knn_k
        self.patch_size = int(patch_size)
        self.ihc_region = ihc_region
        self.max_edge_distance = max_edge_distance
        self.positive_threshold = positive_threshold
        self.zscore_ihc = zscore_ihc

        self.seg_model = self._load_cellpose(seg_model_type)
        self.morph_encoder = None
        self.resnet_transform = None
        self.feature_processor = None
        self.feature_extractor = None
        self._uni2_transform = None
        self.morph_feature_dim = 6
        self._load_morphology_encoder(feature_extractor_name)

        self.register_buffer('hed_from_rgb', hed_from_rgb())

    def _load_cellpose(self, seg_model_type):
        try:
            from cellpose import models
        except ImportError as err:
            raise ImportError("Graph construction requires cellpose. Install it with `pip install cellpose`.") from err
        return models.CellposeModel(model_type=seg_model_type, gpu=(self.device.type == 'cuda'))

    def _load_morphology_encoder(self, feature_extractor_name):
        if self.morphology_encoder_type == 'rgb_stats':
            self.morph_feature_dim = 6
            return
        if self.morphology_encoder_type == 'resnet50':
            try:
                from torchvision import models, transforms
            except ImportError as err:
                raise ImportError("morphology_encoder='resnet50' requires torchvision.") from err
            try:
                weights = models.ResNet50_Weights.DEFAULT
                resnet = models.resnet50(weights=weights)
                self.resnet_transform = weights.transforms()
            except AttributeError:
                resnet = models.resnet50(pretrained=True)
                self.resnet_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            self.morph_encoder = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
            self.morph_encoder.eval()
            self.morph_feature_dim = 2048
            return
        if self.morphology_encoder_type == 'vit':
            try:
                from transformers import ViTImageProcessor, ViTModel
            except ImportError as err:
                raise ImportError("morphology_encoder='vit' requires transformers.") from err
            self.feature_processor = ViTImageProcessor.from_pretrained(feature_extractor_name)
            self.feature_extractor = ViTModel.from_pretrained(feature_extractor_name).to(self.device)
            self.feature_extractor.eval()
            self.morph_feature_dim = int(self.feature_extractor.config.hidden_size)
            return
        if self.morphology_encoder_type == 'uni2':
            try:
                import timm
                from timm.data import resolve_data_config
                from timm.data.transforms_factory import create_transform
                try:
                    from timm.layers import SwiGLUPacked
                except Exception:
                    SwiGLUPacked = None
            except ImportError as err:
                raise ImportError("morphology_encoder='uni2' requires timm.") from err

            timm_kwargs = {
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667 * 2,
                'num_classes': 0,
                'no_embed_class': True,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True,
            }
            if SwiGLUPacked is not None:
                timm_kwargs['mlp_layer'] = SwiGLUPacked
            self.morph_encoder = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).to(self.device)
            self.morph_encoder.eval()
            cfg = resolve_data_config(self.morph_encoder.pretrained_cfg, model=self.morph_encoder)
            self._uni2_transform = create_transform(**cfg)
            self.morph_feature_dim = getattr(self.morph_encoder, 'num_features', 1536)
            return
        raise ValueError("morphology_encoder must be one of rgb_stats, resnet50, vit, or uni2")

    @torch.no_grad()
    def forward(self, he_pil, ihc_pil):
        he_np = np.array(he_pil.convert('RGB'))
        masks = self.segment_cells(he_np)
        if int(masks.max()) == 0:
            return None

        edge_index, centroids = self.build_graph_topology(masks)
        x_he = self.extract_he_features(he_pil, masks)
        x_ihc = self.extract_ihc_features(ihc_pil, masks)
        if x_he.numel() == 0 or x_ihc.numel() == 0:
            return None

        edge_attr = self.edge_weights(edge_index, x_he)
        adjacency = self.dense_adjacency(edge_index, edge_attr, x_he.size(0))
        centroids_t = torch.tensor(centroids, dtype=torch.float32)
        image_size = torch.tensor([he_np.shape[0], he_np.shape[1]], dtype=torch.float32)
        coords_yx = centroids_t / torch.clamp(image_size.view(1, 2), min=1.0)
        coords_xy = torch.stack([coords_yx[:, 1], coords_yx[:, 0]], dim=1).clamp(0.0, 1.0)
        graph = Data(
            x=torch.cat([x_he.detach().cpu(), coords_xy, x_ihc.detach().cpu()], dim=1),
            x_he=x_he.detach().cpu(),
            x_ihc=x_ihc.detach().cpu(),
            coords=coords_xy.detach().cpu(),
            centroids=centroids_t.detach().cpu(),
            edge_index=edge_index.detach().cpu(),
            edge_attr=edge_attr.detach().cpu(),
            adjacency=adjacency.detach().cpu(),
            image_size=image_size.detach().cpu(),
        )
        graph.num_nodes = int(x_he.size(0))
        return graph

    def segment_cells(self, he_np):
        result = self.seg_model.eval(he_np, diameter=None)
        masks = result[0] if isinstance(result, tuple) else result
        if isinstance(masks, list):
            masks = masks[0]
        return np.asarray(masks)

    @torch.no_grad()
    def extract_he_features(self, he_pil, masks):
        he_np = np.array(he_pil.convert('RGB'))
        h, w = he_np.shape[:2]
        half = self.patch_size // 2
        features = []
        for node_id in range(1, int(masks.max()) + 1):
            rows, cols = np.where(masks == node_id)
            if rows.size == 0 or cols.size == 0:
                continue
            cy, cx = int(np.mean(rows)), int(np.mean(cols))
            y0, y1 = cy - half, cy + half + 1
            x0, x1 = cx - half, cx + half + 1
            pad_top = max(0, -y0)
            pad_left = max(0, -x0)
            pad_bottom = max(0, y1 - h)
            pad_right = max(0, x1 - w)
            if pad_top or pad_bottom or pad_left or pad_right:
                padded = np.pad(he_np, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
                y0 += pad_top
                y1 += pad_top
                x0 += pad_left
                x1 += pad_left
            else:
                padded = he_np
            patch_np = padded[y0:y1, x0:x1]
            patch = Image.fromarray(patch_np.astype(np.uint8))
            features.append(self.encode_patch(patch))
        if not features:
            return torch.empty(0, self.morph_feature_dim, device=self.device)
        return torch.stack(features)

    @torch.no_grad()
    def encode_patch(self, patch_pil):
        if self.morphology_encoder_type == 'rgb_stats':
            patch = torch.from_numpy(np.array(patch_pil).astype(np.float32) / 255.0).to(self.device)
            flat = patch.view(-1, 3)
            return torch.cat([flat.mean(dim=0), flat.std(dim=0)], dim=0)
        if self.morphology_encoder_type == 'resnet50':
            patch_tensor = self.resnet_transform(patch_pil).unsqueeze(0).to(self.device)
            emb = self.morph_encoder(patch_tensor)
            return emb.view(-1)
        if self.morphology_encoder_type == 'vit':
            inputs = self.feature_processor(images=patch_pil, return_tensors='pt').to(self.device)
            outputs = self.feature_extractor(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze(0)
        patch_tensor = self._uni2_transform(patch_pil).unsqueeze(0).to(self.device)
        emb = self.morph_encoder(patch_tensor)
        return emb.squeeze(0)

    @torch.no_grad()
    def extract_ihc_features(self, ihc_pil, masks):
        ihc_np = np.array(ihc_pil.convert('RGB'))
        ihc = torch.from_numpy(ihc_np).float().to(self.device) / 255.0
        dab_od = dab_od_from_rgb(ihc)
        masks_tensor = torch.from_numpy(masks).to(self.device)
        half = self.patch_size // 2
        height, width = dab_od.shape

        features = []
        for node_id in range(1, int(masks.max()) + 1):
            rows, cols = np.where(masks == node_id)
            if rows.size == 0 or cols.size == 0:
                continue
            if self.ihc_region == 'mask':
                cell_mask = masks_tensor == node_id
                if not torch.any(cell_mask):
                    continue
                dab_values = dab_od[cell_mask]
            else:
                cy, cx = int(np.mean(rows)), int(np.mean(cols))
                y0 = max(0, cy - half)
                y1 = min(height, cy + half + 1)
                x0 = max(0, cx - half)
                x1 = min(width, cx + half + 1)
                dab_values = dab_od[y0:y1, x0:x1].reshape(-1)
            std = dab_values.std()
            if torch.isnan(std):
                std = torch.tensor(0.0, device=self.device)
            features.append(torch.stack([
                dab_values.mean(),
                std,
                (dab_values > self.positive_threshold).float().mean(),
            ]))

        if not features:
            return torch.empty(0, 3, device=self.device)
        features = torch.stack(features)
        if self.zscore_ihc:
            features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)
        return features

    @torch.no_grad()
    def separate_stains(self, rgb_tensor):
        rgb_tensor = torch.clamp(rgb_tensor, min=1e-6, max=1.0)
        od = -torch.log(rgb_tensor)
        stains = torch.matmul(od.view(-1, 3), self.hed_from_rgb.to(self.device))
        return stains.view(rgb_tensor.shape)

    def build_graph_topology(self, masks):
        num_nodes = int(masks.max())
        centroids = np.zeros((num_nodes, 2), dtype=np.float32)
        for node_id in range(1, num_nodes + 1):
            rows, cols = np.where(masks == node_id)
            if rows.size > 0 and cols.size > 0:
                centroids[node_id - 1] = [np.mean(rows), np.mean(cols)]

        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long), centroids
        if self.graph_construction_method == 'delaunay':
            edge_index = self.delaunay_edges(centroids)
        else:
            edge_index = self.knn_edges(centroids, min(self.knn_k, num_nodes - 1))

        if self.max_edge_distance is not None and edge_index.numel() > 0:
            c = torch.from_numpy(centroids)
            dist = torch.norm(c[edge_index[0]] - c[edge_index[1]], dim=1)
            edge_index = edge_index[:, dist <= float(self.max_edge_distance)]
        return edge_index, centroids

    @staticmethod
    def knn_edges(centroids, k):
        c = torch.from_numpy(centroids).float()
        dist = torch.cdist(c, c)
        idx = dist.topk(k + 1, largest=False, dim=1).indices[:, 1:]
        src = torch.arange(c.size(0)).view(-1, 1).expand(-1, k).reshape(-1)
        dst = idx.reshape(-1)
        return torch.stack([src, dst], dim=0).long()

    @staticmethod
    def delaunay_edges(centroids):
        if centroids.shape[0] < 3:
            return torch.empty((2, 0), dtype=torch.long)
        try:
            from scipy.spatial import Delaunay
        except ImportError as err:
            raise ImportError("graph_construction_method='delaunay' requires scipy.") from err
        tri = Delaunay(centroids)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = sorted((int(simplex[i]), int(simplex[j])))
                    edges.add((u, v))
                    edges.add((v, u))
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    @staticmethod
    def edge_weights(edge_index, x_he):
        if edge_index.numel() == 0:
            return torch.empty((0, 1))
        x_norm = torch.nn.functional.normalize(x_he.detach().cpu(), p=2, dim=1)
        src, dst = edge_index[0], edge_index[1]
        return torch.sum(x_norm[src] * x_norm[dst], dim=1, keepdim=True)

    @staticmethod
    def dense_adjacency(edge_index, edge_attr, num_nodes):
        adj = torch.eye(num_nodes, dtype=torch.float32)
        if edge_index.numel() == 0:
            return adj
        weights = torch.clamp(edge_attr.detach().cpu().view(-1), min=0.0) + 1e-3
        adj[edge_index[0], edge_index[1]] = weights
        adj[edge_index[1], edge_index[0]] = weights
        return adj / torch.clamp(adj.sum(dim=1, keepdim=True), min=1e-6)


def image_pairs(dataroot, phase):
    from data.image_folder import make_dataset
    dir_a = os.path.join(dataroot, phase + 'A')
    dir_b = os.path.join(dataroot, phase + 'B')
    a_paths = sorted(make_dataset(dir_a, float('inf')))
    b_paths = sorted(make_dataset(dir_b, float('inf')))
    b_by_name = {os.path.splitext(os.path.basename(path))[0]: path for path in b_paths}
    pairs = []
    for idx, a_path in enumerate(a_paths):
        base = os.path.splitext(os.path.basename(a_path))[0]
        b_path = b_by_name.get(base, b_paths[idx % len(b_paths)] if b_paths else None)
        if b_path is not None:
            pairs.append((base, a_path, b_path))
    return pairs


def build_graphs(args):
    constructor = GraphConstructor.from_cli_args(args)
    graph_root = os.path.join(args.dataroot, args.graph_processed_dir)
    out_dir = os.path.join(graph_root, args.phase)
    if args.graph_save_mode == 'files':
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(graph_root, exist_ok=True)
    pairs = image_pairs(args.dataroot, args.phase)
    if args.graph_save_mode == 'single':
        out_path = os.path.join(graph_root, args.phase + '.pt')
        print('building %d graphs into %s' % (len(pairs), out_path))
        if os.path.exists(out_path) and not args.overwrite:
            print('graph cache already exists: %s' % out_path)
            return
        cache = {'graphs': {}, 'order': [], 'phase': args.phase}
        for idx, (base, he_path, ihc_path) in enumerate(pairs):
            graph = build_one_graph(constructor, base, he_path, ihc_path, idx, len(pairs))
            if graph is None:
                continue
            cache['graphs'][base] = graph
            cache['order'].append(base)
        torch.save(cache, out_path)
        print('saved graph cache %s graphs=%d' % (out_path, len(cache['order'])))
        return

    print('building %d graphs into %s' % (len(pairs), out_dir))
    for idx, (base, he_path, ihc_path) in enumerate(pairs):
        out_path = os.path.join(out_dir, base + '.pt')
        if os.path.exists(out_path) and not args.overwrite:
            continue
        graph = build_one_graph(constructor, base, he_path, ihc_path, idx, len(pairs))
        if graph is not None:
            torch.save(graph, out_path)
            print('[%d/%d] saved %s nodes=%d' % (idx + 1, len(pairs), out_path, graph.num_nodes))


def build_one_graph(constructor, base, he_path, ihc_path, idx, total):
    try:
        he = Image.open(he_path).convert('RGB')
        ihc = Image.open(ihc_path).convert('RGB')
        graph = constructor(he, ihc)
        if graph is None:
            print('[%d/%d] skipped %s: no graph' % (idx + 1, total, base))
            return None
        print('[%d/%d] built %s nodes=%d' % (idx + 1, total, base, graph.num_nodes))
        return graph
    except Exception as err:
        print('[%d/%d] failed %s: %s' % (idx + 1, total, base, err))
        return None


def main():
    parser = argparse.ArgumentParser(description='Precompute GSGStain cell graphs.')
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--phase', default='train')
    parser.add_argument('--graph_processed_dir', default='processedcut')
    parser.add_argument('--graph_save_mode', default='single', choices=['single', 'files'])
    parser.add_argument('--overwrite', action='store_true')
    GraphConstructor.add_cli_args(parser)
    args = parser.parse_args()
    build_graphs(args)


if __name__ == '__main__':
    main()
