import torch
from torch import nn
import torch.nn.functional as F
from .stain_utils import dab_semantic_map

try:
    from torch_geometric.nn import PNAConv
except ImportError:
    PNAConv = None


class GraphSemanticLayer(nn.Module):
    """A compact PNA-style graph layer used as an optional fallback.

    It aggregates neighbour mean, max, min, and standard deviation before updating each
    node. This keeps the implementation lightweight while preserving the paper's
    idea of graph-context reasoning over cell-level staining semantics.
    """

    def __init__(self, dim):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        mean = torch.bmm(adj, x)
        second = torch.bmm(adj, x * x)
        std = torch.sqrt(torch.clamp(second - mean * mean, min=1e-6))
        max_val = self.masked_reduce(x, adj, reduce='max')
        min_val = self.masked_reduce(x, adj, reduce='min')
        out = self.update(torch.cat([x, mean, max_val, min_val, std], dim=2))
        return self.norm(x + out)

    @staticmethod
    def masked_reduce(x, adj, reduce):
        mask = adj > 0
        values = x.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        if reduce == 'max':
            values = values.masked_fill(~mask.unsqueeze(3), -1e9)
            reduced = values.max(dim=2)[0]
        else:
            values = values.masked_fill(~mask.unsqueeze(3), 1e9)
            reduced = values.min(dim=2)[0]
        has_neighbor = mask.any(dim=2, keepdim=True)
        return torch.where(has_neighbor, reduced, torch.zeros_like(reduced))


class PyGPNAConvLayer(nn.Module):
    """PNAConv wrapper for padded batched cell graphs."""

    def __init__(self, dim, k_neighbors=8, degree=9):
        super().__init__()
        if PNAConv is None:
            raise ImportError("gsrm_conv='pna' requires torch_geometric. Use --gsrm_conv custom to use the built-in fallback.")
        expected_degree = max(int(degree), int(k_neighbors) + 1, 1)
        deg = torch.zeros(expected_degree + 1, dtype=torch.long)
        deg[expected_degree] = 1
        self.register_buffer('deg', deg)
        self.conv = PNAConv(
            in_channels=dim,
            out_channels=dim,
            aggregators=['mean', 'min', 'max', 'std'],
            scalers=['identity', 'amplification', 'attenuation'],
            deg=deg,
            edge_dim=1,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj, node_mask=None):
        edge_index, edge_attr = self.adjacency_to_edges(adj, node_mask)
        if edge_index.numel() == 0:
            return x
        out = self.conv(x.reshape(-1, x.size(2)), edge_index, edge_attr=edge_attr)
        out = out.view_as(x)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(2)
        return self.norm(x + out)

    @staticmethod
    def adjacency_to_edges(adj, node_mask=None):
        b, n, _ = adj.shape
        valid = adj > 0
        if node_mask is not None:
            valid = valid & node_mask.bool().unsqueeze(2) & node_mask.bool().unsqueeze(1)
        batch_idx, src, dst = valid.nonzero(as_tuple=True)
        offsets = batch_idx * n
        edge_index = torch.stack([src + offsets, dst + offsets], dim=0)
        edge_attr = adj[batch_idx, src, dst].unsqueeze(1)
        return edge_index, edge_attr


class GraphSemanticRectificationModule(nn.Module):
    """Graph semantic rectification for weakly paired H&E-to-IHC staining.

    The original GSGStain paper builds cell graphs with Cellpose and UNI2-h.
    This module provides the same training interface with a differentiable
    grid-node fallback, so the model can train inside the CUT codebase without
    mandatory external pathology toolchains. Precomputed cell features can be
    added later by replacing the feature construction methods.
    """

    def __init__(self, grid_size=8, hidden_dim=128, he_dim=2048, num_layers=4,
                 k_neighbors=8, pyramid_levels=3, conv_type='pna', pna_degree=9):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.he_dim = he_dim
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        self.pyramid_levels = pyramid_levels
        self.conv_type = conv_type

        self.he_proj = nn.Sequential(
            nn.Linear(he_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_proj = nn.Linear(2, hidden_dim)
        self.ihc_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if conv_type == 'pna':
            self.layers = nn.ModuleList([PyGPNAConvLayer(hidden_dim, k_neighbors, pna_degree) for _ in range(num_layers)])
        elif conv_type == 'custom':
            self.layers = nn.ModuleList([GraphSemanticLayer(hidden_dim) for _ in range(num_layers)])
        else:
            raise ValueError("Unsupported gsrm conv type: %s" % conv_type)
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, real_h, real_i, fake_i=None, graph_data=None):
        if graph_data is None:
            h_context = self.context_features(real_h)
            target_sem = self.stain_features(real_i)
            coords = self.grid_coordinates(real_h.device, real_h.size(0))
            node_mask = None
            graph_mode = 'grid_fallback'
            adj = self.graph_adjacency(coords, h_context)
        else:
            h_context, target_sem, coords, node_mask, adj = self.precomputed_features(graph_data)
            graph_mode = 'cell_graph'
            if adj is None:
                adj = self.graph_adjacency(coords, h_context)

        h_context = self.fit_he_dim(h_context)
        target_sem = self.fit_semantic_dim(target_sem)
        x = self.he_proj(h_context) + self.coord_proj(coords) + self.ihc_proj(target_sem)

        for layer in self.layers:
            if self.conv_type == 'pna':
                x = layer(x, adj, node_mask)
            else:
                x = layer(x, adj)

        rectified_sem = self.semantic_head(x)
        losses = self.rectification_losses(rectified_sem, target_sem, node_mask, coords)
        output = {
            'rectified_semantics': rectified_sem,
            'target_semantics': target_sem,
            'context_features': h_context,
            'adjacency': adj,
            'graph_mode': graph_mode,
            'node_coords': coords,
            'node_mask': node_mask,
            'loss_HPC': losses[0],
            'loss_IRC': losses[1],
            'loss_GSRM': losses[0] + losses[1],
        }
        if fake_i is not None:
            if graph_data is None:
                output['fake_semantics'] = self.stain_features(fake_i)
            else:
                output['fake_semantics'] = self.stain_features_at_coords(fake_i, coords)
        return output

    def context_features(self, image):
        image = self.to_unit(image)
        pooled = F.adaptive_avg_pool2d(image, (self.grid_size, self.grid_size))
        return self.nodes_from_map(pooled)

    def stain_features(self, image):
        image = self.to_unit(image)
        sem_map = dab_semantic_map(image)
        dab_od = sem_map[:, 0:1]
        dab_score = sem_map[:, 2:3]
        mean = F.adaptive_avg_pool2d(dab_od, (self.grid_size, self.grid_size))
        mean_sq = F.adaptive_avg_pool2d(dab_od * dab_od, (self.grid_size, self.grid_size))
        std = torch.sqrt(torch.clamp(mean_sq - mean * mean, min=1e-6))
        positive = F.adaptive_avg_pool2d(dab_score, (self.grid_size, self.grid_size))
        feat = torch.cat([mean, std, positive], dim=1)
        return self.nodes_from_map(feat)

    def graph_adjacency(self, coords, context):
        b, n, _ = coords.shape
        k = min(self.k_neighbors + 1, n)
        delta = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = torch.sqrt(torch.clamp((delta * delta).sum(dim=3), min=1e-12))
        idx = dist.topk(k, dim=-1, largest=False).indices
        mask = torch.zeros(b, n, n, device=coords.device, dtype=context.dtype)
        mask.scatter_(2, idx, 1.0)

        context_n = F.normalize(context, dim=2)
        sim = torch.clamp(torch.bmm(context_n, context_n.transpose(1, 2)), min=0.0)
        weights = mask * (sim + 1e-3)
        eye = torch.eye(n, device=coords.device, dtype=context.dtype).unsqueeze(0)
        weights = weights + eye
        return weights / torch.clamp(weights.sum(dim=2, keepdim=True), min=1e-6)

    def precomputed_features(self, graph_data):
        x_he = graph_data['x_he']
        x_ihc = graph_data['x_ihc']
        coords = graph_data['coords'] if 'coords' in graph_data else graph_data['centroids']
        node_mask = graph_data['mask'] if 'mask' in graph_data else torch.ones(x_he.shape[:2], device=x_he.device, dtype=x_he.dtype)
        adj = graph_data['adjacency'] if 'adjacency' in graph_data else None

        if x_ihc.size(2) >= 3:
            target_sem = x_ihc[:, :, :3]
        else:
            target_sem = F.pad(x_ihc, (0, 3 - x_ihc.size(2)))
        target_sem = torch.clamp(target_sem, 0.0, 1.0)
        coords = torch.clamp(coords[:, :, :2], 0.0, 1.0)
        return x_he, target_sem, coords, node_mask, adj

    def stain_features_at_coords(self, image, coords):
        sem_map = self.stain_feature_map(image)
        grid = coords * 2.0 - 1.0
        grid = grid.view(grid.size(0), 1, grid.size(1), 2)
        sampled = F.grid_sample(sem_map, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return sampled.squeeze(2).permute(0, 2, 1).contiguous()

    def stain_feature_map(self, image):
        image = self.to_unit(image)
        return dab_semantic_map(image)

    def rectification_losses(self, pred, target, node_mask=None, coords=None):
        if node_mask is not None:
            return self.masked_rectification_losses(pred, target, node_mask, coords)

        pred_map = self.to_map(pred)
        target_map = self.to_map(target)
        loss_hpc = 0.0
        weight = 1.0
        for _ in range(self.pyramid_levels):
            loss_hpc = loss_hpc + weight * F.mse_loss(pred_map, target_map)
            if pred_map.size(2) == 1 or pred_map.size(3) == 1:
                break
            pred_map = F.avg_pool2d(pred_map, kernel_size=2, stride=2)
            target_map = F.avg_pool2d(target_map, kernel_size=2, stride=2)
            weight *= 0.5

        pred_fp = F.normalize(pred.contiguous().view(pred.size(0), -1), dim=1)
        target_fp = F.normalize(target.contiguous().view(target.size(0), -1), dim=1)
        pred_rel = torch.mm(pred_fp, pred_fp.t())
        target_rel = torch.mm(target_fp, target_fp.t())
        loss_irc = torch.mean(torch.abs(pred_rel - target_rel))
        return loss_hpc, loss_irc

    def masked_rectification_losses(self, pred, target, node_mask, coords=None):
        mask = node_mask.unsqueeze(2)
        if coords is None:
            denom = torch.clamp(mask.sum() * pred.size(2), min=1.0)
            loss_hpc = (((pred - target) ** 2) * mask).sum() / denom
        else:
            loss_hpc = self.node_grid_pyramid_loss(pred, target, coords, node_mask)

        pred_fp = self.masked_fingerprint(pred, node_mask)
        target_fp = self.masked_fingerprint(target, node_mask)
        pred_rel = torch.mm(pred_fp, pred_fp.t())
        target_rel = torch.mm(target_fp, target_fp.t())
        loss_irc = torch.mean(torch.abs(pred_rel - target_rel))
        return loss_hpc, loss_irc

    def node_grid_pyramid_loss(self, pred, target, coords, node_mask):
        pred_map, valid_map = self.nodes_to_grid(pred, coords, node_mask)
        target_map, _ = self.nodes_to_grid(target, coords, node_mask)
        loss = 0.0
        weight = 1.0
        for _ in range(self.pyramid_levels):
            denom = torch.clamp(valid_map.sum() * pred_map.size(1), min=1.0)
            loss = loss + weight * ((((pred_map - target_map) ** 2) * valid_map).sum() / denom)
            if pred_map.size(2) == 1 or pred_map.size(3) == 1:
                break
            current_valid = valid_map
            pred_map, valid_map = self.downsample_valid_map(pred_map, current_valid)
            target_map, _ = self.downsample_valid_map(target_map, current_valid)
            weight *= 0.5
        return loss

    def nodes_to_grid(self, features, coords, node_mask):
        b, n, c = features.shape
        k = self.grid_size
        coords = torch.clamp(coords[:, :, :2], 0.0, 1.0)
        gx = torch.clamp((coords[:, :, 0] * k).long(), 0, k - 1)
        gy = torch.clamp((coords[:, :, 1] * k).long(), 0, k - 1)
        index = gy * k + gx
        mask = node_mask.unsqueeze(2).to(features.dtype)

        feat_sum = features.new_zeros(b, k * k, c)
        count = features.new_zeros(b, k * k, 1)
        feat_sum.scatter_add_(1, index.unsqueeze(2).expand(-1, -1, c), features * mask)
        count.scatter_add_(1, index.unsqueeze(2), mask)
        grid = feat_sum / torch.clamp(count, min=1.0)
        valid = (count > 0).to(features.dtype)
        grid = grid.view(b, k, k, c).permute(0, 3, 1, 2).contiguous()
        valid = valid.view(b, k, k, 1).permute(0, 3, 1, 2).contiguous()
        return grid, valid

    @staticmethod
    def downsample_valid_map(value_map, valid_map):
        count = F.avg_pool2d(valid_map, kernel_size=2, stride=2) * 4.0
        value_sum = F.avg_pool2d(value_map * valid_map, kernel_size=2, stride=2) * 4.0
        valid = (count > 0).to(value_map.dtype)
        value = value_sum / torch.clamp(count, min=1.0)
        return value, valid

    @staticmethod
    def masked_fingerprint(features, node_mask):
        mask = node_mask.unsqueeze(2)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        mean = (features * mask).sum(dim=1) / denom
        return F.normalize(mean, dim=1)

    def graph_semantic_consistency(self, fake_semantics, rectified_semantics, node_mask=None):
        if node_mask is None:
            return F.mse_loss(fake_semantics, rectified_semantics.detach())
        mask = node_mask.unsqueeze(2)
        denom = torch.clamp(mask.sum() * fake_semantics.size(2), min=1.0)
        return (((fake_semantics - rectified_semantics.detach()) ** 2) * mask).sum() / denom

    def to_map(self, semantics):
        b = semantics.size(0)
        return semantics.view(b, self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)

    @staticmethod
    def nodes_from_map(feat):
        return feat.permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1, feat.size(1))

    def fit_he_dim(self, x):
        if x.size(2) == self.he_dim:
            return x
        if x.size(2) > self.he_dim:
            return x[:, :, :self.he_dim]
        return F.pad(x, (0, self.he_dim - x.size(2)))

    @staticmethod
    def fit_semantic_dim(x):
        if x.size(2) == 3:
            return x
        if x.size(2) > 3:
            return x[:, :, :3]
        return F.pad(x, (0, 3 - x.size(2)))

    def grid_coordinates(self, device, batch_size):
        line = torch.linspace(0, 1, self.grid_size, device=device)
        yy, xx = torch.meshgrid(line, line, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).view(1, -1, 2)
        return coords.expand(batch_size, -1, -1)

    @staticmethod
    def to_unit(image):
        return torch.clamp((image + 1.0) * 0.5, 0.0, 1.0)
