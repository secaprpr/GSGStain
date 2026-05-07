import os.path
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import util.util as util


class GSGStainDataset(BaseDataset):
    """Weakly paired consecutive-section dataset for GSGStain.

    The dataset expects H&E images in trainA/testA and reference IHC images in
    trainB/testB. By default it pairs sorted filenames by index and applies the
    same crop/flip transform to both images, preserving coarse spatial
    correspondence for graph-semantic supervision.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--gsg_pair_mode', type=str, default='sorted', choices=['sorted', 'random'],
                            help='use sorted weak pairs or random B-domain samples')
        parser.add_argument('--graph_processed_dir', type=str, default='processedcut',
                            help='optional directory under dataroot containing precomputed cell graphs')
        parser.add_argument('--graph_max_nodes', type=int, default=512,
                            help='pad or truncate cached cell graphs to this number of nodes')
        parser.add_argument('--graph_filter_missing', action='store_true',
                            help='drop A-domain images without cached graphs; enabled automatically for gsg_stage=train_gsrm')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.processed_root = os.path.join(opt.dataroot, opt.graph_processed_dir)
        self.processed_dir = os.path.join(opt.dataroot, opt.graph_processed_dir, opt.phase)

        if opt.phase == 'test' and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, 'valA')):
            self.dir_A = os.path.join(opt.dataroot, 'valA')
            self.dir_B = os.path.join(opt.dataroot, 'valB')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.B_paths_by_name = {os.path.splitext(os.path.basename(path))[0]: path for path in self.B_paths}
        self.graph_cache = None
        self.graph_paths = {}
        single_cache_path = os.path.join(self.processed_root, opt.phase + '.pt')
        if os.path.isfile(single_cache_path):
            try:
                self.graph_cache = torch.load(single_cache_path, map_location='cpu', weights_only=False)
                print('loaded graph cache %s' % single_cache_path)
            except Exception as err:
                print('Warning: failed to load graph cache %s: %s' % (single_cache_path, err))
                self.graph_cache = None
        elif os.path.isdir(self.processed_dir):
            for graph_name in sorted(os.listdir(self.processed_dir)):
                if graph_name.endswith('.pt'):
                    base_name = os.path.splitext(graph_name)[0]
                    self.graph_paths[base_name] = os.path.join(self.processed_dir, graph_name)
        self.filter_missing_graphs = opt.graph_filter_missing or getattr(opt, 'gsg_stage', None) == 'train_gsrm'
        if self.filter_missing_graphs:
            self.filter_paths_with_graphs()
        if self.A_size == 0 or self.B_size == 0:
            raise RuntimeError('GSGStainDataset requires images in both %s and %s' % (self.dir_A, self.dir_B))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.gsg_pair_mode == 'sorted':
            base_name = os.path.splitext(os.path.basename(A_path))[0]
            B_path = self.B_paths_by_name.get(base_name)
            if B_path is None:
                index_B = index % self.B_size
                B_path = self.B_paths[index_B]
        else:
            index_B = self.random_index_B()
            B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        params = get_params(modified_opt, A_img.size)
        transform = get_transform(modified_opt, params)
        A = transform(A_img)
        B = transform(B_img)

        data = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        graph_data = self.load_graph(A_path)
        data.update(graph_data)
        return data

    def __len__(self):
        return self.A_size

    def random_index_B(self):
        import random
        return random.randint(0, self.B_size - 1)

    def load_graph(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        graph = self.get_cached_graph(base_name)
        if graph is None:
            return self.empty_graph()

        if not all(self.graph_has(graph, key) for key in ['x_he', 'x_ihc', 'centroids']):
            print('Warning: graph %s misses x_he, x_ihc, or centroids' % base_name)
            return self.empty_graph()

        x_he = self.graph_get(graph, 'x_he').float()
        x_ihc = self.graph_get(graph, 'x_ihc').float()
        coords = self.load_coords(graph)

        max_nodes = self.opt.graph_max_nodes
        num_nodes = min(x_he.size(0), max_nodes)
        mask = torch.zeros(max_nodes, dtype=torch.float32)
        he_pad = torch.zeros(max_nodes, x_he.size(1), dtype=torch.float32)
        ihc_pad = torch.zeros(max_nodes, x_ihc.size(1), dtype=torch.float32)
        coord_pad = torch.zeros(max_nodes, 2, dtype=torch.float32)
        adj_pad = torch.zeros(max_nodes, max_nodes, dtype=torch.float32)

        if num_nodes > 0:
            mask[:num_nodes] = 1.0
            he_pad[:num_nodes] = x_he[:num_nodes]
            ihc_pad[:num_nodes] = x_ihc[:num_nodes]
            coord_pad[:num_nodes] = coords[:num_nodes, :2]
            adj_pad[:num_nodes, :num_nodes] = self.graph_adjacency(graph, num_nodes)

        return {
            'graph_x_he': he_pad,
            'graph_x_ihc': ihc_pad,
            'graph_centroids': coord_pad,
            'graph_coords': coord_pad,
            'graph_adj': adj_pad,
            'graph_mask': mask,
        }

    def empty_graph(self):
        max_nodes = self.opt.graph_max_nodes
        he_dim = getattr(self.opt, 'graph_he_dim', 2048)
        return {
            'graph_x_he': torch.zeros(max_nodes, he_dim, dtype=torch.float32),
            'graph_x_ihc': torch.zeros(max_nodes, 3, dtype=torch.float32),
            'graph_centroids': torch.zeros(max_nodes, 2, dtype=torch.float32),
            'graph_coords': torch.zeros(max_nodes, 2, dtype=torch.float32),
            'graph_adj': torch.zeros(max_nodes, max_nodes, dtype=torch.float32),
            'graph_mask': torch.zeros(max_nodes, dtype=torch.float32),
        }

    def filter_paths_with_graphs(self):
        kept_paths = []
        missing = 0
        for path in self.A_paths:
            base_name = os.path.splitext(os.path.basename(path))[0]
            graph = self.get_cached_graph(base_name)
            if graph is not None and all(self.graph_has(graph, key) for key in ['x_he', 'x_ihc', 'centroids']):
                kept_paths.append(path)
            else:
                missing += 1
        self.A_paths = kept_paths
        self.A_size = len(self.A_paths)
        print('GSGStain graph filter kept %d A images and skipped %d without usable graphs' % (self.A_size, missing))

    def get_cached_graph(self, base_name):
        if self.graph_cache is not None:
            if isinstance(self.graph_cache, dict) and 'graphs' in self.graph_cache:
                return self.graph_cache['graphs'].get(base_name)
            if isinstance(self.graph_cache, dict):
                return self.graph_cache.get(base_name)
            return None

        graph_path = self.graph_paths.get(base_name)
        if graph_path is None:
            return None
        try:
            return torch.load(graph_path, map_location='cpu', weights_only=False)
        except Exception as err:
            print('Warning: failed to load graph %s: %s' % (graph_path, err))
            return None

    @staticmethod
    def graph_has(graph, key):
        return key in graph if isinstance(graph, dict) else hasattr(graph, key)

    @staticmethod
    def graph_get(graph, key):
        return graph[key] if isinstance(graph, dict) else getattr(graph, key)

    def normalize_centroids(self, centroids, graph):
        if centroids.numel() == 0:
            return centroids
        if self.graph_has(graph, 'image_size'):
            image_size = self.graph_get(graph, 'image_size').float()
            height = torch.clamp(image_size[0], min=1.0)
            width = torch.clamp(image_size[1], min=1.0)
            y = centroids[:, 0] / height
            x = centroids[:, 1] / width
            return torch.stack([x, y], dim=1).clamp(0.0, 1.0)
        scale = torch.clamp(centroids.max(dim=0, keepdim=True)[0], min=1.0)
        norm = centroids / scale
        return torch.stack([norm[:, 1], norm[:, 0]], dim=1).clamp(0.0, 1.0)

    def load_coords(self, graph):
        if self.graph_has(graph, 'coords'):
            return self.graph_get(graph, 'coords').float().clamp(0.0, 1.0)
        return self.normalize_centroids(self.graph_get(graph, 'centroids').float(), graph)

    def graph_adjacency(self, graph, num_nodes):
        if self.graph_has(graph, 'adjacency'):
            adj = self.graph_get(graph, 'adjacency').float()
            return adj[:num_nodes, :num_nodes]

        adj = torch.eye(num_nodes, dtype=torch.float32)
        if not self.graph_has(graph, 'edge_index'):
            return adj
        edge_index = self.graph_get(graph, 'edge_index').long()
        if edge_index.numel() == 0:
            return adj
        edge_index = edge_index[:, (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)]
        if edge_index.numel() == 0:
            return adj
        if self.graph_has(graph, 'edge_attr'):
            edge_attr = self.graph_get(graph, 'edge_attr').float().view(-1)
            edge_attr = edge_attr[:edge_index.size(1)]
            weights = torch.clamp(edge_attr, min=0.0) + 1e-3
        else:
            weights = torch.ones(edge_index.size(1), dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = weights
        adj[edge_index[1], edge_index[0]] = weights
        return adj / torch.clamp(adj.sum(dim=1, keepdim=True), min=1e-6)
