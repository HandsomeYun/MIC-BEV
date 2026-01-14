import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import math


class BipartiteCamBEVGAT(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_cameras: int,
                 hidden: int = 128,
                 layers: int = 3,
                 heads: int = 4,
                 edge_dim: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.gat = pyg_nn.GAT(
            in_channels=in_dim,            
            hidden_channels=hidden,
            num_layers=layers,
            out_channels=num_cameras,      
            heads=heads,
            dropout=dropout,
            v2=True,                       # use GATv2Conv instead of GATConv
            act="elu",                     # ELU activation
            edge_dim=edge_dim,             # pass edge features
        )

    def forward(self, x, edge_index, edge_attr):
        # Returns shape [num_nodes, 1], so squeeze to [num_nodes]:
        return self.gat(x, edge_index, edge_attr=edge_attr).squeeze(-1)


def compute_cam_bev_weights(gat,          # BipartiteCamBEVGAT instance
                            bev_mask,     # (N_cam, B, num_bev)  bool
                            bev_query,    # (B,    num_bev, C)
                            img_feats,    # list[N_cam] each (num_tokens, B, C)
                            cam_positions, # (N_cam, 3) camera positions in world coordinates，
                            cam_yaw,
                            cam_pitch,
                            bev_grid_coords, bev_resolution, span, world_min
                            ): 
        """
        Estimate per-camera importance for each BEV cell with a bipartite GAT.

        Parameters
        ----------
        gat : nn.Module
            Instance of ``BipartiteCamBEVGAT`` returning one scalar logit per node.
        bev_mask : (N_cam, B, num_bev) bool
            Visibility mask - ``True`` if camera *c* sees BEV cell *q* in batch *b*.
            (This is the usual mask output of BEVFormer's `point_sampling`.)
        bev_query : (B, num_bev, C) float32/float16
            Current BEV embeddings (will be used as node features).
        img_feats : list[T]  with  T = (num_tokens, B, C)
            One feature map list entry per camera.  Tokens are first averaged to a
            single ``(B, C)`` vector and then used as **camera-node features.**
        cam_positions : (N_cam, 3) float32
            Camera positions in world coordinates (x, y, z).
        bev_grid_coords : (num_bev, 2) float32
            BEV grid coordinates in world coordinates (x, y).
        bev_resolution : float
            BEV grid resolution in meters.
        ego_position : (3,) float32
            Ego vehicle position in world coordinates (x, y, z).

        Returns
        -------
        cam_bev_weights : (B, num_bev, N_cam)
            Softmax-normalised importance weights (sum over cameras = 1).
            Non-visible pairs are automatically ≈0.
        cam_bev_logits  : (total_nodes,)
            Raw GAT scores for every node in the concatenated batch graph
            (useful for auxiliary losses or debugging).
        bev_feats       : (B, num_bev, C)
            Camera-weighted feature for each BEV cell, computed as
            ``(weights @ pooled_cam_feats)`` — ready to residual-add to the BEV
            query or feed into downstream heads.
        """
        # 1.  Build *camera* node features  (pooled over tokens)
        cam_feats = torch.stack(img_feats, 0)          # (N_cam, tokens, B, C)
        cam_feats = cam_feats.mean(1)                  # (N_cam, B, C)
        cam_feats = cam_feats.permute(1, 0, 2)         # (B, N_cam, C)

        B, num_bev, C = bev_query.shape
        N_cam = cam_feats.size(1)
        
        # Transform camera positions from absolute world coordinates to BEV grid coordinates
        world_min_x, world_min_y = world_min
        bev_resolution_x, bev_resolution_y = bev_resolution

        # 4. Vectorized coordinate transformation
        cam_positions_grid = cam_positions.clone()
        cam_positions_grid[:, 0] = (cam_positions[:, 0] - world_min_x) / bev_resolution_x
        cam_positions_grid[:, 1] = (cam_positions[:, 1] - world_min_y) / bev_resolution_y
        # cam_positions_grid[:, 2] remains unchanged
        
        # ---- 2.  Compute geometric features - Vectorized ---- 
        # Expand dimensions for broadcasting: (num_bev, 1, 2) - (1, N_cam, 2) = (num_bev, N_cam, 2)
        diff_cells = bev_grid_coords.unsqueeze(1) - cam_positions_grid[:, :2].unsqueeze(0)  # (num_bev, N_cam, 2)
        diff_m = diff_cells * bev_resolution.view(1,1,2)
        dxdy   = diff_m / span.view(1,1,2)
        
        # Create height tensor with proper broadcasting
        height = cam_positions_grid[:, 2].view(1, N_cam, 1).expand(num_bev, N_cam, 1)  # (num_bev, N_cam, 1)
        
        # 2-A. diff already has dx,dy   (num_bev, N_cam, 2)
        planar_dist = torch.norm(dxdy, dim=-1, keepdim=True)        # (...,1)

        # 2-B. angle wrt heading
        cam_fwd = torch.stack([torch.cos(cam_yaw), torch.sin(cam_yaw)], dim=-1)   # (N_cam,2)
        cam_fwd = cam_fwd.unsqueeze(0) 
        unit_dir = F.normalize(dxdy, dim=-1, eps=1e-6)
        cos_d = (unit_dir * cam_fwd).sum(-1, keepdim=True)          # (...,1)
        sin_d = (cam_fwd[...,0] * unit_dir[...,1]                                   # broadcast OK 
                 - cam_fwd[...,1] * unit_dir[...,0]).unsqueeze(-1)
        
        # 2-C. pitch
        sin_p = torch.sin(cam_pitch).view(1, N_cam, 1).expand_as(cos_d)
        cos_p = torch.cos(cam_pitch).view(1, N_cam, 1).expand_as(cos_d)

        geo = torch.cat([
            dxdy,                 # 2
            height,               # 1
            planar_dist,    # 1   (normalize ≈-1…1)
            cos_d, sin_d,         # 2
            sin_p, cos_p          # 2
        ], dim=-1)               # (num_bev,N_cam,9)
        
        
        # ---- 3. Assemble full node‐feature matrix ----
        bev_feats_cat = bev_query # (B, num_bev, C)
        cam_feats_cat = cam_feats # (B, N_cam, C)

        # flatten batch into one big graph:
        bev_nodes = bev_feats_cat.view(B * num_bev, -1)   # (B*num_bev, C)
        cam_nodes = cam_feats_cat.view(B * N_cam, -1)     # (B*N_cam,   C)
        x = torch.cat([bev_nodes, cam_nodes], dim=0)      # (total_nodes, C)

        # ---- 4. Build edge index and edge attributes from bev_mask ----
        num_bev_nodes = B * num_bev
        
        # Vectorized edge construction using torch.where
        # bev_mask is (N_cam, B, num_bev), we need to find all True values
        cam_indices, batch_indices, bev_indices = torch.where(bev_mask)
        
        # Compute global indices
        bev_global_indices = batch_indices * num_bev + bev_indices
        cam_global_indices = num_bev_nodes + batch_indices * N_cam + cam_indices
        
        # Create edge_index: (2, num_edges) where first row is source (cam), second is target (bev)
        edge_index = torch.stack([cam_global_indices, bev_global_indices], dim=0)
        
        # Get corresponding edge attributes
        edge_attr = geo[bev_indices, cam_indices]  # (num_edges, 9)

        # ---- 5. Run your GAT ----
        outputs = gat(x, edge_index, edge_attr=edge_attr) # (total_nodes, N_cam)

        # ---- 6. Split logits, mask & softmax to get per-(bev,cam) weights ----
        bev_camera_scores = outputs[:num_bev_nodes].view(B, num_bev, N_cam) # (B, num_bev, N_cam)

        # for each bev-cell, do a softmax over the cameras it sees:
        # expand cam_logits to (B, num_bev, N_cam) then mask & renormalize
        raw = bev_camera_scores   # (B, num_bev, N_cam)
        raw = raw.masked_fill(~bev_mask.permute(1,2,0), -1e9)  # mask out unseen pairs
        
        # Apply temperature scaling to make attention more distributed
        temperature = 2.0  # Higher temperature = more distributed attention
        cam_bev_weights = F.softmax(raw / temperature, dim=-1)               # (B, num_bev, N_cam)

        return cam_bev_weights