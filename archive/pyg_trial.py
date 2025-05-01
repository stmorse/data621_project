import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    """
    Expects two files:
      - locations.pt: (samples, time_steps, 2, objects)
      - velocities.pt: (samples, time_steps, 2, objects)
      
    Concatenates along the 3rd axis to produce (samples, time_steps, 4, objects).
    A training pair is built from each sample: state at time t (x) and state at time t+1 (y).
    Each returned state is transposed to shape (objects, 4).
    """
    def __init__(self, loc_filepath, vel_filepath):
        locs = torch.load(loc_filepath)   # (S, T, 2, N)
        vels = torch.load(vel_filepath)     # (S, T, 2, N)
        # Concatenate along the dimension axis (axis=2) to yield (S, T, 4, N)
        data = torch.cat([locs, vels], dim=2)  
        self.data = data
        self.indices = []
        S, T, _, _ = data.shape
        # Build pairs from t and t+1 for every sample
        for s in range(S):
            for t in range(T - 1):
                self.indices.append((s, t))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        s, t = self.indices[idx]
        # Get states: shape (4, N); then permute to (N, 4)
        x = self.data[s, t].permute(1, 0).float()
        y = self.data[s, t + 1].permute(1, 0).float()
        return x, y

# (Re)use the previous GraphAutoEncoder definition:
class Encoder(nn.Module):
    def __init__(self, in_dim, node_hidden, node_embed, edge_hidden):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, node_hidden),
            nn.ReLU(),
            nn.Linear(node_hidden, node_embed)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_embed, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1)
        )

    def forward(self, x):
        n = x.size(0)
        z = self.node_mlp(x)  # [n, node_embed]

        # Build complete graph (no self-loops)
        idx = torch.arange(n, device=x.device)
        i_idx = idx.repeat_interleave(n)
        j_idx = idx.repeat(n)
        mask = i_idx != j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        edge_index = torch.stack([i_idx, j_idx], dim=0)

        zi = z[i_idx]
        zj = z[j_idx]
        edge_input = torch.cat([zi, zj], dim=1)
        edge_logits = self.edge_mlp(edge_input).squeeze(-1)
        return edge_index, edge_logits

class Decoder(nn.Module):
    def __init__(self, in_dim, msg_hidden, msg_dim, upd_hidden, out_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, msg_hidden),
            nn.ReLU(),
            nn.Linear(msg_hidden, msg_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + msg_dim, upd_hidden),
            nn.ReLU(),
            nn.Linear(upd_hidden, out_dim)
        )

    def forward(self, x, edge_index, edge_logits):
        n = x.size(0)
        edge_weights = torch.sigmoid(edge_logits)
        j_features = x[edge_index[1]]
        messages = self.msg_mlp(j_features) * edge_weights.unsqueeze(1)
        aggregated = torch.zeros(n, messages.size(1), device=x.device)
        aggregated = aggregated.index_add(0, edge_index[0], messages)
        x_cat = torch.cat([x, aggregated], dim=1)
        return self.update_mlp(x_cat)

class GraphAutoEncoder(nn.Module):
    def __init__(self, in_dim, node_hidden, node_embed, edge_hidden,
                 msg_hidden, msg_dim, upd_hidden, out_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, node_hidden, node_embed, edge_hidden)
        self.decoder = Decoder(in_dim, msg_hidden, msg_dim, upd_hidden, out_dim)

    def forward(self, x):
        edge_index, edge_logits = self.encoder(x)
        x_next = self.decoder(x, edge_index, edge_logits)
        return x_next, edge_index, edge_logits

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        batch_loss = 0.0
        B = batch_x.size(0)
        for i in range(B):
            x = batch_x[i].to(device)  # (objects, 4)
            y_true = batch_y[i].to(device)
            y_pred, _, _ = model(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            batch_loss += loss.item()
            total_samples += 1
        optimizer.step()
        epoch_loss += batch_loss
    return epoch_loss / total_samples if total_samples else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = 4  # 2 for position and 2 for velocity after concatenation.
    model = GraphAutoEncoder(
        in_dim=in_dim,
        node_hidden=16,
        node_embed=8,
        edge_hidden=16,
        msg_hidden=16,
        msg_dim=8,
        upd_hidden=16,
        out_dim=in_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # File paths for your data.
    loc_filepath = 'data/mini/loc_'
    vel_filepath = 'velocities.pt'
    dataset = TrajectoryDataset(loc_filepath, vel_filepath)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

if __name__ == '__main__':
    main()
