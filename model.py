import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from graph import mark_as_pipeline_stage, mark_as_ghost_stage


class CustomGraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super(CustomGraphSAGEModel, self).__init__()
        assert num_layers >= 1, "num_layers must be at least 1"

        # print(f'in_channels: {in_channels}, hidden_channels: {hidden_channels}, out_channels: {out_channels}, num_layers: {num_layers}')

        self.num_layers = num_layers

        # Initialize the first convolutional chain
        for i in range(num_layers):
            if i == 0:
                setattr(
                    self,
                    f"chain1_conv_{i}",
                    mark_as_pipeline_stage(SAGEConv(in_channels, hidden_channels)),
                )
            else:
                setattr(
                    self,
                    f"chain1_conv_{i}",
                    mark_as_pipeline_stage(SAGEConv(hidden_channels, hidden_channels)),
                )
                setattr(
                    self,
                    f"chain1_residual_{i-1}",
                    mark_as_ghost_stage(nn.Linear(hidden_channels, hidden_channels)),
                )

        # Initialize the second convolutional chain
        for i in range(num_layers):
            if i == 0:
                setattr(
                    self,
                    f"chain2_conv_{i}",
                    mark_as_pipeline_stage(SAGEConv(in_channels, hidden_channels)),
                )
            else:
                setattr(
                    self,
                    f"chain2_conv_{i}",
                    mark_as_pipeline_stage(SAGEConv(hidden_channels, hidden_channels)),
                )
                setattr(
                    self,
                    f"chain2_residual_{i-1}",
                    mark_as_ghost_stage(nn.Linear(hidden_channels, hidden_channels)),
                )

        # Final convolutional layer to combine both chains
        self.final_conv = mark_as_pipeline_stage(
            SAGEConv(2 * hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # Forward pass through the first chain
        x1 = x
        for i in range(self.num_layers):
            conv = getattr(self, f"chain1_conv_{i}")
            if i > 0:
                residual = getattr(self, f"chain1_residual_{i-1}")(x1)
            else:
                residual = 0
            x1 = conv(x1, edge_index) + residual
            if i < self.num_layers - 1:
                x1 = F.relu(x1)
                x1 = F.dropout(x1, p=0.5, training=self.training)

        # Forward pass through the second chain
        x2 = x
        for i in range(self.num_layers):
            conv = getattr(self, f"chain2_conv_{i}")
            if i > 0:
                residual = getattr(self, f"chain2_residual_{i-1}")(x2)
            else:
                residual = 0
            x2 = conv(x2, edge_index) + residual
            if i < self.num_layers - 1:
                x2 = F.relu(x2)
                x2 = F.dropout(x2, p=0.5, training=self.training)

        combined = torch.cat([x1, x2], dim=1)

        # Final pass through the combined layers
        out = self.final_conv(combined, edge_index)

        return F.log_softmax(out, dim=1)


class CustomGraphLinearModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGraphLinearModel, self).__init__()

        # First chain layers
        self.chain1_linear_0 = mark_as_pipeline_stage(
            nn.Linear(in_channels, hidden_channels)
        )
        self.chain1_linear_1 = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.chain1_linear_2 = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.chain1_linear_3 = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Second chain layers
        self.chain2_linear_0 = mark_as_pipeline_stage(
            nn.Linear(in_channels, hidden_channels)
        )
        self.chain2_linear_1 = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.chain2_linear_2 = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.chain2_linear_3 = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Final layer to combine both chains
        self.final_linear = mark_as_pipeline_stage(
            nn.Linear(2 * hidden_channels, out_channels)
        )

    def forward(self, x):
        # First chain forward pass
        x1 = F.relu(self.chain1_linear_0(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = F.relu(self.chain1_linear_1(x1))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = F.relu(self.chain1_linear_2(x1))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.chain1_linear_3(x1)  # No activation or dropout after final layer

        # Second chain forward pass
        x2 = F.relu(self.chain2_linear_0(x))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = F.relu(self.chain2_linear_1(x2))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = F.relu(self.chain2_linear_2(x2))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.chain2_linear_3(x2)  # No activation or dropout after final layer

        # Combine both chains
        combined = torch.cat([x1, x2], dim=1)

        # Final linear layer
        out = self.final_linear(combined)
        return F.log_softmax(out, dim=1)


class CustomGraphLinearModel2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGraphLinearModel2, self).__init__()

        # First chain layers
        self.chain1_linear_0 = mark_as_pipeline_stage(
            nn.Linear(in_channels, hidden_channels)
        )
        self.chain1_linear_1 = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.chain1_linear_2 = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, hidden_channels)
        )

        # self.chain2_linear_0 = mark_as_pipeline_stage(nn.Linear(in_channels, hidden_channels))
        # self.chain2_linear_1 = mark_as_pipeline_stage(nn.Linear(hidden_channels, hidden_channels))
        # self.chain2_linear_2 = mark_as_pipeline_stage(nn.Linear(hidden_channels, hidden_channels))

        self.final_linear = mark_as_pipeline_stage(
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        # First chain forward pass
        x1 = F.relu(self.chain1_linear_0(x))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = F.relu(self.chain1_linear_1(x1))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = F.relu(self.chain1_linear_2(x1))

        # x2 = F.relu(self.chain2_linear_0(x))
        # x2 = F.dropout(x2, p=0.5, training=self.training)
        # x2 = F.relu(self.chain2_linear_1(x2))
        # x2 = F.dropout(x2, p=0.5, training=self.training)
        # x2 = F.relu(self.chain2_linear_2(x2))

        # combined = torch.cat([x1, x2], dim=1)
        x3 = F.relu(self.final_linear(x1))

        return F.log_softmax(x3, dim=1)
