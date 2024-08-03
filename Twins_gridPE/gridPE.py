import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSplitPositionalEncoding:
    def __init__(self, pos_list, dim):
        Q = np.load("./data/Q_matrix.npy")
        self.dim = dim
        self.max_len = len(pos_list)
        self.grid_pe = np.zeros((self.max_len,dim//2,2))

        assert self.dim % 2 == 0

        # Initialize angle and rotation
        theta = 2 * np.pi / 3
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        random_angle_list = np.random.uniform(0, 2 * np.pi, dim)

        # Precompute the position encodings
        for k in range(dim // 2):
            random_angle = random_angle_list[k]
            scale_factor = 200 ** (k / dim)
            omega_n0 = (np.array([np.cos(random_angle), np.sin(random_angle)]) * scale_factor)
            omega_n1 = R.dot(omega_n0)
            omega_n2 = R.dot(omega_n1)

            # Compute cos and sin for all positions for this dimension
            for idx in range(self.max_len):
                coords = pos_list[idx]
                eiw0x = np.exp(1j * np.dot(omega_n0, coords))
                eiw1x = np.exp(1j * np.dot(omega_n1, coords))
                eiw2x = np.exp(1j * np.dot(omega_n2, coords))

                g = Q.dot(np.array([eiw0x, eiw1x, eiw2x]))
                g = np.sum(g, axis=0)
                self.grid_pe[idx, k, 0] = np.real(g)
                self.grid_pe[idx, k, 1] = np.imag(g)
        self.grid_pe = self.grid_pe.reshape(self.max_len, -1)
        self.grid_pe = self.grid_pe.transpose()
                
    def apply_encoding(self, x):
        batch_size, heads, num_queries, dim = x.shape
        assert (
            dim == self.dim # dim需要是偶数
        ), "Dimension of x must match dimension of positional encoding"
        assert (
            num_queries <= self.max_len # 位置编码长度超过看到的轨迹长度
        ), "Input sequence length exceeds maximum length"

        # Ensure grid_pe is a tensor, convert it if necessary
        if not isinstance(self.grid_pe, torch.Tensor):
            self.grid_pe = torch.from_numpy(
                self.grid_pe
            ).float()  # Convert to tensor and ensure type is float

        # Move grid_pe to the same device as x
        self.grid_pe = self.grid_pe.to(x.device)

        # Apply the encoding
        # Since grid_pe is [dim, num_queries], we don't need to transpose it
        # We broadcast grid_pe's second dimension to the num_queries dimension in x
        # And expand grid_pe across the batch and heads dimensions
        grid_pe_expanded = self.grid_pe.unsqueeze(0).unsqueeze(
            0
        )  # Shape: [1, 1, dim, num_queries]
        grid_pe_expanded = grid_pe_expanded.expand(
            batch_size, heads, -1, -1
        )  # Shape: [batch_size, heads, dim, num_queries]

        # Perform element-wise multiplication and sum across the dim dimension to get the result
        encoded = torch.einsum("bhqd,bhdq->bhqd", x, grid_pe_expanded)

        return encoded

class GridRotatePositionalEncoding:
    def __init__(self, pos_list, dim):
        Q = np.load("./data/Q_matrix.npy")
        self.dim = dim
        self.max_len = len(pos_list)
        self.grid_pe = np.zeros((self.max_len,dim//2,2))

        assert self.dim % 2 == 0

        # Initialize angle and rotation
        theta = 2 * np.pi / 3
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        random_angle_list = np.random.uniform(0, 2 * np.pi, dim)

        # Initialize arrays to store real and imaginary parts separately
        real_part = np.zeros((self.max_len, dim // 2))
        imag_part = np.zeros((self.max_len, dim // 2))

        # Precompute the position encodings
        for k in range(dim // 2):
            random_angle = random_angle_list[k]
            scale_factor = 200 ** (k / dim) 
            omega_n0 = (np.array([np.cos(random_angle), np.sin(random_angle)]) * scale_factor)
            omega_n1 = R.dot(omega_n0)
            omega_n2 = R.dot(omega_n1)

            # Compute cos and sin for all positions for this dimension
            for idx in range(self.max_len):
                coords = pos_list[idx]
                eiw0x = np.exp(1j * np.dot(omega_n0, coords))
                eiw1x = np.exp(1j * np.dot(omega_n1, coords))
                eiw2x = np.exp(1j * np.dot(omega_n2, coords))

                g = Q.dot(np.array([eiw0x, eiw1x, eiw2x]))
                g = np.sum(g, axis=0)
                real_part[idx, k] = np.real(g)
                imag_part[idx, k] = np.imag(g) 
        # Combine real and imaginary parts
        self.grid_pe = np.concatenate((real_part, imag_part), axis=1)
                
    def apply_encoding(self, x):
        batch_size, heads, num_queries, dim = x.shape
        assert (
            dim == self.dim 
        ), "Dimension of x must match dimension of positional encoding"
        assert (
            num_queries <= self.max_len 
        ), "Input sequence length exceeds maximum length"

        # Ensure grid_pe is a tensor, convert it if necessary
        if not isinstance(self.grid_pe, torch.Tensor):
            self.grid_pe = torch.from_numpy(
                self.grid_pe
            ).float()  # Convert to tensor and ensure type is float

        # Move grid_pe to the same device as x
        self.grid_pe = self.grid_pe.to(x.device)

        # Apply the encoding
        # Since grid_pe is [dim, num_queries], we don't need to transpose it
        # We broadcast grid_pe's second dimension to the num_queries dimension in x
        # And expand grid_pe across the batch and heads dimensions
        grid_pe_expanded = self.grid_pe.unsqueeze(0).unsqueeze(
            0
        )  # Shape: [1, 1, dim, num_queries]
        grid_pe_expanded = grid_pe_expanded.expand(
            batch_size, heads, -1, -1
        )  # Shape: [batch_size, heads, dim, len(pos_list)]

        # Apply the rotation position encoding
        x_rotated = self._rotate(x, grid_pe_expanded)

        return x_rotated
    
    def _rotate(self, x, grid_pe):
        # Split the input x and grid_pe into two parts along the last dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)
        cos, sin = torch.chunk(grid_pe, 2, dim=-1)

        # Apply rotation
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # Concatenate the rotated parts
        x_rotated = torch.cat([x1_rot, x2_rot], dim=-1)

        return x_rotated


class GridMergingPositionalEncoding:
    def __init__(self, pos_list, dim):
        Q = np.load('./data/Q_matrix.npy')
        self.dim = dim
        self.max_len = len(pos_list)
        self.grid_pe = np.zeros((dim, self.max_len))

        # Initialize angle and rotation
        theta = 2 * np.pi / 3
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        random_angle_list = np.random.uniform(0, 2 * np.pi, dim)

        # Precompute the position encodings
        for k in range(dim):
            random_angle = random_angle_list[k]
            scale_factor = 200 ** (k / dim)
            omega_n0 = np.array([np.cos(random_angle), np.sin(random_angle)]) * scale_factor
            omega_n1 = R.dot(omega_n0)
            omega_n2 = R.dot(omega_n1)

            # Compute cos and sin for all positions for this dimension 
            for idx in range(self.max_len):
                coords = pos_list[idx]
                eiw0x = np.exp(1j * np.dot(omega_n0, coords))
                eiw1x = np.exp(1j * np.dot(omega_n1, coords))
                eiw2x = np.exp(1j * np.dot(omega_n2, coords))
                
                g = Q.dot(np.array([eiw0x, eiw1x, eiw2x]))
                g = np.sum(g, axis=0)
                self.grid_pe[k, idx] = np.real(g) + np.imag(g)

class GridComplexPositionalEncoding:
    def __init__(self, pos_list, dim):
        Q = np.load("./data/Q_matrix.npy")
        self.dim = dim
        self.max_len = len(pos_list)
        self.grid_pe = np.zeros((dim, self.max_len), dtype=np.complex128) 

        # Initialize angle and rotation
        theta = 2 * np.pi / 3
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        random_angle_list = np.random.uniform(0, 2 * np.pi, dim)

        # Precompute the position encodings
        for k in range(dim):
            random_angle = random_angle_list[k]
            scale_factor = 200 ** (
                k / dim
            )
            omega_n0 = (
                np.array([np.cos(random_angle), np.sin(random_angle)]) * scale_factor
            )
            omega_n1 = R.dot(omega_n0)
            omega_n2 = R.dot(omega_n1)

            # Compute cos and sin for all positions for this dimension
            for idx in range(self.max_len):
                coords = pos_list[idx]
                eiw0x = np.exp(1j * np.dot(omega_n0, coords))
                eiw1x = np.exp(1j * np.dot(omega_n1, coords))
                eiw2x = np.exp(1j * np.dot(omega_n2, coords))

                g = Q.dot(np.array([eiw0x, eiw1x, eiw2x]))
                g = np.sum(g, axis=0)
                self.grid_pe[k, idx] = g

    def apply_encoding(self, x):
        batch_size, heads, num_queries, dim = x.shape
        assert (
            dim == self.dim
        ), "Dimension of x must match dimension of positional encoding"
        assert (
            num_queries <= self.max_len
        ), "Input sequence length exceeds maximum length"

        # if not torch.is_complex(x):
        #     x = torch.complex(x.float(), torch.zeros_like(x).float())

        if not isinstance(self.grid_pe, torch.Tensor):
            self.grid_pe = torch.view_as_complex(
                torch.from_numpy(
                    np.stack((self.grid_pe.real, self.grid_pe.imag), axis=-1)
                )
            )

        # Move grid_pe to the same device as x and ensure complex type
        self.grid_pe = self.grid_pe.to(device=x.device, dtype=torch.complex128)

        # Apply the encoding
        grid_pe_expanded = self.grid_pe.unsqueeze(0).unsqueeze(
            0
        )  # Shape: [1, 1, dim, num_queries]
        grid_pe_expanded = grid_pe_expanded.expand(
            batch_size, heads, -1, -1
        )  # Shape: [batch_size, heads, dim, num_queries]

        x = x.to(dtype=torch.complex128)

        # Perform element-wise multiplication and sum across the dim dimension to get the result
        encoded = torch.einsum("bhqd,bhdq->bhqd", x, grid_pe_expanded)

        return encoded

class ComplexToReal(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexToReal, self).__init__()
        self.real_layer = nn.Linear(input_dim, output_dim)
        self.imag_layer = nn.Linear(input_dim, output_dim)
        self.combine_layer = nn.Linear(2 * output_dim, output_dim)

    def forward(self, complex_tensor):
        real_part = complex_tensor.real.float()
        imag_part = complex_tensor.imag.float()

        real_output = self.real_layer(real_part)
        imag_output = self.imag_layer(imag_part)

        combined_output = torch.cat((real_output, imag_output), dim=-1)
        final_output = self.combine_layer(combined_output)
        return final_output
    
class GridDeepPositionalEncoding:
    def __init__(self, pos_list, dim):
        Q = np.load("./data/Q_matrix.npy")
        self.dim = dim
        self.max_len = len(pos_list)
        self.grid_pe = np.zeros((dim, self.max_len), dtype=np.complex128)

        # Initialize angle and rotation
        theta = 2 * np.pi / 3
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        random_angle_list = np.random.uniform(0, 2 * np.pi, dim)

        # Precompute the position encodings
        for k in range(dim):
            random_angle = random_angle_list[k]
            scale_factor = 200 ** (
                k / dim
            )
            omega_n0 = (
                np.array([np.cos(random_angle), np.sin(random_angle)]) * scale_factor
            )
            omega_n1 = R.dot(omega_n0)
            omega_n2 = R.dot(omega_n1)

            # Compute cos and sin for all positions for this dimension
            for idx in range(self.max_len):
                coords = pos_list[idx]
                eiw0x = np.exp(1j * np.dot(omega_n0, coords))
                eiw1x = np.exp(1j * np.dot(omega_n1, coords))
                eiw2x = np.exp(1j * np.dot(omega_n2, coords))

                g = Q.dot(np.array([eiw0x, eiw1x, eiw2x]))
                g = np.sum(g, axis=0)
                self.grid_pe[k, idx] = g

    def apply_encoding(self, x):
        batch_size, heads, num_queries, dim = x.shape
        assert (
            dim == self.dim
        ), "Dimension of x must match dimension of positional encoding"
        assert (
            num_queries <= self.max_len
        ), "Input sequence length exceeds maximum length"

        # if not torch.is_complex(x):
        #     x = torch.complex(x.float(), torch.zeros_like(x).float())

        # if not isinstance(self.grid_pe, torch.Tensor):
        #     self.grid_pe = torch.view_as_complex(
        #         torch.from_numpy(
        #             np.stack((self.grid_pe.real, self.grid_pe.imag), axis=-1)
        #         )
        #     )

        # Move grid_pe to the same device as x and ensure complex type
        # self.grid_pe = self.grid_pe.to(device=x.device, dtype=torch.complex128)
        self.grid_pe = self.grid_pe.to(device=x.device)

        # Apply the encoding
        grid_pe_expanded = self.grid_pe.unsqueeze(0).unsqueeze(
            0
        )  # Shape: [1, 1, dim, num_queries]
        grid_pe_expanded = grid_pe_expanded.expand(
            batch_size, heads, -1, -1
        )  # Shape: [batch_size, heads, dim, num_queries]

        # x = x.to(dtype=torch.complex128)

        # Perform element-wise multiplication and sum across the dim dimension to get the result
        encoded = torch.einsum("bhqd,bhdq->bhqd", x, grid_pe_expanded)

        return encoded