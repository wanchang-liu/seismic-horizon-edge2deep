from torch import nn


# Attention Mechanism
class Attention_Gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)  # Convolution of the downsampled gating signal
        x1 = self.W_x(x)  # Convolution of the upsampled x (l)
        psi = self.relu(g1 + x1)  # Concatenate + ReLU
        psi = self.psi(psi)  # Reduce the channel to 1 and apply Sigmoid to get the weight matrix
        return x * psi  # Return the weighted x
