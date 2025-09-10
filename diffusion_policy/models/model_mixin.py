import torch
import torch.nn as nn


class ModelMixin(nn.Module):
    def __init__(self):
        super().__init__()

    def _get_first_parameter(self):
        try:
            return next(self.parameters())
        except StopIteration:
            raise RuntimeError(f"{self.__class__.__name__} has no parameters.")

    @property
    def device(self):
        return self._get_first_parameter().device

    @property
    def dtype(self):
        return self._get_first_parameter().dtype

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def size(self):
        """Return the total size of parameters and buffers in bytes."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return param_size + buffer_size

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, strict=True):
        """Load a model checkpoint, automatically handling DataParallel models."""
        state_dict = torch.load(path)

        # If loading from a checkpoint, sometimes state_dict is inside a dict
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Check if keys are prefixed with "module."
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":

    class Model(ModelMixin):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Linear(100, 4), nn.Sigmoid())

        def forward(self, x):
            return self.layers(x)

    model = Model()

    print(f"{model.device = }")

    model.to("mps")

    print(f"after device change: {model.device = }")
    print(f"{model.dtype = }")
    print(f"{model.num_parameters = }")
    print(f"{model.size = }")

    input = torch.randn(8, 3).to(model.device)
    print(f"{input = }")
    output = model(input)
    print(f"{output = }")
