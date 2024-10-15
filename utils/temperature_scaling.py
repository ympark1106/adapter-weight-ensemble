import logging
import torch
import numpy as np
from torch import nn
from calibrate.evaluation.metrics import ECELoss

logger = logging.getLogger(__name__)

class ModelWithTemperature(nn.Module):
    """
    A decorator to wrap a model with temperature scaling.
    The model should output logits, not softmax or log softmax.
    """
    def __init__(self, model, device="cuda:0", log=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0).to(device)
        self.log = log
        self.device = device

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped model.
        This allows accessing model.linear, model.forward_features, etc.
        """
        if name != 'model':  # Prevent infinite recursion
            return getattr(self.model, name)
        return super().__getattr__(name)

    def forward(self, input):
        # Use the model's forward_features and then apply temperature scaling
        if hasattr(self.model, 'forward_features'):
            logits = self.model.forward_features(input)[:, 0, :]
        else:
            logits = self.model(input)

        # Apply the linear layer if it exists in the model
        if hasattr(self.model, 'linear'):
            logits = self.model.linear(logits)
        else:
            raise AttributeError(f"The model {type(self.model).__name__} does not have a 'linear' layer.")

        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        return logits / self.temperature

    def set_temperature(self, valid_loader, cross_validate='ece'):
        """
        Tune the temperature of the model using the validation set.
        Cross-validation can be based on ECE or NLL.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss().to(self.device)

        # Collect logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.forward(input)
                logits_list.append(logits)
                labels_list.append(label.to(self.device))

        # Concatenate logits and labels
        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            logger.info(f'Before temperature - NLL: {before_temperature_nll:.4f}, ECE: {before_temperature_ece:.4f}')

        # Optimize temperature using grid search
        nll_val = float('inf')
        ece_val = float('inf')
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for _ in range(100):
            self.temperature.data = torch.tensor([T], device=self.device)
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll
            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        # Set optimal temperature
        if cross_validate == 'ece':
            self.temperature.data = torch.tensor([T_opt_ece], device=self.device)
        else:
            self.temperature.data = torch.tensor([T_opt_nll], device=self.device)

        # Log results after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            logger.info(f'Optimal temperature: {self.temperature.item():.3f}')
            logger.info(f'After temperature - NLL: {after_temperature_nll:.4f}, ECE: {after_temperature_ece:.4f}')

    def get_temperature(self):
        return self.temperature.item()

