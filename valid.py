# inference_evaluation.py

import torch
import numpy as np
import evaluation

def rein_forward(model, inputs):
    # Forward pass for the model
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)
    return output

def validate(model, valid_loader, device, evaluation):
    """
    Perform inference on the test_loader using the given model and evaluate results.

    Args:
    - model: The model to use for inference.
    - test_loader: DataLoader for the test set.
    - device: Device to perform inference on (e.g., 'cpu' or 'cuda').
    - evaluation: Evaluation module with evaluate function.

    Returns:
    - None, but prints evaluation results.
    """
    outputs = []
    targets = []
    
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(valid_loader):
            # Move inputs and targets to the specified device
            inputs, target = inputs.to(device), target.to(device)
            
            # Forward pass
            output = rein_forward(model, inputs)
            
            # Append results
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    # Concatenate all outputs and targets
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    
    # Call the evaluation function
    ece = evaluation.calculate_ece(outputs, targets)
    
    
    return ece
