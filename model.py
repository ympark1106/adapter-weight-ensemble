import torch
import torch.nn as nn

import dino_variant
import rein

def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False

model_load = dino_variant._small_dino  
variant = dino_variant._small_variant

model = torch.hub.load('facebookresearch/dinov2', model_load)
dino_state_dict = model.state_dict()

model = rein.ReinsDinoVisionTransformer(
    **dino_variant._small_variant
)

# model = rein.ReinsDinoVisionTransformer_Dropout(
#     **dino_variant._small_variant,
#     dropout_rate=0.5
# )

set_requires_grad(model, ["reins", "linear"])

model.load_state_dict(dino_state_dict, strict=False)
model.linear = nn.Linear(dino_variant._small_variant['embed_dim'], 10)

print(model)

