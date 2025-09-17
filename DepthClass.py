
'''

Copyright (c) 2025 Bashayer Abdallah
Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
Commercial use is prohibited.

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from FFM import FFM
from autodistill_sam_clip import SAMCLIP  


class DepthClass(nn.Module):
    def __init__(self, ontology):
        super(DepthClass, self).__init__()

        # Swin-Large encoder for RGB images
        self.swin_large = create_model(
            'swinv2_large_window12to16_192to256_22kft1k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        # CNN encoder for depth maps
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Output size: (W/2, H/2)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1),  # Output size: (W/4, H/4)
            nn.BatchNorm2d(192),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # Feature Fusion Module (FFM)
        self.ffm = FFM(
            in_channels=[192, 192, 384, 768, 1536],  # E, s1, s2, s3, s4 channel dimensions
            out_channels=256,
            embedding_dim=128
        )

        # Decoder blocks (for upsampling and prediction)
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ELU(inplace=True)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ELU(inplace=True)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ELU(inplace=True)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(512, 128, kernel_size=3, padding=1),
                nn.ELU(inplace=True)
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(128, 60, kernel_size=3, padding=1),
                nn.ELU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(60, 1, kernel_size=1),
                nn.Sigmoid()
            )
        ])

        # SAM-CLIP text encoder for classification.
        self.sam_clip_encoder = SAMCLIP(ontology = ontology)

        # Learnable temperature parameter (initial value can be tuned)
        self.temperature = nn.Parameter(torch.tensor(0.07))


        # Projection layer to match image embedding dimension to text embedding dimension.
        self.image_projection = nn.Linear(1536, 512)

    def forward(self, x1, x2, prompts=None):
        """
        x1: RGB image tensor.
        x2: Depth map tensor.
        prompts: A list of text prompts (e.g., class names) to compare against the image.
        """
        # 1. Process the RGB image with the Swin-Large encoder.
        swin_features = self.swin_large(x1)  # List of feature maps: [s1, s2, s3, s4]
        swin_features = [s.permute(0, 3, 1, 2) if s.dim() == 4 else s for s in swin_features]

        # 2. Process the depth map with the CNN encoder.
        cnn_features = self.cnn_encoder(x2)

        # 3. Fuse features.
        fused_features = self.ffm([*swin_features], [cnn_features])
        skip_connections = list(reversed(fused_features))

        # 4. Decode fused features.
        out = skip_connections[0]
        for i, decoder_block in enumerate(self.decoder_blocks):
            out = decoder_block(out)
            # print(f"Decoder block {i} output shape before skip cat: {out.shape}")
            if i < len(skip_connections) - 1:
                out = torch.cat((out, skip_connections[i + 1]), dim=1)
                # print(f"After concat skip {i} output shape: {out.shape}")

        # 5. Classification branch using SAM-CLIP.
        image_feature = swin_features[-1].mean(dim=[2, 3])  # (B, C)
        image_feature_proj = self.image_projection(image_feature)
        image_feature_norm = F.normalize(image_feature_proj, p=2, dim=1)

        # 6. Encode text prompts.
        # Encode text prompts.
        if prompts is None:
            prompts = self.sam_clip_encoder.ontology
   
        # Tokenize the text prompts.
        tokens = self.sam_clip_encoder.tokenize(prompts)
        # Query the device of the token_embedding weights:
        device_clip = next(self.sam_clip_encoder.clip_model.token_embedding.parameters()).device
        # Move tokens to the same device as the token_embedding:
        tokens = tokens.to(device_clip)
        # Get text embeddings:
        text_features = self.sam_clip_encoder.clip_model.encode_text(tokens)
        text_features_norm = F.normalize(text_features, p=2, dim=1)
        text_features_norm = text_features_norm.to(torch.float32)
        text_features_norm = text_features_norm.to(x1.device)  # Ensure text features are on the same device as x1.

        # 7. Compute cosine similarity.
        similarity = torch.matmul(image_feature_norm, text_features_norm.t())

        # Scale the similarities by the learnable temperature.
        # Instead of moving self.temperature manually, assume it is already on the right device.
        logits = similarity * self.temperature.exp()

        # Compute probabilities and predicted classes.
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return out, logits, probs, preds, prompts

