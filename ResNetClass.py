import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F
import torch.nn.functional as F




class DepthDecoder(nn.Module):
    def __init__(self, encoder_feature_dims):
        super().__init__()

        # Decoder blocks with channel alignment layers
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_feature_dims[3], 768, kernel_size=3, stride=1, padding=1),  # Align channels to skip_connection[1]
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True),
                nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1),  # Align channels to skip_connection[2]
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(384, 192, kernel_size=3, stride=1, padding=1),  # Align channels to skip_connection[3]
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(192, 60, kernel_size=3, stride=1, padding=1),  # Final output block
                nn.BatchNorm2d(60),
                nn.ReLU(inplace=True),
                nn.Conv2d(60, 60, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(60),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(60, 40, kernel_size=3, stride=1, padding=1),  # Final output block
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
                nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True)
            ),
        ])
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(40, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, swin_features):
        # Reverse swin features for skip connections
        swin_features = [s.permute(0, 3, 1, 2) if s.dim() == 4 else s for s in swin_features]
        skip_connections = list(reversed(swin_features))

        # Debugging: Print the sizes of the skip connections
        # for idx, skip in enumerate(skip_connections):
        # print(f"Skip connection {idx} size: {skip.shape}")

        # Start decoding using the smallest skip connection (s4)
        out = skip_connections[0]
        # print(f"Initial output size: {out.shape}")

        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample output
            if i < len(skip_connections) - 1:
                out = F.interpolate(out, size=skip_connections[i + 1].shape[2:], mode="bilinear", align_corners=True)
                # print(f"After upsampling at block {i}, output size: {out.shape}")

            # Align channels with skip connection
            out = decoder_block(out)
            # print(f"After decoder block {i}, output size: {out.shape}")

            # Add skip connection
            if i < len(skip_connections) - 1:
                out = out + skip_connections[i + 1]
                # print(f"After adding skip connection {i + 1}, output size: {out.shape}")

        # Apply final upsampling to match target resolution (e.g., 256x256)
        out = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=True)
        # print(f"After final upsampling, output size: {out.shape}")

        # Apply final layer to generate depth map
        out = self.final(out)
        # print(f"Final output size: {out.shape}")
        return out


# Classification Module
class ClassificationModule(nn.Module):
    def __init__(self, visual_dim, num_classes, embedding_dim=512):
        super().__init__()
        self.visual_dim = visual_dim  # Should match the dimension of pooled_features
        self.embedding_dim = embedding_dim  # Default: 512
        self.num_classes = num_classes

        # Learnable class embeddings
        self.class_embeddings = nn.Embedding(num_classes, embedding_dim)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(visual_dim + embedding_dim, 512),  # Align input dimension
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, visual_features, class_indices):
        # Retrieve class embeddings
        class_embeddings = self.class_embeddings(class_indices)
        class_embeddings_mean = class_embeddings.mean(dim=0).unsqueeze(0)  # (1, embedding_dim)
        class_embeddings_mean = class_embeddings_mean.expand(visual_features.size(0), -1)  # Match batch size

        # Concatenate visual features with class embeddings
        fused_features = torch.cat((visual_features, class_embeddings_mean), dim=-1)

        # Debugging: Print shapes
        # print(f"Pooled Visual Features Shape: {visual_features.shape}")
        # print(f"Class Embeddings Mean Shape: {class_embeddings_mean.shape}")
        # print(f"Fused Features Shape: {fused_features.shape}")

        # Predict class logits
        class_logits = self.classification_head(fused_features)
        return class_logits



# Full DepthClassNet Model
class DepthClassNet(nn.Module):
    def __init__(self, swin_large, depth_decoder, classification_module):
        super().__init__()
        self.swin_large = swin_large
        self.depth_decoder = depth_decoder
        self.classification_module = classification_module

    def forward(self, rgb_image, class_indices):
        # Extract feature maps from Swin Transformer
        swin_features = self.swin_large(rgb_image)  # [s1, s2, s3, s4]

        # Debugging: Print Swin feature sizes
        # for idx, feat in enumerate(swin_features):
            # print(f"Swin Feature {idx} size: {feat.shape}")

        # Predict depth map using skip connections
        depth_map = self.depth_decoder(swin_features)

        # Use the deepest feature map (s4) for classification
        deepest_feature = swin_features[-1]  # Shape: (batch_size, H, W, C)
        deepest_feature = deepest_feature.permute(0, 3, 1, 2)  # Convert to (batch_size, C, H, W)
        pooled_features = deepest_feature.mean(dim=(2, 3))  # Global Average Pooling

        # Debugging: Print pooled feature size
        # print(f"Corrected Pooled Features Shape: {pooled_features.shape}")

        # Predict class logits
        class_logits = self.classification_module(pooled_features, class_indices)

        return depth_map, class_logits


# Instantiate Model Components
def create_MDEclass(num_classes, pretrained=True, img_size=256):
    print("Initializing Swin Transformer...")

    # Initialize Swin-Large encoder
    swin_large = create_model(
        'swin_large_patch4_window7_224',
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3),  # Extract features at multiple stages
        img_size=(img_size, img_size)
    )

    # Extract feature dimensions for the decoder
    feature_dims = swin_large.feature_info.channels()

    # Return the full model with all components
    return DepthClassNet(
        swin_large=swin_large,
        depth_decoder=DepthDecoder(encoder_feature_dims=feature_dims),
        classification_module=ClassificationModule(
            visual_dim=feature_dims[-1],  # Deepest feature map dimension
            num_classes=num_classes
        )
    )



# # Test the Model
#
# num_classes = 9
# img_size = 256
# batch_size = 1
#
# # Create dummy input
# dummy_rgb_images = torch.randn(batch_size, 3, img_size, img_size)
# dummy_class_indices = torch.randint(0, num_classes, (batch_size,))
#
# # Instantiate model
# model = create_MDEclass(num_classes=num_classes, img_size=img_size)
#
# # Move model and data to device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)
# dummy_rgb_images = dummy_rgb_images.to(device)
# dummy_class_indices = dummy_class_indices.to(device)
#
# # Forward pass
# model.eval()
# with torch.no_grad():
#     depth_map, class_logits = model(dummy_rgb_images, dummy_class_indices)
#
# # Output results
# print("Depth Map Shape:", depth_map.shape)  # Expected: (batch_size, 1, img_size, img_size)
# print("Class Logits Shape:", class_logits.shape)  # Expected: (batch_size, num_classes)
