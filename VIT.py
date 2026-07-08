from transformers import ViTModel
import torch


class ViTForImageClassification(torch.nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.hidden_size = self.vit.config.hidden_size
        self.classifier = torch.nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


if __name__ == "__main__":
    from torchinfo import summary

    model = ViTForImageClassification(101)
    x = torch.randn(1, 3, 224, 224)
    summary(
        model,
        input_data=x,
        col_names=("input_size", "output_size", "num_params", "trainable"),
    )
    with torch.no_grad():
        y = model(x)
    print(y.shape)
