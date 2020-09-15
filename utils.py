import torch
from torchtext.data.metrics import bleu_score

def bleu(data, model, vocab,, device):
    targets = []
    outputs = []

    for (image,caption) in tqdm(data):
        
        targets.append([trg_text])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

import torchvision.transforms as transforms
from PIL import Image

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("/content/test/A brown dog is sprayed with water.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: A brown dog is sprayed with water.jpg")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    
    test_img2 = transform(
        Image.open("/content/test/A person kayaking in the ocean.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: A person kayaking in the ocean")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )

    test_img3 = transform(
        Image.open("/content/Images/1007129816_e794419615.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: A man in an orange hat starring at something .")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    model.train()
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step