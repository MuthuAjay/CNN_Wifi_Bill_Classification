from pathlib import Path
import fitz
from typing import Optional, List, Dict
import torch.nn
import matplotlib.pyplot as plt
from torchvision import transforms
import PIL

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_image(model: torch.nn.Module,
                  path: str | Path,
                  classes: List) -> str:
    rgba_image = PIL.Image.open(path)
    rgb_image = rgba_image.convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    # change the data type
    custom_image = transform(rgb_image)
    custom_image = custom_image.type(torch.float32)

    # cutom_image in 0 to 255 range
    # custom_image = custom_image / 255.

    # change the shape
    # 1. create a transform fn
    transform = transforms.Compose([transforms.Resize(size=(64, 64))])

    custom_image = transform(custom_image)
    # 2.It should contain batch size in the shape
    custom_image = custom_image.unsqueeze(dim=(0))
    # Change the device type cpu or cuda
    custom_image = custom_image.to(device)

    # predict the image
    model.eval()
    with torch.inference_mode():
        pred_logits = model(custom_image)
        preds = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)
        print(torch.softmax(pred_logits, dim=1))
        prob = torch.softmax(pred_logits, dim=1).max().item()
    plt.imshow(custom_image.squeeze(dim=0).permute(1, 2, 0).to("cpu"))
    plt.title(f"{classes[preds]} - Prob : {prob:.3f}")

    return classes[preds]


def predict(path: Optional[List[Path | str]],
            model: torch.nn.Module,
            classes) -> Dict:
    result = {}
    for file in path:

        if isinstance(file, str):
            file = Path(file)

        if file.is_file():

            if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif'}:
                result[str(file)] = predict_image(model=model,
                                                  path=file,
                                                  classes=classes)

            elif file.suffix.lower() == ".pdf":
                # Open the PDF file
                doc = fitz.open(file)

                # Iterate over each page in the PDF
                for page_number in range(doc.page_count):
                    page = doc[page_number]

                    # Get the pixmap for the page
                    pix = page.get_pixmap(matrix=fitz.Identity, dpi=None,
                                          colorspace=fitz.csRGB, clip=None, alpha=True, annots=True)
                    temp = Path("/content")
                    page_image_file = temp / f"{file.stem}.png"
                    pix.save(page_image_file)
                    print(f"Extracted image from PDF page {page_number}: {page_image_file}")
                    break

                # Close the PDF document
                doc.close()
                result[str(file)] = predict_image(model=model,
                                                  path=file,
                                                  classes=classes)
            else:
                result[str(file)] = "Invalid File"
                print(f"Skipping unsupported file: {file}")

    return result
