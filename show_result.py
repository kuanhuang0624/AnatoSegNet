import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random
import PIL.Image as Image
from sklearn.metrics import f1_score
from torchvision import transforms
from models.AnatoNet import UResNetWithAttention


def calculate_f1_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    f1 = f1_score(target, pred, average='binary')
    return f1

def calculate_iou(pred, target):
    pred_tumor = torch.tensor(pred == 1, dtype=torch.float32)
    target_tumor = torch.tensor(target == 1, dtype=torch.float32)
    intersection = (pred_tumor * target_tumor).sum()
    union = pred_tumor.sum() + target_tumor.sum() - intersection
    iou = intersection / (union + 1e-6)
    return iou.item()

if __name__ == "__main__":
    # Set up random seed
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def load_images_and_masks(directory):
        mask_paths = []
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                mask_paths.append(os.path.join(directory, filename))
        return mask_paths

    # Load benign and malignant images and masks
    mask_paths = load_images_and_masks('../Dataset_BUSI_with_GT/labels')
    # mask_paths = load_images_and_masks('../BrEaST-Lesions_USG-images_and_masks/labels')

    masks_train, masks_test = train_test_split(mask_paths, test_size=0.2, random_state=42)

    # Initialize model and set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UResNetWithAttention(n_classes=2).to(device)

    model.load_state_dict(torch.load("./checkpoints/unet_differential_attention_model_busi.pth"))
    model.eval()

    tfms = transforms.Compose([transforms.ToTensor()])
    results_dir = "./results/busi/proposed/"
    os.makedirs(results_dir, exist_ok=True)

    total_iou = 0.0
    total_f1 = 0.0
    num_samples = len(masks_test)

    # Loop through the test set
    for mask_path in masks_test:
        image_name = mask_path.split('/')[-1]
        image_path = mask_path.replace("labels", "images")
        image = Image.open(image_path).convert("RGB")
        label = Image.open(mask_path)
        label_array = np.array(label)

        # Resize label and image to 256x256 for model input
        label_resize = label.resize((256, 256), Image.NEAREST)
        label_resize = np.array(label_resize, dtype=np.float32)

        image_resize = image.resize((256, 256))
        image_resize = np.array(image_resize)
        img_new = np.float32(image_resize) / 127.5 - 1
        img_new = tfms(img_new).unsqueeze(0).to(device)

        # Predict the mask
        with torch.no_grad():
            pred = model(img_new)
            pred = torch.argmax(pred, dim=1).cpu().numpy()

        # Calculate IoU and F1 score
        iou = calculate_iou(pred, label_resize)
        f1 = calculate_f1_score(pred, label_resize)
        total_iou += iou
        total_f1 += f1

        print(f"Image: {image_name}, IoU: {iou:.4f}, F1: {f1:.4f}")

        # Resize prediction back to original image size
        pred_resized = Image.fromarray(pred[0].astype(np.uint8)).resize(image.size, Image.NEAREST)
        pred_resized = np.array(pred_resized)

        # Create a black and white binary mask
        binary_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        binary_mask[(label_array > 0) & (pred_resized > 0)] = 255  # White for true positives
        # binary_mask[(label_array == 0) & (pred_resized == 0)] = 255  # White for correct background

        # Create a 3-channel mask for visualization
        color_mask = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)

        # Add false negative and false positive colors to the mask
        false_negatives = (label_array > 0) & (pred_resized == 0)  # Red: False Negative
        false_positives = (label_array == 0) & (pred_resized > 0)  # Green: False Positive

        color_mask[false_negatives] = [255, 0, 0]  # Red
        color_mask[false_positives] = [0, 255, 0]  # Green

        # Save the result as a colored mask
        save_path = os.path.join(results_dir, f"{image_name}")
        Image.fromarray(color_mask).save(save_path)

    # Compute and print the average IoU and F1 score for the test set
    avg_iou = total_iou / num_samples
    avg_f1 = total_f1 / num_samples
    print(f"\nAverage IoU on test set: {avg_iou:.4f}")
    print(f"Average F1 score on test set: {avg_f1:.4f}")
