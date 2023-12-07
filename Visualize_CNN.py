import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from reverse_normalize import reverse_normalize

def visualize_feature_maps(means, stds, best_models, test_loader, num_interest_layers=3):

    # Select a random image from the test loader
    random_index = random.randint(0, len(test_loader.dataset) - 1)
    random_image, _ = test_loader.dataset[random_index]

    # Original code for displaying the original image
    image = reverse_normalize(random_image, means, stds)
    image_np = image.numpy().transpose((1, 2, 0))

    # Code for visualizing feature maps
    model_name = "CustomCNN"  # Assuming the model name you want to visualize
    model = best_models[best_models["model_name"] == model_name]["model"].values[0]

    # Get interested layers
    interested_layers = list(range(num_interest_layers))
    feature_maps = model.get_interested_layers(random_image.unsqueeze(0), interested_layers)

    # Create a subplot grid
    fig, axes = plt.subplots(1, len(interested_layers) + 2, figsize=(40, 10))

    # Display the original image
    axes[0].imshow(image_np)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    # Display the transformed image
    axes[1].imshow(random_image.numpy().transpose((1, 2, 0)).clip(0, 1))
    axes[1].axis('off')
    axes[1].set_title('Image after normalization')


    # Visualize the feature maps
    for i, feature_map in enumerate(feature_maps):
        ax = axes[i + 2]
        ax.imshow(feature_map[0, 0].detach().numpy(), cmap='viridis')  # Assuming a single channel in the feature map
        ax.set_title(f'Layer {interested_layers[i]}')
        ax.axis('off')

        # Set the aspect ratio to be equal
        ax.set_aspect('equal')

        # Adjust the number of ticks on the x and y axes
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()