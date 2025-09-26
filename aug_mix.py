from scipy.stats import dirichlet, beta
from augmentations import *

def apply_operation(image, mask, operation):
    ops = {
        "rotate": random_rotation,
        "flip": flip_image,
        "contrast_CLAHE": contrast_CLAHE,
        "blur": gaussian_blur,
        "noise": add_gaussian_noise,
        "translate": apply_translation,
        "gamma_correction" : gamma_correction,
        "none" : default_operation
    }
    return ops[operation](image, mask)

def aug_mix(image, mask, k=3, alpha=1.0):
    h, w = image.shape[:2]
    

    # Weights mixing (Dirichlet)
    weights = dirichlet.rvs([alpha] * k)[0]
    

    geometric_ops = ["rotate", "flip", "none", "translate"]
    color_ops = ["contrast_CLAHE", "blur", "noise", "gamma_correction"]

    geom_op = random.choice(geometric_ops)
    image, mask = apply_operation(image, mask, geom_op)

    for weight in weights:
        ops_chain = random.sample(color_ops, random.randint(1, len(color_ops)))
        augmented_image = image.copy()

        for op in ops_chain:
            augmented_image, _ = apply_operation(augmented_image, mask, op)

        x_aug = np.zeros_like(image, dtype=np.float32)
        x_aug += weight * augmented_image.astype(np.float32)
       

    m = beta.rvs(alpha, alpha)*0.5
    x_augmix = m * image + (1 - m) * x_aug
    x_augmix = np.clip(x_augmix, 0, 255).astype(np.uint8)


    return x_augmix, mask