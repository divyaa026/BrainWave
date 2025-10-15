"""
Image Generator for BrainWave Analyzer
Generates synthetic images that correspond to different brain states
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict
from enum import Enum
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import minmax_scale


class ImageStyle(Enum):
    """Different image styles corresponding to brain states"""
    CALM_LANDSCAPE = "calm_landscape"      # Alpha waves
    STRUCTURED_OBJECT = "structured"       # Beta waves  
    COMPLEX_TEXTURE = "complex"            # Gamma waves
    MOTION_DYNAMIC = "motion"              # Motor imagery
    SOFT_DREAMY = "dreamy"                 # Sleep state


class ImageGenerator:
    """Generates synthetic images corresponding to brain states"""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (64, 64),
                 n_colors: int = 3):
        """
        Initialize image generator
        
        Args:
            image_size: (height, width) of generated images
            n_colors: Number of color channels
        """
        self.image_size = image_size
        self.n_colors = n_colors
        self.height, self.width = image_size
        
    def generate_brain_state_image(self, 
                                  brain_state_idx: int,
                                  style_variation: float = 0.5) -> np.ndarray:
        """
        Generate image corresponding to brain state
        
        Args:
            brain_state_idx: Index of brain state (0-4)
            style_variation: Variation in style (0-1)
            
        Returns:
            Image array of shape (height, width, n_colors) in range [0, 1]
        """
        if brain_state_idx == 0:  # Relaxed (Alpha) - Calm landscapes
            return self._generate_calm_landscape(style_variation)
        elif brain_state_idx == 1:  # Focused (Beta) - Structured objects
            return self._generate_structured_object(style_variation)
        elif brain_state_idx == 2:  # Active (Gamma) - Complex textures
            return self._generate_complex_texture(style_variation)
        elif brain_state_idx == 3:  # Motor imagery - Motion/dynamic
            return self._generate_motion_dynamic(style_variation)
        elif brain_state_idx == 4:  # Sleep - Soft/dreamy
            return self._generate_soft_dreamy(style_variation)
        else:
            # Default to calm landscape
            return self._generate_calm_landscape(style_variation)
    
    def _generate_calm_landscape(self, variation: float) -> np.ndarray:
        """Generate calm landscape for alpha waves (relaxed state)"""
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Create gradient sky (blue to light blue)
        for y in range(self.height):
            gradient_factor = y / self.height
            blue_intensity = 0.6 + 0.3 * gradient_factor
            green_intensity = 0.4 + 0.2 * gradient_factor
            red_intensity = 0.2 + 0.1 * gradient_factor
            
            img[y, :, 0] = red_intensity + variation * 0.1 * np.random.randn(self.width)
            img[y, :, 1] = green_intensity + variation * 0.1 * np.random.randn(self.width)
            img[y, :, 2] = blue_intensity + variation * 0.1 * np.random.randn(self.width)
        
        # Add soft clouds
        for _ in range(3):
            cloud_x = np.random.randint(0, self.width - 20)
            cloud_y = np.random.randint(0, self.height // 3)
            cloud_size = np.random.randint(10, 20)
            
            cv2.circle(img, (cloud_x, cloud_y), cloud_size, 
                      (0.8, 0.8, 0.9), -1)
        
        # Add calm water/horizon line
        horizon_y = int(self.height * 0.7)
        img[horizon_y:, :, 0] = 0.1  # Dark blue water
        img[horizon_y:, :, 1] = 0.3
        img[horizon_y:, :, 2] = 0.5
        
        # Add gentle waves
        for y in range(horizon_y, self.height, 5):
            wave_offset = int(5 * np.sin(y * 0.2))
            wave_start = max(0, wave_offset)
            wave_end = min(self.width, self.width + wave_offset)
            img[y, wave_start:wave_end, :] += 0.1
        
        return np.clip(img, 0, 1)
    
    def _generate_structured_object(self, variation: float) -> np.ndarray:
        """Generate structured geometric object for beta waves (focused state)"""
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Background (light gray)
        img.fill(0.9)
        
        # Create geometric shapes
        center_x, center_y = self.width // 2, self.height // 2
        
        # Main geometric shape (rectangle or circle)
        if np.random.random() < 0.5:
            # Rectangle
            rect_size = (20, 30)
            top_left = (center_x - rect_size[0]//2, center_y - rect_size[1]//2)
            bottom_right = (center_x + rect_size[0]//2, center_y + rect_size[1]//2)
            
            cv2.rectangle(img, top_left, bottom_right, (0.2, 0.4, 0.8), -1)
            cv2.rectangle(img, top_left, bottom_right, (0.1, 0.2, 0.6), 2)
        else:
            # Circle
            radius = 15
            cv2.circle(img, (center_x, center_y), radius, (0.2, 0.4, 0.8), -1)
            cv2.circle(img, (center_x, center_y), radius, (0.1, 0.2, 0.6), 2)
        
        # Add smaller geometric elements
        for _ in range(5):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            size = np.random.randint(3, 8)
            
            if np.random.random() < 0.5:
                cv2.circle(img, (x, y), size, (0.6, 0.6, 0.6), -1)
            else:
                cv2.rectangle(img, (x-size, y-size), (x+size, y+size), 
                             (0.6, 0.6, 0.6), -1)
        
        # Add grid pattern overlay
        grid_spacing = 10
        for x in range(0, self.width, grid_spacing):
            cv2.line(img, (x, 0), (x, self.height), (0.7, 0.7, 0.7), 1)
        for y in range(0, self.height, grid_spacing):
            cv2.line(img, (0, y), (self.width, y), (0.7, 0.7, 0.7), 1)
        
        return np.clip(img, 0, 1)
    
    def _generate_complex_texture(self, variation: float) -> np.ndarray:
        """Generate complex texture for gamma waves (active state)"""
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Create fractal-like pattern
        for scale in [1, 2, 4, 8]:
            noise = np.random.randn(self.height // scale, self.width // scale, 3)
            noise_resized = cv2.resize(noise, (self.width, self.height))
            img += (1.0 / scale) * noise_resized
        
        # Normalize
        img = (img - img.min()) / (img.max() - img.min())
        
        # Add color variations
        color_shift = np.random.uniform(-0.3, 0.3, 3)
        img += color_shift
        
        # Add high-frequency details
        high_freq_noise = 0.2 * np.random.randn(self.height, self.width, 3)
        img += high_freq_noise
        
        # Apply some smoothing to create interesting patterns
        img = gaussian_filter(img, sigma=1.0)
        
        return np.clip(img, 0, 1)
    
    def _generate_motion_dynamic(self, variation: float) -> np.ndarray:
        """Generate motion/dynamic image for motor imagery"""
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Create motion blur effect
        center_x, center_y = self.width // 2, self.height // 2
        
        # Main object with motion trail
        object_positions = []
        for i in range(10):
            angle = i * 2 * np.pi / 10
            x = int(center_x + 15 * np.cos(angle))
            y = int(center_y + 15 * np.sin(angle))
            object_positions.append((x, y))
        
        # Draw motion trail
        for i, (x, y) in enumerate(object_positions):
            if 0 <= x < self.width and 0 <= y < self.height:
                intensity = 1.0 - (i * 0.1)  # Fade trail
                cv2.circle(img, (x, y), 3, 
                          (intensity, intensity * 0.5, intensity * 0.3), -1)
        
        # Add directional lines
        for _ in range(5):
            start_x = np.random.randint(0, self.width)
            start_y = np.random.randint(0, self.height)
            end_x = start_x + np.random.randint(-20, 20)
            end_y = start_y + np.random.randint(-20, 20)
            
            # Clamp to image bounds
            end_x = np.clip(end_x, 0, self.width - 1)
            end_y = np.clip(end_y, 0, self.height - 1)
            
            cv2.line(img, (start_x, start_y), (end_x, end_y), 
                    (0.8, 0.4, 0.2), 2)
        
        # Add speed lines
        for _ in range(10):
            y = np.random.randint(0, self.height)
            start_x = np.random.randint(0, self.width // 2)
            end_x = start_x + np.random.randint(10, 30)
            end_x = min(end_x, self.width - 1)
            
            cv2.line(img, (start_x, y), (end_x, y), (0.6, 0.6, 0.6), 1)
        
        return np.clip(img, 0, 1)
    
    def _generate_soft_dreamy(self, variation: float) -> np.ndarray:
        """Generate soft, dreamy image for sleep state"""
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Create soft gradient background
        for y in range(self.height):
            for x in range(self.width):
                # Soft purple-pink gradient
                gradient_factor = (x + y) / (self.width + self.height)
                
                img[y, x, 0] = 0.8 + 0.2 * gradient_factor  # Red
                img[y, x, 1] = 0.6 + 0.3 * gradient_factor  # Green  
                img[y, x, 2] = 0.9 + 0.1 * gradient_factor  # Blue
        
        # Add soft, blurry shapes
        for _ in range(8):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            size = np.random.randint(15, 30)
            
            # Create soft circle
            mask = np.zeros((self.height, self.width), dtype=np.float32)
            cv2.circle(mask, (x, y), size, 1.0, -1)
            mask = gaussian_filter(mask, sigma=5.0)
            
            # Add to image with soft colors
            color = np.random.uniform(0.3, 0.7, 3)
            for c in range(3):
                img[:, :, c] += 0.3 * mask * color[c]
        
        # Apply overall soft blur
        img = gaussian_filter(img, sigma=2.0)
        
        return np.clip(img, 0, 1)
    
    def generate_dataset(self, 
                        n_samples: int,
                        brain_state_labels: np.ndarray) -> np.ndarray:
        """
        Generate dataset of images corresponding to brain states
        
        Args:
            n_samples: Number of images to generate
            brain_state_labels: Brain state indices for each image
            
        Returns:
            Image dataset of shape (n_samples, height, width, n_colors)
        """
        images = np.zeros((n_samples, self.height, self.width, self.n_colors), 
                         dtype=np.float32)
        
        for i in range(n_samples):
            variation = np.random.uniform(0.3, 0.8)
            images[i] = self.generate_brain_state_image(
                brain_state_labels[i], 
                variation
            )
        
        return images


def create_synthetic_image_dataset(n_samples: int,
                                  brain_state_labels: np.ndarray,
                                  image_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Create synthetic image dataset
    
    Args:
        n_samples: Number of images
        brain_state_labels: Brain state indices
        image_size: Image dimensions
        
    Returns:
        Image dataset
    """
    generator = ImageGenerator(image_size=image_size)
    return generator.generate_dataset(n_samples, brain_state_labels)


if __name__ == "__main__":
    # Demo usage
    print("Generating synthetic image dataset...")
    
    # Create generator
    generator = ImageGenerator(image_size=(64, 64))
    
    # Generate sample for each brain state
    brain_states = ["Relaxed (Alpha)", "Focused (Beta)", "Active (Gamma)", 
                   "Motor Imagery", "Sleep"]
    
    for i, state_name in enumerate(brain_states):
        print(f"Generating {state_name} image...")
        img = generator.generate_brain_state_image(i, variation=0.5)
        print(f"  Shape: {img.shape}, Range: [{img.min():.2f}, {img.max():.2f}]")
        
        # Save sample image
        img_display = (img * 255).astype(np.uint8)
        cv2.imwrite(f"sample_{state_name.lower().replace(' ', '_')}.png", 
                   cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
    
    print("Sample images saved!")
    
    # Generate small dataset
    print("Creating small dataset...")
    labels = np.random.randint(0, 5, 20)
    dataset = create_synthetic_image_dataset(20, labels)
    print(f"Dataset created with shape: {dataset.shape}")
