import copy
import random
import numpy as np
import torch
from typing import List, Dict, Any, Union

import cv2
from detectron2.data import DatasetMapper
import detectron2.data.detection_utils as detection_utils
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from PIL import Image, ImageFilter, ImageEnhance


class DataAugmentation:
    """
    Data augmentation techniques specifically designed for object detection tasks.
    Includes augmentations that preserve bounding box annotations.
    """
    
    def __init__(self, augmentation_factor: int = 5):
        """
        Initialize data augmentation with specified augmentation factor.
        
        Args:
            augmentation_factor: How many times to augment each image (default: 5)
        """
        self.augmentation_factor = augmentation_factor
        
    def horizontal_flip(self, image: np.ndarray, annotations: List[Dict]) -> tuple:
        """Apply horizontal flip to image and adjust bounding boxes."""
        height, width = image.shape[:2]
        flipped_image = cv2.flip(image, 1)
        
        flipped_annotations = []
        for ann in annotations:
            bbox = ann['bbox'].copy()
            if ann['bbox_mode'] == BoxMode.XYXY_ABS:
                # x1, y1, x2, y2 format
                x1, y1, x2, y2 = bbox
                bbox = [width - x2, y1, width - x1, y2]
            elif ann['bbox_mode'] == BoxMode.XYWH_ABS:
                # x, y, w, h format
                x, y, w, h = bbox
                bbox = [width - x - w, y, w, h]
            
            new_ann = ann.copy()
            new_ann['bbox'] = bbox
            flipped_annotations.append(new_ann)
            
        return flipped_image, flipped_annotations
    
    def brightness_adjustment(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust brightness of the image."""
        if factor is None:
            factor = random.uniform(0.7, 1.3)
        
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        
        # Convert back to numpy array
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def contrast_adjustment(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust contrast of the image."""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
            
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def saturation_adjustment(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust saturation of the image."""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
            
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Color(pil_image)
        enhanced = enhancer.enhance(factor)
        
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def rotation(self, image: np.ndarray, annotations: List[Dict], angle: float = None) -> tuple:
        """Apply rotation to image and adjust bounding boxes."""
        if angle is None:
            angle = random.uniform(-10, 10)  # Small rotation to avoid major distortion
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation to image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Transform bounding boxes
        rotated_annotations = []
        for ann in annotations:
            bbox = ann['bbox'].copy()
            
            if ann['bbox_mode'] == BoxMode.XYXY_ABS:
                x1, y1, x2, y2 = bbox
                # Get all four corners of the bounding box
                corners = np.array([
                    [x1, y1, 1],
                    [x2, y1, 1], 
                    [x2, y2, 1],
                    [x1, y2, 1]
                ]).T
                
                # Apply rotation to corners
                rotated_corners = rotation_matrix @ corners
                
                # Get new bounding box
                x_coords = rotated_corners[0, :]
                y_coords = rotated_corners[1, :]
                
                new_x1 = max(0, min(x_coords))
                new_y1 = max(0, min(y_coords))
                new_x2 = min(width, max(x_coords))
                new_y2 = min(height, max(y_coords))
                
                # Skip if bounding box becomes invalid
                if new_x2 <= new_x1 or new_y2 <= new_y1:
                    continue
                    
                bbox = [new_x1, new_y1, new_x2, new_y2]
            
            new_ann = ann.copy()
            new_ann['bbox'] = bbox
            rotated_annotations.append(new_ann)
            
        return rotated_image, rotated_annotations
    
    def scale_jitter(self, image: np.ndarray, annotations: List[Dict], scale_factor: float = None) -> tuple:
        """Apply scale jittering to image and bounding boxes."""
        if scale_factor is None:
            scale_factor = random.uniform(0.9, 1.1)
            
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Resize image
        scaled_image = cv2.resize(image, (new_width, new_height))
        
        # If scaled image is larger, crop to original size
        if scale_factor > 1.0:
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            scaled_image = scaled_image[start_y:start_y + height, start_x:start_x + width]
            
            # Adjust bounding boxes for crop
            scaled_annotations = []
            for ann in annotations:
                bbox = ann['bbox'].copy()
                if ann['bbox_mode'] == BoxMode.XYXY_ABS:
                    x1, y1, x2, y2 = bbox
                    bbox = [x1 - start_x, y1 - start_y, x2 - start_x, y2 - start_y]
                    
                    # Check if bbox is still valid after cropping
                    if bbox[0] >= width or bbox[1] >= height or bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    
                    # Clip to image boundaries
                    bbox[0] = max(0, bbox[0])
                    bbox[1] = max(0, bbox[1])
                    bbox[2] = min(width, bbox[2])
                    bbox[3] = min(height, bbox[3])
                
                new_ann = ann.copy()
                new_ann['bbox'] = bbox
                scaled_annotations.append(new_ann)
        else:
            # If scaled image is smaller, pad to original size
            pad_x = (width - new_width) // 2
            pad_y = (height - new_height) // 2
            
            padded_image = np.zeros((height, width, 3), dtype=image.dtype)
            padded_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = scaled_image
            scaled_image = padded_image
            
            # Adjust bounding boxes for padding and scaling
            scaled_annotations = []
            for ann in annotations:
                bbox = ann['bbox'].copy()
                if ann['bbox_mode'] == BoxMode.XYXY_ABS:
                    x1, y1, x2, y2 = bbox
                    # Scale and then pad
                    bbox = [x1 * scale_factor + pad_x, y1 * scale_factor + pad_y,
                           x2 * scale_factor + pad_x, y2 * scale_factor + pad_y]
                
                new_ann = ann.copy()
                new_ann['bbox'] = bbox
                scaled_annotations.append(new_ann)
                
        return scaled_image, scaled_annotations
    
    def gaussian_noise(self, image: np.ndarray, noise_level: float = None) -> np.ndarray:
        """Add Gaussian noise to the image."""
        if noise_level is None:
            noise_level = random.uniform(0, 15)
            
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """Apply Gaussian blur to the image."""
        if kernel_size is None:
            kernel_size = random.choice([3, 5])
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def augment_single_image(self, image: np.ndarray, annotations: List[Dict]) -> List[tuple]:
        """
        Apply various augmentations to a single image.
        
        Returns:
            List of (augmented_image, augmented_annotations) tuples
        """
        augmented_samples = []
        
        # Original image (always include)
        augmented_samples.append((image.copy(), copy.deepcopy(annotations)))
        
        for i in range(self.augmentation_factor - 1):
            current_image = image.copy()
            current_annotations = copy.deepcopy(annotations)
            
            # Randomly select augmentations to apply
            augmentations_to_apply = random.sample([
                'horizontal_flip',
                'brightness',
                'contrast', 
                'saturation',
                'rotation',
                'scale_jitter',
                'gaussian_noise',
                'gaussian_blur'
            ], k=random.randint(2, 4))  # Apply 2-4 augmentations
            
            for aug_name in augmentations_to_apply:
                if aug_name == 'horizontal_flip' and random.random() < 0.5:
                    current_image, current_annotations = self.horizontal_flip(current_image, current_annotations)
                elif aug_name == 'brightness':
                    current_image = self.brightness_adjustment(current_image)
                elif aug_name == 'contrast':
                    current_image = self.contrast_adjustment(current_image)
                elif aug_name == 'saturation':
                    current_image = self.saturation_adjustment(current_image)
                elif aug_name == 'rotation' and random.random() < 0.3:  # Less frequent rotation
                    current_image, current_annotations = self.rotation(current_image, current_annotations)
                elif aug_name == 'scale_jitter' and random.random() < 0.4:
                    current_image, current_annotations = self.scale_jitter(current_image, current_annotations)
                elif aug_name == 'gaussian_noise' and random.random() < 0.3:
                    current_image = self.gaussian_noise(current_image)
                elif aug_name == 'gaussian_blur' and random.random() < 0.2:
                    current_image = self.gaussian_blur(current_image)
            
            # Only add if we still have valid annotations
            if current_annotations:
                augmented_samples.append((current_image, current_annotations))
        
        return augmented_samples


class AugmentedDatasetMapper(DatasetMapper):
    """
    Custom DatasetMapper that applies data augmentation when enabled in config.
    """
    
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        
        # Check if data augmentation is enabled
        self.use_augmentation = getattr(cfg, 'DATA_AUGMENTATION', False) and is_train
        
        if self.use_augmentation:
            # Get augmentation factor from config, default to 5
            augmentation_factor = getattr(cfg, 'AUGMENTATION_FACTOR', 5)
            self.augmenter = DataAugmentation(augmentation_factor=augmentation_factor)
            
            # Keep track of which augmented version to use for each image
            self._augmented_cache = {}
            self._current_indices = {}
            
            print(f"Data augmentation enabled with {augmentation_factor}x factor")
        else:
            print("Data augmentation disabled")
    
    def __call__(self, dataset_dict):
        """
        Apply data augmentation and return augmented samples.
        """
        if not self.use_augmentation:
            # Use the original DatasetMapper behavior
            return super().__call__(dataset_dict)
        
        # Create a unique key for this dataset item
        file_name = dataset_dict.get("file_name", "")
        image_id = dataset_dict.get("image_id", 0)
        item_key = (file_name, image_id)
        
        # Check if we've already processed this image
        if item_key not in self._augmented_cache:
            # Load the original image
            original_dict = copy.deepcopy(dataset_dict)
            
            # We need to apply augmentation to the raw image before detectron2 transforms
            image = detection_utils.read_image(original_dict["file_name"], format=self.image_format)
            detection_utils.check_image_size(original_dict, image)
            
            # Extract annotations in the correct format
            annotations = original_dict.get("annotations", [])
            
            # Convert annotations to the format expected by our augmenter
            aug_annotations = []
            for ann in annotations:
                aug_ann = {
                    'bbox': ann['bbox'],
                    'bbox_mode': ann['bbox_mode'],
                    'category_id': ann['category_id']
                }
                aug_annotations.append(aug_ann)
            
            # Apply our custom augmentation
            augmented_samples = self.augmenter.augment_single_image(image, aug_annotations)
            
            # Store the augmented samples
            processed_samples = []
            for aug_image, aug_anns in augmented_samples:
                # Create a modified dataset_dict for this augmented sample
                sample_dict = copy.deepcopy(original_dict)
                
                # Convert augmented annotations back to detectron2 format
                new_annotations = []
                for aug_ann in aug_anns:
                    new_ann = {
                        'bbox': aug_ann['bbox'],
                        'bbox_mode': aug_ann['bbox_mode'],
                        'category_id': aug_ann['category_id']
                    }
                    # Copy other annotation fields if they exist
                    for key in annotations[0].keys():
                        if key not in new_ann:
                            new_ann[key] = annotations[0][key]
                    new_annotations.append(new_ann)
                
                sample_dict["annotations"] = new_annotations
                processed_samples.append((aug_image, sample_dict))
            
            self._augmented_cache[item_key] = processed_samples
            self._current_indices[item_key] = 0
        
        # Get the current augmented sample to use
        cached_samples = self._augmented_cache[item_key]
        current_idx = self._current_indices[item_key] % len(cached_samples)
        
        # Update index for next time this image is requested
        self._current_indices[item_key] = current_idx + 1
        
        # Get the augmented image and corresponding dataset dict
        augmented_image, augmented_dict = cached_samples[current_idx]
        
        # Now apply the standard detectron2 pipeline to the augmented image
        # We need to temporarily replace the image in dataset_dict
        temp_file_name = augmented_dict["file_name"]
        
        # Apply standard detectron2 transformations
        aug_input = T.AugInput(augmented_image)
        transforms = self.augmentations(aug_input)
        transformed_image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        
        image_shape = transformed_image.shape[:2]  # h, w
        augmented_dict["image"] = torch.as_tensor(np.ascontiguousarray(transformed_image.transpose(2, 0, 1)))
        
        # Transform annotations using detectron2's method
        if "annotations" in augmented_dict:
            self._transform_annotations(augmented_dict, transforms, image_shape)
        
        return augmented_dict
