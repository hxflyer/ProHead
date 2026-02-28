#!/usr/bin/env python3
"""
Test script to load Geo EXR file and print min/max values
"""

import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def test_geo_exr(exr_path):
    """
    Load Geo EXR file and print min/max values
    """
    print(f"Loading EXR file: {exr_path}")
    
    if not os.path.exists(exr_path):
        print(f"File not found: {exr_path}")
        return
    
    # Try multiple loading methods
    img_array = None
    
    # Method 1: Try OpenEXR
    try:
        import OpenEXR
        import Imath
        
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()
        
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        channels_data = {}
        for channel in ['R', 'G', 'B', 'A']:
            if channel in header['channels']:
                channel_type = header['channels'][channel].type
                if channel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                    pixel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
                    channel_array = np.frombuffer(pixel_data, dtype=np.float32).reshape((height, width))
                else:
                    pixel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.HALF))
                    channel_array = np.frombuffer(pixel_data, dtype=np.float16).reshape((height, width))
                channels_data[channel] = channel_array
        
        if channels_data:
            img_array = np.stack([channels_data.get(ch, np.zeros((height, width))) 
                                for ch in ['R', 'G', 'B', 'A']], axis=2)
            print("Loaded with OpenEXR library")
    except ImportError:
        print("OpenEXR library not available, trying alternatives...")
    except Exception as e:
        print(f"OpenEXR loading failed: {e}")
    
    # Method 2: Try imageio
    if img_array is None:
        try:
            import imageio.v3 as imageio
            img_array = imageio.imread(exr_path)
            print("Loaded with imageio")
        except Exception as e:
            print(f"imageio loading failed: {e}")
    
    # Method 3: Try PIL
    if img_array is None:
        try:
            from PIL import Image
            img = Image.open(exr_path)
            img_array = np.array(img)
            print("Loaded with PIL")
        except Exception as e:
            print(f"PIL loading failed: {e}")
    
    if img_array is None:
        print("All loading methods failed!")
        return
    
    # Print results
    print(f"\nImage Info:")
    print(img_array)
    print(f"  Shape: {img_array.shape}")
    print(f"  Data type: {img_array.dtype}")
    print(f"  Min value: {img_array.min()}")
    print(f"  Max value: {img_array.max()}")
    
    # Per-channel analysis if multi-channel
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        channels = ['Red', 'Green', 'Blue', 'Alpha'] if img_array.shape[2] == 4 else ['Red', 'Green', 'Blue']
        print(f"\nPer-channel min/max:")
        for i, channel_name in enumerate(channels[:img_array.shape[2]]):
            channel_data = img_array[:, :, i]
            print(f"  {channel_name}: min={channel_data.min()}, max={channel_data.max()}")

if __name__ == "__main__":
    # Test with a Geo EXR file
    test_file = r"G:\training1\Geo_0_0.exr"
    test_geo_exr(test_file)
