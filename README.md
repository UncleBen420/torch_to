# Converter.py

**File:** converter.py  
**Author:** Remy Vuagniaux  
**Date:** January 12, 2024  
**Description:** This file contains the implementation for the `Converter` class.

## Usage

This file contains a `Converter` class capable of converting a torch model into different model formats such as ONNX, TFLITE, KERAS, and TFLITE optimized for EDGE TPU.

## Dependencies

- numpy
- torch
- tensorflow
- keras
- onnx2keras
- onnx

## Implementation Details

The `Converter` class in this file provides functionality to convert a PyTorch model into various formats. It supports the following conversion types:

- ONNX
- TFLITE
- TFLITE_UINT8
- KERAS
- EDGE_TPU

The class includes options for specifying a dummy input, calibration data for quantization, ONNX opset version, input and output names, and more.

## Usage Example

Here is a simple example of how to use the `Converter` class:

```python
from converter import Converter

# Create a dummy input tensor (replace with your actual input shape and values)
dummy_input = ...

# Create an instance of the Converter class
converter = Converter(conv_type='ONNX', dummy_input=dummy_input)

# Convert a PyTorch model
model_to_convert = ...

# Specify the directory and name for the converted model
converted_model, success = converter(model_to_convert, dir='converted_models', name='converted_model')

# Check if the conversion was successful
if success:
    print("Model converted successfully!")
else:
    print("Conversion failed. Please check the warnings for details.")
