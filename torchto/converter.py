"""
File: converter.py
Author: Remy Vuagniaux
Date: January 12, 2024
Description: This file contain the implementation for the converter class.

Usage:
- This file contains a converter class capable of convert a torch model into different model format such as ONNX, TFLITE, KERAS and TFLITE optimized for EDGE TPU.

Dependencies:
- numpy, torch, tensorflow, keras, onnx2keras, onnx
"""

import os
import torch
import torch.nn as nn
import onnx
from onnx2keras import onnx_to_keras
import numpy as np
import warnings
from onnx_tf.backend import prepare
import tensorflow as tf
from keras.src.saving.serialization_lib import  enable_unsafe_deserialization
from torchto.utils import *


class Converter():
    def __init__(self,
                 conv_type,
                 dummy_input,
                 calibration_samples=None,
                 calibration_labels=None,
                 onnx_opset_version=12,
                 onnx_input_name='input',
                 onnx_output_name='output',
                 verbose=True,
                ):
        """
        Initialize the Converter class.

        Parameters:
        - conv_type (str): Conversion type, one of ['ONNX', 'TFLITE', 'TFLITE_UINT8', 'KERAS', 'EDGE_TPU'].
        - dummy_input (torch.Tensor): Dummy input tensor for the model (having the correct image shape and batch size of 1).
        - calibration_samples (numpy.array, optional): sample for calibration (required for quantization).
        - calibration_labels (numpy.array, optional): label for calibration (required for quantization).
        - onnx_opset_version (int, optional): ONNX opset version for exporting the ONNX model.
        - onnx_input_name (str, optional): Input name for the ONNX model.
        - onnx_output_name (str, optional): Output name for the ONNX model.
        - verbose (bool, optional): Whether to print verbose information during conversion.
        """
        # Validate the conversion type.
        if conv_type not in ['ONNX', 'TFLITE', 'TFLITE_UINT8', 'KERAS', 'EDGE_TPU']:
            raise ValueError(f"Unsupported conversion type: {conv_type}. Supported types are ['ONNX', 'TFLITE', 'TFLITE_UINT8', 'KERAS', 'EDGE_TPU']")

        # Validate calibration data for certain conversion types that require quantization.
        if conv_type in ['TFLITE_UINT8', 'EDGE_TPU'] and (calibration_samples is None or calibration_labels is None):
            raise ValueError("Quantization requires a calibration data loader. Please provide a valid calibration_dataloader.")

        # Initialize class attributes.
        self.conv_type = conv_type
        self.dummy_input = dummy_input.cpu()
        self.onnx_opset_version = onnx_opset_version
        self.onnx_input_name = onnx_input_name
        self.onnx_output_name = onnx_output_name
        self.verbose = verbose
        self.calibration_length = len(calibration_samples)
        self.calibration_dataset = calibration_samples
        self.calibration_labels = calibration_labels
      

    def __call__(self, model, dir='converted_model', name='model', test_conversion=True):
        """
        Convert the given model to the specified target format.

        Parameters:
        - model: The PyTorch model to be converted.
        - dir (str, optional): Directory to save the converted model files.
        - name (str, optional): Base name for the converted model files.
        - test_conversion (bool, optional): Whether to test the converted model's predictions against the original model.

        Returns:
        - converted_model: The converted model.
        - is_ok (bool): A flag indicating whether the conversion was successful.
        """
        is_ok = True
        input_model_type = check_model_type(model)
        if input_model_type == 'PyTorch':
            # Ensure the model is in evaluation mode on the CPU.
            model.eval().cpu()

        if input_model_type == self.conv_type:
            return model, True

        # Create the working directory if it doesn't exist.
        if not os.path.exists(dir):
            if self.verbose:
                print("Creating working directory: {}".format(dir))
            os.mkdir(dir)

        # Store the original CUDA_VISIBLE_DEVICES value and set it to -1 to ensure that conversion function happend on cpu.
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Convert the model based on the specified conversion type.
        if input_model_type == 'PyTorch':
            converted_model, conversion_success = self.from_torch_2_onnx(model, dir=dir, name=name, test_conversion=test_conversion)
            is_ok *= conversion_success
        else:
            converted_model = model

        if self.conv_type in ['TFLITE']:
            converted_model, conversion_success = self.from_onnx_2_tflite(converted_model, dir=dir, name=name, test_conversion=test_conversion)
            is_ok *= conversion_success

        if self.conv_type in ['TFLITE_UINT8']:
            converted_model, conversion_success = self.from_onnx_2_tflite_uint8(converted_model, dir=dir, name=name, test_conversion=test_conversion)
            is_ok *= conversion_success

        if self.conv_type in ['KERAS', 'EDGE_TPU']:
            converted_model, conversion_success = self.from_onnx_2_keras(converted_model, dir=dir, name=name, test_conversion=test_conversion)
            is_ok *= conversion_success

        if self.conv_type in ['EDGE_TPU']:
            converted_model, conversion_success = self.from_keras_2_tflite(converted_model, dir=dir, name=name, test_conversion=test_conversion)
            is_ok *= conversion_success

        # Restore the original CUDA_VISIBLE_DEVICES value.
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

        return converted_model, is_ok

        
    def from_torch_2_onnx(self, model, dir='converted_model', name='model', test_conversion=True):
        """
        Convert a PyTorch model to ONNX format.
    
        Parameters:
        - model: The PyTorch model to be converted.
        - dir (str, optional): Directory to save the converted ONNX model.
        - name (str, optional): Base name for the converted ONNX model.
        - test_conversion (bool, optional): Whether to test the converted model's predictions against the original model.
    
        Returns:
        - converted_model: The converted ONNX model.
        - is_ok (bool): A flag indicating whether the conversion was successful.
        """
        is_ok = True
    
        # Set the path for the ONNX model.
        model_path = os.path.join(dir, name + '.onnx')
    
        # Export the PyTorch model to ONNX format.
        torch.onnx.export(model, 
                          self.dummy_input, 
                          model_path,
                          opset_version=self.onnx_opset_version, 
                          input_names=[self.onnx_input_name],
                          output_names=[self.onnx_output_name],
                          export_params=True,
                          verbose=self.verbose
                         )
    
        # Load the converted ONNX model.
        converted_model = onnx.load(model_path)
    
        # Test the conversion if specified.
        if test_conversion:
            # Get the initial prediction from the original PyTorch model.
            initial_prediction = model(self.dummy_input).detach().numpy()
    
            # Perform inference using the converted ONNX model.
            converted_prediction = onnx_infer(model_path, self.dummy_input.numpy().astype(np.float32))
    
            # Check if predictions match within a tolerance.
            if np.sum((initial_prediction - converted_prediction)**2) > 1e-5 and self.verbose:
                warnings.warn("CONVERTED MODEL IS CORRUPTED: Prediction before conversion != after conversion")
                is_ok = False
    
        return converted_model, is_ok

        
    def from_onnx_2_tflite(self, onnx_model, dir='converted_model', name='model', test_conversion=True):
        """
        Convert an ONNX model to TensorFlow Lite (TFLite) format.
    
        Parameters:
        - onnx_model: The ONNX model to be converted.
        - dir (str, optional): Directory to save the converted TFLite model.
        - name (str, optional): Base name for the converted TFLite model.
        - test_conversion (bool, optional): Whether to test the converted model's predictions against the original model.
    
        Returns:
        - converted_model: The converted TFLite model.
        - is_ok (bool): A flag indicating whether the conversion was successful.
        """
        is_ok = True
    
        # Set the path for the TensorFlow model.
        model_path = os.path.join(dir, name)
    
        # Save the ONNX model as a TensorFlow model.
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(model_path)
    
        # Create a TFLite converter from the saved TensorFlow model.
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
        # Convert the TensorFlow model to TFLite format.
        converted_model = self._2_tflite(converter, 
                                         dir=dir, 
                                         name=name, 
                                         test_conversion=test_conversion)
    
        # Test the conversion if specified.
        if test_conversion:
            # Get the initial prediction from the original ONNX model.
            initial_prediction = onnx_infer(model_path + '.onnx', self.dummy_input.numpy().astype(np.float32))
    
            # Perform inference using the converted TFLite model.
            converted_prediction = tflite_infer(model_path + '.tflite', self.dummy_input.numpy().astype(np.float32))
    
            # Check if predictions match within a tolerance.
            if np.sum((initial_prediction - converted_prediction)**2) > 1e-5 and self.verbose:
                warnings.warn("CONVERTED MODEL IS CORRUPTED: Prediction before conversion != after conversion")
                is_ok = False
    
        return converted_model, is_ok


    def from_onnx_2_tflite_uint8(self, onnx_model, dir='converted_model', name='model', test_conversion=True):
        """
        Convert an ONNX model to quantized TensorFlow Lite (TFLite) format.
    
        Parameters:
        - onnx_model: The ONNX model to be converted.
        - dir (str, optional): Directory to save the converted quantized TFLite model.
        - name (str, optional): Base name for the converted quantized TFLite model.
        - test_conversion (bool, optional): Whether to test the converted model's predictions against the original model.
    
        Returns:
        - converted_model: The converted quantized TFLite model.
        - is_ok (bool): A flag indicating whether the conversion was successful.
        """
        is_ok = True
    
        # Set the path for the TensorFlow model.
        model_path = os.path.join(dir, name)
    
        # Save the ONNX model as a TensorFlow model.
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(model_path)
    
        # Create a TFLite converter from the saved TensorFlow model.
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
        # Convert the TensorFlow model to quantized TFLite format.
        converted_model = self._2_tflite(converter, 
                                         dir=dir, 
                                         name=name, 
                                         test_conversion=test_conversion, 
                                         quantize=True)
    
        # Test the conversion if specified.
        if test_conversion:
            # Get the initial prediction from the original ONNX model using calibration dataset.
            initial_prediction = onnx_infer(model_path + '.onnx', self.calibration_dataset)
    
            # Perform inference using the converted quantized TFLite model.
            converted_prediction = tflite_infer(model_path + '.tflite', self.calibration_dataset, quantized=True)
    
            # Calculate and compare accuracies.
            initial_acc = get_accuracy(initial_prediction, self.calibration_labels)
            converted_acc = get_accuracy(converted_prediction, self.calibration_labels)
    
            # Check if accuracy difference is significant and print a warning if needed.
            if abs(initial_acc - converted_acc) > 0.01 and self.verbose:
                warnings.warn("QUANTIZED MODEL HAS AN ACCURACY CHANGE: Accuracy before conversion: {}, after conversion: {}".format(int(initial_acc * 100), 
                                                                                                                                  int(converted_acc * 100)))
                is_ok = False
    
        return converted_model, is_ok

    def from_onnx_2_keras(self, onnx_model, dir='converted_model', name='model', test_conversion=True):
        """
        Convert an ONNX model to Keras format.
    
        Parameters:
        - onnx_model: The ONNX model to be converted.
        - dir (str, optional): Directory to save the converted Keras model.
        - name (str, optional): Base name for the converted Keras model.
        - test_conversion (bool, optional): Whether to test the converted model's predictions against the original model.
    
        Returns:
        - converted_model: The converted Keras model.
        - is_ok (bool): A flag indicating whether the conversion was successful.
        """
        is_ok = True
    
        # Set the path for the Keras model.
        model_path = os.path.join(dir, name)
    
        # Rename ONNX nodes and enable unsafe deserialization.
        rename_onnx_nodes(onnx_model)
        enable_unsafe_deserialization()
    
        # Convert ONNX model to Keras format.
        converted_model = onnx_to_keras(onnx_model, ['input'], change_ordering=True, name_policy="renumerate", verbose=self.verbose)
    
        # Save the Keras model.
        converted_model.save(model_path + '.keras')
    
        # Load the saved Keras model without compilation.
        converted_model = tf.keras.models.load_model(model_path + '.keras', compile=False)
    
        # Test the conversion if specified.
        if test_conversion:
            # Get the initial prediction from the original ONNX model.
            initial_prediction = onnx_infer(model_path + '.onnx', self.dummy_input.numpy().astype(np.float32))
    
            # Perform inference using the converted Keras model.
            converted_prediction = keras_infer(converted_model, self.dummy_input.numpy().astype(np.float32))
    
            # Check if predictions match within a tolerance.
            if np.sum((initial_prediction - converted_prediction)**2) > 1e-5 and self.verbose:
                warnings.warn("CONVERTED MODEL IS CORRUPTED: Prediction before conversion != after conversion")
                is_ok = False
    
        return converted_model, is_ok


    def from_keras_2_tflite(self, keras_model, dir='converted_model', name='model', test_conversion=True): 
        """
        Convert a Keras model to TensorFlow Lite (TFLite) format, optimized for EDGE TPU.
    
        Parameters:
        - keras_model: The Keras model to be converted.
        - dir (str, optional): Directory to save the converted TFLite model.
        - name (str, optional): Base name for the converted TFLite model.
        - test_conversion (bool, optional): Whether to test the converted model's predictions against the original model.
    
        Returns:
        - converted_model: The converted TFLite model.
        - is_ok (bool): A flag indicating whether the conversion was successful.
        """
        is_ok = True
    
        # Set the path for the TFLite model.
        model_path = os.path.join(dir, name)
    
        # Create a TFLite converter from the Keras model.
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
        # Convert the Keras model to TFLite format with quantization and channel first ordering.
        converted_model = self._2_tflite(converter, 
                                         dir=dir, 
                                         name=name, 
                                         test_conversion=test_conversion, 
                                         quantize=True, 
                                         channel_first=True)
    
        # Test the conversion if specified.
        if test_conversion:
            # Get the initial prediction from the original Keras model using calibration dataset.
            initial_prediction = keras_infer(keras_model, self.calibration_dataset)
    
            # Perform inference using the converted quantized TFLite model with channel first ordering.
            converted_prediction = tflite_infer(model_path + '.tflite', self.calibration_dataset, quantized=True, channel_first=True)
    
            # Calculate and compare accuracies.
            initial_acc = get_accuracy(initial_prediction, self.calibration_labels)
            converted_acc = get_accuracy(converted_prediction, self.calibration_labels)
    
            # Check if accuracy difference is significant and print a warning if needed.
            if abs(initial_acc - converted_acc) > 0.01 and self.verbose:
                warnings.warn("QUANTIZED MODEL HAS AN ACCURACY CHANGE: Accuracy before conversion: {}, after conversion: {}".format(int(initial_acc * 100), 
                                                                                                                                  int(converted_acc * 100)))
                is_ok = False
    
        return converted_model, is_ok


    def _2_tflite(self, converter, dir='converted_model', name='model', quantize=False, test_conversion=True, channel_first=False):
        """
        Convert a model using the given converter to TensorFlow Lite (TFLite) format.
    
        Parameters:
        - converter: The TFLite converter to be used for the conversion.
        - dir (str, optional): Directory to save the converted TFLite model.
        - name (str, optional): Base name for the converted TFLite model.
        - quantize (bool, optional): Whether to perform quantization during the conversion.
        - test_conversion (bool, optional): Whether to test the converted model's predictions against the original model.
        - channel_first (bool, optional): Whether the input data has channels first ordering.
    
        Returns:
        - converted_model: The converted TFLite model.
        """
        is_ok = True
    
        # Set the path for the TFLite model.
        model_path = os.path.join(dir, name)
    
        # Transpose calibration dataset if channel_first is True.
        cal_data = self.calibration_dataset.transpose([0, 2, 3, 1]) if channel_first else self.calibration_dataset
    
        # Configure converter settings for quantization.
        if quantize:
            def representative_data_gen():
                for input_value in tf.data.Dataset.from_tensor_slices(cal_data).batch(1).take(self.calibration_length):
                    # Model has only one input so each data point has one element.
                    yield [input_value]
    
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
    
        # Convert the model using the provided converter.
        converted_model = converter.convert()
    
        # Save the TFLite model.
        with open(model_path + '.tflite', "wb") as f:
            f.write(converted_model)
    
        return converted_model
            
            
        