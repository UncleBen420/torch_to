"""
File: utils.py
Author: Remy Vuagniaux
Date: January 12, 2024
Description: This file contain utils function that help for the conversion of models..

Dependencies:
- numpy, tensorflow, onnxruntime, onnx
"""

import os
import onnx
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch

def check_model_type(model):
    if isinstance(model, torch.nn.Module):
        return "PyTorch"
    elif isinstance(model, onnx.ModelProto):
        return "ONNX"
    else:
        raise ValueError("Unknown model type.")

def keras_infer(k_model, inputs):
    """
    Perform inference using a Keras model.

    Parameters:
    - k_model (keras.Model): The Keras model for inference.
    - inputs (numpy.ndarray): Input data for inference, with shape (batch_size, height, width, channels).

    Returns:
    - numpy.ndarray: Model predictions.

    Note:
    - The 'inputs' array is transposed to match the expected input format of the Keras model.
    """
    # Transpose the input data to match the channel first format of the Keras model.
    # The original shape is assumed to be (batch_size, channels, height, width).
    inputs = inputs.transpose([0, 2, 3, 1])

    # Perform inference using the Keras model.
    predictions = k_model(inputs)

    return predictions


def onnx_infer(onnx_model_path, inputs):
    """
    Perform inference using an ONNX model.

    Parameters:
    - onnx_model_path (str): Path to the ONNX model file.
    - inputs (numpy.ndarray): Array of input data for inference, where each element has shape (channels, height, width).

    Returns:
    - numpy.ndarray: Model predictions.

    Note:
    - The function uses the ONNX Runtime to load the model and perform inference.
    - Input data is expected to be a list of numpy arrays, each representing an image.
    """
    # Load the ONNX model using ONNX Runtime.
    session = ort.InferenceSession(onnx_model_path)

    # List to store the model predictions for each input image.
    preds = []
    for image in inputs:
        input_name = session.get_inputs()[0].name

        # Expand the dimensions of the input image to match the model's expected input shape. (BCHW)
        expanded_image = np.expand_dims(image, axis=0)

        # Run inference for the current input image and retrieve the predictions.
        preds.append(np.array(session.run(None, {input_name: expanded_image})).reshape(-1))

    return np.array(preds)

    
def tflite_infer(tflite_model_path,
                 inputs,
                 quantized=False,
                 channel_first=False
                ):
    """
    Perform inference using a TensorFlow Lite (TFLite) model.

    Parameters:
    - tflite_model_path (str): Path to the TFLite model file.
    - inputs (numpy.ndarray): Array of input data for inference, where each element has shape (channels, height, width).
    - quantized (bool, optional): Whether the model is quantized. Defaults to False.
    - channel_first (bool, optional): Whether input data follows the channel-first format. Defaults to False.

    Returns:
    - numpy.ndarray: Model predictions.

    Note:
    - The function uses TensorFlow Lite Interpreter to load the model and perform inference.
    - Input data is expected to be a list of numpy arrays, each representing an image.
    - If 'quantized' is True, input data is assumed to be quantized, and appropriate normalization is applied.
    - If 'channel_first' is True, input data is transposed to channel-first format before inference.
    """
    # Create a TensorFlow Lite Interpreter and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get details of the input tensor and quantization parameters.
    input_details = interpreter.get_input_details()
    params = input_details[0]['quantization_parameters']

    scale = params['scales']
    zero_point = params['zero_points']

    # List to store the model predictions for each input image.
    preds = []
    for image in inputs:
        # Normalize input data if it is quantized.
        if quantized:
            normalized_input = image / scale + zero_point
            np.clip(normalized_input, 0, 255, out=normalized_input)
            input_data = normalized_input
            input_tensor = tf.convert_to_tensor(input_data, dtype=tf.uint8)
        else:
            input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

        # Expand dimensions of the input tensor to match the model's expected input shape. (BCHW)
        input_tensor = tf.expand_dims(input_tensor, 0)

        # Transpose input into BHWC tensor if channel-first format is used.
        if channel_first:
            input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])

        # Set input tensor for the interpreter and invoke the model.
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        # Retrieve and flatten the model predictions and append to the list.
        preds.append(interpreter.get_tensor(interpreter.get_output_details()[0]['index']).flatten())

    # Normalize predictions if the model is quantized.
    return np.array(preds) / 255. if quantized else np.array(preds)

def rename_onnx_nodes(onnx_model):
    """
    Rename nodes and initializers in an ONNX model. This all correct conversion from pytorch to keras.

    Parameters:
    - onnx_model (onnx.ModelProto): The ONNX model to be modified in-place.

    Note:
    - This function removes leading slashes ('/') from node names, input names, and initializer names in the ONNX model.
    """
    # Iterate through each node in the ONNX model's graph.
    for node in onnx_model.graph.node:
        # Remove leading slash from node name.
        if node.name.startswith("/"):
            node.name = node.name[1:]

        # Remove leading slash from input names.
        for i in range(len(node.input)):
            input_v = node.input[i]
            if input_v.startswith("/"):
                node.input[i] = input_v[1:]

        # Remove leading slash from output names.
        for i in range(len(node.output)):
            out_v = node.output[i]
            if out_v.startswith("/"):
                node.output[i] = out_v[1:]

    # Iterate through each initializer in the ONNX model's graph.
    for initializer in onnx_model.graph.initializer:
        # Remove leading slash from initializer name.
        if initializer.name.startswith("/"):
            initializer.name = initializer.name[1:]


def softmax(x):
    """
    Compute softmax values for each set of scores in x.

    Parameters:
    - x (numpy.ndarray): Input array representing raw logits.

    Returns:
    - numpy.ndarray: Softmax probabilities for each set of logits.
    """
    # Shift input values to avoid numerical instability.
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    
    # Compute softmax probabilities.
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def get_accuracy(preds, labels):
    """
    Compute accuracy based on model predictions and true labels.

    Parameters:
    - preds (numpy.ndarray): Model predictions.
    - labels (numpy.ndarray): True labels.

    Returns:
    - float: Accuracy of the model predictions.
    """
    # Convert model predictions to class indices using argmax and apply softmax.
    preds = np.argmax(softmax(preds), axis=1, keepdims=True)

    # Count the number of correct predictions.
    total_correct = np.sum(labels == preds)

    # Calculate the total number of samples.
    total_samples = len(labels)

    # Calculate and return the accuracy.
    return total_correct / total_samples