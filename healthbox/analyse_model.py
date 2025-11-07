import torch
import time
import numpy as np
import psutil
import os
from typing import Dict, Any, Tuple, Optional, Union
from PIL import Image
import json
from datetime import datetime
import pynvml
import thop  # For FLOPs calculation

def analyze_model(
        model_instance,
        sample_input: Union[Image.Image, str, np.ndarray, torch.Tensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        num_runs: int = 10,
        warmup_runs: int = 3,
        save_path: Optional[str] = None,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze a model instance and generate an ID card with performance metrics.

    Args:
        model_instance: An instance of a model with a .predict() method.
        sample_input: A sample input for the model (PIL image, path, numpy array, or tensor).
        input_shape: Shape of input to create if sample_input is None.
        num_runs: Number of runs for speed measurement.
        warmup_runs: Number of warmup runs before timing.
        save_path: Path to save the analysis results as JSON.
        verbose: Whether to print the analysis results.

    Returns:
        Dictionary containing all model analysis metrics.
    """
    # Check if the model has a predict method
    if not hasattr(model_instance, 'predict') or not callable(model_instance.predict):
        raise TypeError("The provided instance must have a callable 'predict' method.")

    # Get device from the model instance if available
    device = getattr(model_instance, 'device', torch.device("cpu"))

    # Initialize results dictionary
    results = {
            "model_name": model_instance.__class__.__name__,
            "device": str(device),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Determine model type and create sample input if not provided
    model_type = _determine_model_type(model_instance)
    results["model_type"] = model_type

    if sample_input is None:
        if input_shape is None:
            # Default to a common image size
            input_shape = (3, 224, 224)
        sample_input = _create_sample_input(input_shape, model_type)

    # 1. Model Complexity Analysis
    if verbose:
        print("Analyzing model complexity...")

    complexity_results = _analyze_model_complexity(model_instance, device, sample_input, verbose)
    results.update(complexity_results)

    # 2. Inference Speed Analysis
    if verbose:
        print("Measuring inference speed...")

    speed_results = _measure_inference_speed(model_instance, sample_input, num_runs, warmup_runs, device)
    results.update(speed_results)

    # 3. Memory Usage Analysis
    if verbose:
        print("Analyzing memory usage...")

    memory_results = _analyze_memory_usage(model_instance, sample_input, device, verbose)
    results.update(memory_results)

    # 4. Model-Specific Information
    if verbose:
        print("Collecting model-specific information...")

    model_info = _collect_model_info(model_instance)
    results["model_info"] = model_info

    # 5. Generate Model ID Card
    if verbose:
        _print_model_id_card(results)

    # Save results if path provided
    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Analysis results saved to {save_path}")

    return results

def _determine_model_type(model_instance) -> str:
    """Determine if the model is for classification, detection, segmentation, etc."""
    class_name = model_instance.__class__.__name__.lower()

    if "classifier" in class_name or "classification" in class_name:
        return "Classification"
    elif "detection" in class_name or "detector" in class_name:
        return "Object Detection"
    elif "segmentation" in class_name or "segmenter" in class_name:
        return "Segmentation"
    else:
        # Try to infer from the output of predict
        try:
            # Create a minimal input
            if hasattr(model_instance, 'device'):
                device = model_instance.device
            else:
                device = torch.device("cpu")

            # Create a small dummy input based on the model type
            if hasattr(model_instance, 'transforms'):
                # For models with transforms (like classifiers)
                dummy_tensor = torch.rand((1, 3, 224, 224))
                dummy_pil = Image.fromarray((dummy_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype('uint8'))
                sample_input = dummy_pil
            else:
                # For models without transforms (like YOLO)
                sample_input = np.zeros((224, 224, 3), dtype=np.uint8)

            # Get a sample output
            with torch.no_grad():
                output = model_instance.predict(sample_input)

            # Analyze the output structure
            if isinstance(output, dict):
                if "class" in output and "confidence" in output:
                    return "Classification"
                elif "detections" in output or "boxes" in output:
                    return "Object Detection"
                elif "mask" in output or "segmentation" in output:
                    return "Segmentation"

            return "Unknown"
        except Exception:
            return "Unknown"

def _create_sample_input(input_shape: Tuple[int, ...], model_type: str) -> Union[Image.Image, np.ndarray]:
    """Create a sample input based on the model type and input shape."""
    if model_type == "Object Detection":
        # For detection models, create a numpy array
        return np.zeros((input_shape[1], input_shape[2], input_shape[0]), dtype=np.uint8)
    else:
        # For classification and other models, create a PIL image
        dummy_tensor = torch.rand(input_shape)
        return Image.fromarray((dummy_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

def _analyze_model_complexity(model_instance, device, sample_input, verbose=True) -> Dict[str, Any]:
    """Analyze model complexity including parameters and FLOPs."""
    results = {}

    # Check if the model has a 'model' attribute (like in the classifier examples)
    model = getattr(model_instance, 'model', None)

    if model:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results["model_complexity"] = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": total_params - trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }

        # Calculate FLOPs using thop
        try:
            # Create a tensor input for thop
            if isinstance(sample_input, Image.Image):
                # Convert PIL to tensor
                transform = getattr(model_instance, 'transforms', None)
                if transform:
                    tensor_input = transform(sample_input).unsqueeze(0)
                else:
                    # Basic transform if none available
                    tensor_input = torch.nn.functional.interpolate(
                            torch.from_numpy(np.array(sample_input)).permute(2, 0, 1).float().unsqueeze(0),
                            size=(224, 224)
                    )
            elif isinstance(sample_input, np.ndarray):
                # Convert numpy to tensor
                tensor_input = torch.from_numpy(sample_input).permute(2, 0, 1).float().unsqueeze(0)
            else:
                # Assume it's already a tensor
                tensor_input = sample_input

            tensor_input = tensor_input.to(device)

            flops, params = thop.profile(model, inputs=(tensor_input,), verbose=False)
            results["model_complexity"]["flops"] = flops
            results["model_complexity"]["flops_gflops"] = flops / 1e9
        except Exception as e:
            if verbose:
                print(f"Could not calculate FLOPs: {e}")
            results["model_complexity"]["flops"] = None
            results["model_complexity"]["flops_gflops"] = None
    else:
        results["model_complexity"] = {"error": "Model instance has no 'model' attribute."}

    return results

def _measure_inference_speed(model_instance, sample_input, num_runs, warmup_runs, device) -> Dict[str, Any]:
    """Measure inference speed of the model."""
    # Warmup runs
    for _ in range(warmup_runs):
        _ = model_instance.predict(sample_input)

    # Synchronize CUDA if applicable
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure inference time
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model_instance.predict(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time_ms = np.mean(times) * 1000
    std_time_ms = np.std(times) * 1000

    return {
            "inference_speed": {
                    "average_time_ms": round(avg_time_ms, 3),
                    "std_time_ms": round(std_time_ms, 3),
                    "min_time_ms": round(np.min(times) * 1000, 3),
                    "max_time_ms": round(np.max(times) * 1000, 3),
                    "throughput_fps": round(1000 / avg_time_ms, 2)
            }
    }

def _analyze_memory_usage(model_instance, sample_input, device, verbose=True) -> Dict[str, Any]:
    """Analyze memory usage of the model."""
    if device.type == "cuda":
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            _ = model_instance.predict(sample_input)

            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

            return {
                    "memory_usage": {
                            "gpu_total_memory_gb": meminfo.total / (1024 ** 3),
                            "gpu_used_memory_before_mb": meminfo.used / (1024 ** 2),
                            "gpu_peak_memory_during_inference_mb": round(max_memory, 2),
                            "gpu_memory_utilization_percent": round((max_memory / (meminfo.total / (1024 ** 2))) * 100, 2)
                    }
            }
        except Exception as e:
            if verbose:
                print(f"Could not analyze GPU memory: {e}")
            return {"memory_usage": {"error": str(e)}}
    else:
        # System memory for CPU
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB

        _ = model_instance.predict(sample_input)

        mem_after = process.memory_info().rss / (1024 ** 2)  # MB

        return {
                "memory_usage": {
                        "system_memory_before_mb": mem_before,
                        "system_memory_after_mb": mem_after,
                        "system_memory_delta_mb": round(mem_after - mem_before, 2)
                }
        }

def _collect_model_info(model_instance) -> Dict[str, Any]:
    """Collect model-specific information."""
    model_info = {
            "class_name": model_instance.__class__.__name__,
            "module_name": model_instance.__class__.__module__,
    }

    # Try to get specific information based on model type
    if hasattr(model_instance, 'NUM_CLASSES') and hasattr(model_instance, 'CLASS_NAMES'):
        # For classification models
        model_info["num_classes"] = model_instance.NUM_CLASSES
        model_info["class_names"] = model_instance.CLASS_NAMES
    elif hasattr(model_instance, 'model_path'):
        # For models with a path attribute
        model_info["model_path"] = model_instance.model_path

    # Try to get more specific info if it's a timm model
    if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'pretrained_cfg'):
        model_info["timm_config"] = model_instance.model.pretrained_cfg

    return model_info

def _print_model_id_card(results: Dict[str, Any]):
    """Print a formatted model ID card."""
    print("\n" + "="*50)
    print("MODEL ID CARD")
    print("="*50)
    print(f"Name: {results['model_name']}")
    print(f"Type: {results['model_type']}")
    print(f"Device: {results['device']}")
    print(f"Analysis Date: {results['analysis_date']}")

    if "model_complexity" in results and "error" not in results["model_complexity"]:
        print("\nModel Complexity:")
        print(f"  Total Parameters: {results['model_complexity']['total_parameters']:,}")
        print(f"  Model Size: {results['model_complexity']['model_size_mb']:.2f} MB")
        if results['model_complexity']['flops'] is not None:
            print(f"  FLOPs: {results['model_complexity']['flops_gflops']:.2f} GFLOPs")

    print("\nInference Speed:")
    print(f"  Average Time: {results['inference_speed']['average_time_ms']:.3f} ms")
    print(f"  Throughput: {results['inference_speed']['throughput_fps']:.2f} FPS")

    print("\nMemory Usage:")
    if "gpu_total_memory_gb" in results["memory_usage"]:
        print(f"  GPU Total Memory: {results['memory_usage']['gpu_total_memory_gb']:.2f} GB")
        print(f"  Peak Memory During Inference: {results['memory_usage']['gpu_peak_memory_during_inference_mb']:.2f} MB")
        print(f"  GPU Memory Utilization: {results['memory_usage']['gpu_memory_utilization_percent']:.2f}%")
    else:
        print(f"  System Memory Delta: {results['memory_usage']['system_memory_delta_mb']:.2f} MB")

    print("\nModel Information:")
    if "num_classes" in results["model_info"]:
        print(f"  Number of Classes: {results['model_info']['num_classes']}")
        print(f"  Class Names: {results['model_info']['class_names']}")
    if "model_path" in results["model_info"]:
        print(f"  Model Path: {results['model_info']['model_path']}")

    print("="*50)

def compare_models(
        models: Dict[str, Any],
        sample_inputs: Optional[Dict[str, Union[Image.Image, str, np.ndarray, torch.Tensor]]] = None,
        input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        num_runs: int = 10,
        save_path: Optional[str] = None,
        verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models and generate a comparison report.

    Args:
        models: Dictionary of model name to model instance
        sample_inputs: Dictionary of model name to sample input (optional)
        input_shapes: Dictionary of model name to input shape (used if sample_inputs is None)
        num_runs: Number of runs for speed measurement
        save_path: Path to save the comparison results as JSON
        verbose: Whether to print the comparison results

    Returns:
        Dictionary containing analysis results for each model
    """
    results = {}

    for name, model in models.items():
        if verbose:
            print(f"\nAnalyzing model: {name}")
            print("-" * 50)

        # Get sample input or input shape for this model
        sample_input = sample_inputs.get(name) if sample_inputs else None
        input_shape = input_shapes.get(name) if input_shapes else None

        results[name] = analyze_model(
                model_instance=model,
                sample_input=sample_input,
                input_shape=input_shape,
                num_runs=num_runs,
                verbose=False
        )

    # Generate comparison table
    if verbose:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        # Header
        print(f"{'Model':<20} {'Type':<15} {'Params':<12} {'Size (MB)':<10} {'Time (ms)':<10} {'FPS':<8} {'Memory (MB)':<12}")
        print("-" * 80)

        # Rows
        for name, result in results.items():
            model_type = result['model_type']
            params = result["model_complexity"].get("total_parameters", "N/A")
            size_mb = result["model_complexity"].get("model_size_mb", "N/A")
            time_ms = result["inference_speed"]["average_time_ms"]
            fps = result["inference_speed"]["throughput_fps"]

            if isinstance(params, int): params = f"{params:,}"
            if isinstance(size_mb, float): size_mb = f"{size_mb:.2f}"

            mem_val = "N/A"
            if "gpu_peak_memory_during_inference_mb" in result["memory_usage"]:
                mem_val = f"{result['memory_usage']['gpu_peak_memory_during_inference_mb']:.2f}"
            elif "system_memory_delta_mb" in result["memory_usage"]:
                mem_val = f"{result['memory_usage']['system_memory_delta_mb']:.2f}"

            print(f"{name:<20} {model_type:<15} {params:<12} {size_mb:<10} {time_ms:<10.3f} {fps:<8.2f} {mem_val:<12}")

        print("="*80)

    # Save results if path provided
    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Comparison results saved to {save_path}")

    return results