import inspect
import numpy as np
from cv2image import ImageProcessor
import cv2transform as trf
def execute_image_function(function_name):
    # Create an instance of the ImageProcessor
    processor = ImageProcessor()
    try:
        file=str(input("enter the name of the image:"))+'.png'
    except:
        print("image not found!")
    # Get the function object based on the provided function name
    function = getattr(trf, function_name, None)
    
    # Get the function's parameter names and types
    parameters = inspect.signature(function).parameters

    # Prompt the user for input values of the function's parameters
    inputs = []
    for param_name, param_info in parameters.items():
        param_type = param_info.annotation.__name__
        if param_name.startswith('im'):
            image = processor.load_image(file)
            inputs.append(image)
        elif param_type == 'ndarray' and not param_name.startswith('im'):
            user_input = input(f"Enter the value for '{param_name}' ({param_type}): ")
            converted_input = np.array([float(val) for val in user_input.split()])
            inputs.append(converted_input)
        elif param_type=='list':
            user_input = input(f"Enter the value for '{param_name}' ({(param_type)}): ")
            converted_input = np.array([float(val) for val in user_input.split()])
            converted_input.tolist()
            inputs.append(converted_input)
        elif param_type=='tuple':
            user_input = input(f"Enter the value for '{param_name}' ({(param_type)}): ")
            converted_input = np.array([float(val) for val in user_input.split()])
            converted_input.tolist()
            inputs.append(tuple(converted_input))
        else:
            user_input = input(f"Enter the value for '{param_name}' ({(param_type)}): ")
            # Convert the user input to the expected parameter type
            converted_input = param_info.annotation(user_input)
            inputs.append(converted_input)

    # Execute the function with the provided input values
    result = function(*inputs)

    # Save the resulting image
    if result is not None:
        output_filename = f"{function_name}_output.png"
        processor.save_image(result, output_filename)
        print(f"Result image saved as '{output_filename}'.")

def available_functions():
    functions = [name for name, obj in inspect.getmembers(trf) if inspect.isfunction(obj)]
    for function in functions:
        print(function,end=" \t")

if __name__ =='__main__':
    available_functions()
    funcname=input("\n enter the name of the fucntion you want to execute: ")
    execute_image_function(funcname)

