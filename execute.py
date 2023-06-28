from image import Image
import transform as tr
import numpy as np
import inspect

def performoperations(func_name):
    func = getattr(tr, func_name, None)
    if not callable(func):
        print("Invalid function name.")
        return   
    signature=inspect.signature(func)
    parameters=signature.parameters
    Arg_names=[]
    l=0
    for param in parameters.values():
        Arg_names.append(param.name)
    inputs=[]
    for param in Arg_names:
        if param.startswith('im'):
            try:
                file1=input(str("enter the name of the image in png format:"))            
                im=Image(filename=file1+'.png')
                value=im
            except FileNotFoundError:
                print("Oops!There is no file named", file1+'.png')
                print("Make Sure that the file exists in the input directory and in the png format.")
                return
            inputs.append(im)
        elif param.startswith('fac') or param.startswith('alp') or param.startswith('mid') or  param.startswith('ang') or param.startswith('compress'):
            value = input(f"Enter the value for {param}, Example 3.2: ")
            inputs.append(float(value))
        elif param.startswith('kernel_s')or param.startswith('x_start')  or param.startswith('y_start') or param.startswith('width') or param.startswith('height')or param.startswith('new_width') or param.startswith('thick') or param.startswith('new_height') or param.startswith('num_cl'):
            value = input(f"Enter the value for {param} , Example: 3(integer):")
            inputs.append(int(value))
        elif param.startswith('color_p'):
            print(f"Enter the values for {param} in the given format : [(255, 0, 0), (0, 255, 0), (0, 0, 255)]")
            value=[]
            for i in range(3):
                x=input(f"first value of the the tuple number:")
                y=input(f"second value of the the tuple number:")
                z=input(f"third value of the the tuple number:")
                
                tup=(x,y,z)
                value.append(tup)
            inputs.append(value)
        elif param.startswith('save'):
            value = file1+'.png'
            inputs.append(value)
            l=1
        elif param=='color':
            print(f"Enter the value for {param} in the given format : (255, 0, 0)")
            x=int(input(f"first value of the the tuple number:"))
            y=int(input(f"second value of the the tuple number:"))
            z=int(input(f"third value of the the tuple number:"))
            tup=(x,y,z)
            inputs.append(tup)
        elif param=='kernel':
            print(f"Enter the values for {param} in the given format : [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]")
            value=[]
            for i in range(3):
                x=int(input(f"first value of the the list number:"))
                y=int(input(f"second value of the the list number:"))
                z=int(input(f"third value of the the list number:"))
                    
                s_list=[x,y,z]
                value.append(s_list)
            value=np.array(value)
            inputs.append(value)
        elif param=='x' or param=='y':
            value=[]
            x=int(input(f"Enter the first value of the {param}"))
            y=int(input(f"Enter the second value of the {param}"))
            value.append(x)
            value.append(y)
            inputs.append(value)
    try:
        if inputs:
            if l==0:
                im = func(*inputs)
                im.write_image(func.__name__ + '_' + file1 + '.png')
            else:
                im = func(*inputs)
        else:
             print("No inputs provided.")
             return
    except:
        for i, input_arg in enumerate(inputs):
            if isinstance(input_arg, Image):
                inputs[i] = input_arg.array
        im = Image(array=func(*inputs))
        im.write_image(func.__name__ + '_' + file1 + '.png')
def available_functions():
    functions = [name for name, obj in inspect.getmembers(tr) if inspect.isfunction(obj)]
    for function in functions:
        print(function,end=" \t")
if __name__ =='__main__':
    available_functions()
    funcname=input("enter the name of the fucntion you want to execute: ")
    performoperations(funcname)