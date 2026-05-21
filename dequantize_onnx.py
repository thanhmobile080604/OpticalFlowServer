import onnx
import numpy as np
from onnx import numpy_helper
import sys


def dequantize_model(input_path, output_path):
    model = onnx.load(input_path)
    graph = model.graph

    init_map = {init.name: init for init in graph.initializer}

    nodes_to_remove = []
    name_replacements = {}

    for node in list(graph.node):
        if node.op_type == 'DequantizeLinear':
            # Expect inputs: [x, scale, zero_point]
            if len(node.input) < 2:
                continue
            x_name = node.input[0]
            scale_name = node.input[1]
            zp_name = node.input[2] if len(node.input) > 2 else None

            if x_name not in init_map:
                # Can't fold if input isn't an initializer
                continue
            if scale_name not in init_map:
                continue
            if zp_name is not None and zp_name not in init_map:
                continue

            q_init = init_map[x_name]
            scale_init = init_map[scale_name]
            zp_init = init_map[zp_name] if zp_name is not None else None

            q_arr = numpy_helper.to_array(q_init)
            scale_arr = numpy_helper.to_array(scale_init).astype(np.float32)
            if zp_init is not None:
                zp_arr = numpy_helper.to_array(zp_init).astype(np.float32)
            else:
                zp_arr = 0.0

            # Compute dequantized
            try:
                dq = (q_arr.astype(np.float32) - zp_arr) * scale_arr
            except Exception:
                # Try broadcasting explicitly
                dq = q_arr.astype(np.float32)
                dq = dq - zp_arr
                dq = dq * scale_arr

            new_name = node.output[0] + '_dequantized'
            new_init = numpy_helper.from_array(dq.astype(np.float32), name=new_name)

            # Add new initializer
            graph.initializer.append(new_init)
            init_map[new_name] = new_init

            # Replace all occurrences of node.output[0] in graph with new_name
            old_out = node.output[0]
            name_replacements[old_out] = new_name

            nodes_to_remove.append(node)

    # Apply replacements
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in name_replacements:
                node.input[i] = name_replacements[inp]
        for i, out in enumerate(node.output):
            if out in name_replacements:
                node.output[i] = name_replacements[out]

    # Remove handled DequantizeLinear nodes
    for rem in nodes_to_remove:
        graph.node.remove(rem)

    # (Optional) remove any unused initializers that were the original quantized tensors
    used = set()
    for node in graph.node:
        for inp in node.input:
            used.add(inp)
        for out in node.output:
            used.add(out)
    # Keep graph outputs as used
    for out in graph.output:
        used.add(out.name)

    new_inits = []
    for init in graph.initializer:
        if init.name in used or init.name.endswith('_dequantized'):
            new_inits.append(init)
    graph.initializer[:] = new_inits

    onnx.save(model, output_path)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python dequantize_onnx.py input.onnx output.onnx')
        sys.exit(1)
    dequantize_model(sys.argv[1], sys.argv[2])
    print('Saved dequantized model to', sys.argv[2])
