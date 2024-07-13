import numpy as np

# Initialize a zero matrix with dimensions (108, 16)
x = np.zeros((108, 16), dtype=np.float32)


def is_number(s):
    """
    Check if the string s is a number.
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


# Loop through each sample
for sample in range(1, 103059):
    # Process keypoint data
    with open(f"../dataset/keypoint/keypoint{sample}.txt", 'r') as k:
        keypoint_index = 0
        res_keypoint = k.read()
        keypoint_pre_data = res_keypoint[309:]
        keypoint_medi_data = keypoint_pre_data.split()
        add_keypoint = [float(keypoint) for keypoint in keypoint_medi_data]

        for j in range(len(add_keypoint)):
            if j > 6:
                x[keypoint_index][j - 7] = add_keypoint[j]
            else:
                x[keypoint_index][j] = add_keypoint[j]
            if j == 6:
                keypoint_index += 1

    # Process line data
    with open(f"../dataset/line/line{sample}.txt", 'r') as l:
        line_index = 2
        res_line = l.read()
        line_pre_data = res_line[310:]
        line_medi_data = line_pre_data.split()
        add_line = [float(line) for line in line_medi_data]

        for j in range(len(add_line)):
            x[line_index][j] = add_line[j]

    # Process dof data
    with open(f"../dataset/dof/dof{sample}.txt", 'r') as d:
        dof_index = 3
        res_line = d.read()
        dof_pre_data = res_line[333:]
        dof_medi_data = dof_pre_data.split()
        add_dof = []
        for dof in dof_medi_data:
            if not is_number(dof):
                dof = ord(dof[0]) * 1000 + ord(dof[1])
            add_dof.append(float(dof))

        for j in range(len(add_dof)):
            if j < 4:
                x[dof_index][j] = add_dof[j]
            elif 3 < j < 8:
                x[dof_index + 1][j - 4] = add_dof[j]
            else:
                x[dof_index + 2][j - 8] = add_dof[j]

    # Process load data
    with open(f"../dataset/load/load{sample}.txt", 'r') as ld:
        load_index = 6
        res_load = ld.read()
        load_pre_data = res_load[340:]
        load_medi_data = load_pre_data.split()
        add_load = []
        for load in load_medi_data:
            if not is_number(load):
                load = ord(load[0]) * 1000 + ord(load[1])
            add_load.append(float(load))

        for j in range(len(add_load)):
            x[load_index][j] = add_load[j]

    # Process node data
    with open(f"../dataset/node/node{sample}.txt", 'r') as n:
        node_index = 7
        res_node = n.read()
        node_pre_data = res_node[331:]
        node_medi_data = node_pre_data.split()
        add_node = [float(node) for node in node_medi_data]
        add_node = np.array(add_node)

        for i in range(int(len(add_node) / 7)):
            x[i + node_index, :7] = add_node.reshape((-1, 7))[i]

    # Process element data
    with open(f"../dataset/element/element{sample}.txt", 'r') as e:
        element_index = 57
        res_element = e.read()
        element_pre_data = res_element[269:]
        element_medi_data = element_pre_data.split()
        add_element = [float(element) for element in element_medi_data]
        add_element = np.array(add_element)

        for i in range(int(len(add_element) / 10)):
            x[i + element_index, :10] = add_element.reshape((-1, 10))[i]

    # Save the processed data to a file
    np.savetxt(f"../dataset/dy_sample/x_vector{sample}.txt", x)
