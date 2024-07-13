import numpy as np
import time

# Initialize labels and data arrays
label = np.zeros((10000, 1))
label[2500:5000] = 1
x = np.zeros((10000, 500000))

def is_number(s):
    """
    Check if the input string is a number.
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

def mesh2mesh_csv():
    """
    Read and process mesh data from files.
    """
    for i in range(1, 2):
        index = i - 1
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Element/Element{i}.txt', 'r') as file_mesh:
            res = file_mesh.read()
            mesh_pre_data = res[298:]
            add_mesh = [int(ele) for ele in mesh_pre_data.split()]
            for j in range(len(add_mesh)):
                x[index][j] = add_mesh[j]

def total2total_csv_patch():
    """
    Read and process multiple types of data from files and combine them.
    """
    for i in range(10001, 20001):
        index = i - 10001
        # Read keypoint data
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Keypoint/Keypoint{i}.txt', 'r') as file_keypoint:
            keypoint_pre_data = file_keypoint.read()[309:]
            add_keypoint = [float(keypoint) for keypoint in keypoint_pre_data.split()]
            for j in range(len(add_keypoint)):
                x[index][j] = add_keypoint[j]

        # Read line data
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Line/Line{i}.txt', 'r') as file_line:
            line_pre_data = file_line.read()[310:]
            add_line = [float(line) for line in line_pre_data.split()]
            for j in range(len(add_keypoint), len(add_keypoint) + len(add_line)):
                x[index][j] = add_line[j - len(add_keypoint)]

        # Read node data
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Node/Node{i}.txt', 'r') as file_node:
            node_pre_data = file_node.read()[331:]
            add_node = [float(node) for node in node_pre_data.split()]
            for j in range(len(add_keypoint) + len(add_line), len(add_keypoint) + len(add_line) + len(add_node)):
                x[index][j] = add_node[j - len(add_keypoint) - len(add_line)]

        # Read element data
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Element/Element{i}.txt', 'r') as file_ele:
            ele_pre_data = file_ele.read()[269:]
            add_ele = [float(ele) for ele in ele_pre_data.split()]
            for j in range(len(add_keypoint) + len(add_line) + len(add_node), len(add_keypoint) + len(add_line) + len(add_node) + len(add_ele)):
                x[index][j] = add_ele[j - len(add_keypoint) - len(add_line) - len(add_node)]

        # Read load data
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Load/Load{i}.txt', 'r') as file_load:
            load_pre_data = file_load.read()[360:]
            load_medi_data = load_pre_data.split()
            add_load = [float(load) if is_number(load) else ord(load[0]) * 1000 + ord(load[1]) for load in load_medi_data]
            for j in range(len(add_keypoint) + len(add_line) + len(add_node) + len(add_ele), len(add_keypoint) + len(add_line) + len(add_node) + len(add_ele) + len(add_load)):
                x[index][j] = add_load[j - len(add_keypoint) - len(add_line) - len(add_node) - len(add_ele)]

        # Read DOF data
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Dof/Dof{i}.txt', 'r') as file_dof:
            dof_pre_data = file_dof.read()[355:]
            dof_medi_data = dof_pre_data.split()
            add_dof = [float(dof) if is_number(dof) else ord(dof[0]) * 1000 + ord(dof[1]) for dof in dof_medi_data]
            for j in range(len(add_keypoint) + len(add_line) + len(add_node) + len(add_ele) + len(add_load), len(add_keypoint) + len(add_line) + len(add_node) + len(add_ele) + len(add_load) + len(add_dof)):
                x[index][j] = add_dof[j - len(add_keypoint) - len(add_line) - len(add_node) - len(add_ele) - len(add_load)]

        # Read result data
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Result/Result{i}.txt', 'r') as file_result:
            result_pre_data = file_result.read()[486: -140]
            add_result = [float(result) for result in result_pre_data.split()]
            for j in range(len(add_keypoint) + len(add_line) + len(add_node) + len(add_ele) + len(add_load) + len(add_dof), len(add_keypoint) + len(add_line) + len(add_node) + len(add_ele) + len(add_load) + len(add_dof) + len(add_result)):
                x[index][j] = add_result[j - len(add_keypoint) - len(add_line) - len(add_node) - len(add_ele) - len(add_load) - len(add_dof)]

def read_data_keypoint():
    """
    Read keypoint data from files.
    """
    data = np.zeros((10000, 20))
    for i in range(10001, 20001):
        index = i - 10001
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Keypoint/Keypoint{i}.txt', 'r') as file_keypoint:
            keypoint_pre_data = file_keypoint.read()[309:]
            add_keypoint = [float(keypoint) for keypoint in keypoint_pre_data.split()]
            for j in range(len(add_keypoint)):
                data[index][j] = add_keypoint[j]
    return data

def save_txt_keypoint(data):
    """
    Save keypoint data to a txt file.
    """
    str_path = "/home/public/ken/Mesh Size for Structual static/Dataset/keypoint.txt"
    np.savetxt(str_path, data, delimiter=',', fmt='%.8f')

def read_data_line():
    """
    Read line data from files.
    """
    data = np.zeros((10000, 20))
    for i in range(10001, 20001):
        index = i - 10001
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Line/Line{i}.txt', 'r') as file_line:
            line_pre_data = file_line.read()[310:]
            add_line = [float(line) for line in line_pre_data.split()]
            for j in range(len(add_line)):
                data[index][j] = add_line[j]
    return data

def save_txt_line(data):
    """
    Save line data to a txt file.
    """
    str_path = "/home/public/ken/Mesh Size for Structual static/Dataset/line.txt"
    np.savetxt(str_path, data, delimiter=',', fmt='%.8f')

def read_data_node():
    """
    Read node data from files.
    """
    data = np.zeros((10000, 75000))
    for i in range(10001, 20001):
        index = i - 10001
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Node/Node{i}.txt', 'r') as file_node:
            node_pre_data = file_node.read()[331:]
            add_node = [float(node) for node in node_pre_data.split()]
            for j in range(len(add_node)):
                x[index][j] = add_node[j]
    return data

def save_txt_node(data):
    """
    Save node data to a txt file.
    """
    str_path = "/home/public/ken/Mesh Size for Structual static/Dataset/node.txt"
    np.savetxt(str_path, data, delimiter=',', fmt='%.8f')

def read_data_element():
    """
    Read element data from files.
    """
    data = np.zeros((10000, 100000))
    for i in range(10001, 20001):
        index = i - 10001
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Element/Element{i}.txt', 'r') as file_ele:
            ele_pre_data = file_ele.read()[269:]
            add_ele = [float(ele) for ele in ele_pre_data.split()]
            for j in range(len(add_ele)):
                data[index][j] = add_ele[j]
    return data

def save_txt_element(data):
    """
    Save element data to a txt file.
    """
    str_path = "/home/public/ken/Mesh Size for Structual static/Dataset/element.txt"
    np.savetxt(str_path, data, delimiter=',', fmt='%.8f')

def read_data_dof():
    """
    Read degree of freedom (DOF) data from files.
    """
    data = np.zeros((10000, 10))
    for i in range(10001, 20001):
        index = i - 10001
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Dof/Dof{i}.txt', 'r') as file_dof:
            dof_pre_data = file_dof.read()[355:]
            dof_medi_data = dof_pre_data.split()
            add_dof = [float(dof) if is_number(dof) else ord(dof[0]) * 1000 + ord(dof[1]) for dof in dof_medi_data]
            for j in range(len(add_dof)):
                x[index][j] = add_dof[j]
    return data

def save_txt_dof(data):
    """
    Save degree of freedom (DOF) data to a txt file.
    """
    str_path = "/home/public/ken/Mesh Size for Structual static/Dataset/dof.txt"
    np.savetxt(str_path, data, delimiter=',', fmt='%.8f')

def read_data_load():
    """
    Read load data from files.
    """
    data = np.zeros((10000, 10))
    for i in range(10001, 20001):
        index = i - 10001
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Load/Load{i}.txt', 'r') as file_load:
            load_pre_data = file_load.read()[360:]
            load_medi_data = load_pre_data.split()
            add_load = [float(load) if is_number(load) else ord(load[0]) * 1000 + ord(load[1]) for load in load_medi_data]
            for j in range(len(add_load)):
                data[index][j] = add_load[j]
    return data

def save_txt_load(data):
    """
    Save load data to a txt file.
    """
    str_path = "/home/public/ken/Mesh Size for Structual static/Dataset/load.txt"
    np.savetxt(str_path, data, delimiter=',', fmt='%.8f')

def read_data_result():
    """
    Read result data from files.
    """
    data = np.zeros((10000, 100000))
    for i in range(10001, 20001):
        index = i - 10001
        with open(f'/home/public/ken/Mesh Size for Structual static/Dataset/Result/Result{i}.txt', 'r') as file_result:
            result_pre_data = file_result.read()[486: -140]
            add_result = [float(result) for result in result_pre_data.split()]
            for j in range(len(add_result)):
                data[index][j] = add_result[j]
    return data

def save_txt_result(data):
    """
    Save result data to a txt file.
    """
    str_path = "/home/public/ken/Mesh Size for Structual static/Dataset/result.txt"
    np.savetxt(str_path, data, delimiter=',', fmt='%.8f')

if __name__ == '__main__':
    start_time = time.time()
    # total2total_csv_patch()
    keypoint = read_data_keypoint()
    save_txt_keypoint(keypoint)
    line = read_data_line()
    save_txt_line(line)
    node = read_data_node()
    save_txt_node(node)
    element = read_data_element()
    save_txt_element(element)
    dof = read_data_dof()
    save_txt_dof(dof)
    load = read_data_load()
    save_txt_load(load)
    result = read_data_result()
    save_txt_result(result)
    end_time = time.time()
    print('Execution time: %f seconds' % (end_time - start_time))
