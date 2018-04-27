
def read_attributes_num(file):
    attribute_num =0
    f = open(file)
    line = f.readline()
    for char in line:
        if(char == "\t"):
            attribute_num += 1
    return attribute_num
