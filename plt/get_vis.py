import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
from sklearn import datasets
from sklearn.manifold import TSNE

def get_2demon_data(data):
    tsne = TSNE(n_components=2,init='pca',random_state=0)
    result = tsne.fit_transform(data)
    # print(result)
    return result

def draw_map(file_path_formula_categories, file_path_2d_vectors_of_formulas):
    vector_2d_formulas = read_data_points(file_path_formula_categories, file_path_2d_vectors_of_formulas)
    x = []
    y = []
    colors = ["red", "blue", "green", "purple", "black", "silver", "cyan", "gold", "violet"]
    markers = ['o', 'D', 'X', '^', '*', 'h', 's', 'v', 'd']
    labels = ["Matrix", "Integral", "Series", "Limit", "Logarithm", "Trigonometric", "Set Theory", "Probability",
              "Derivative"]
    tabs = [0,17,18,20,20,18,20,17,18,20]
    tabstart = [0,17,35,55,75,93,113,130,148]
    tabend = [16,34,54,74,92,112,129,147,167]
    color = []
    marker = []
    max_x = -100
    min_x = 100
    max_y = -100
    min_y = 100

    for key in vector_2d_formulas.keys():
        for item in vector_2d_formulas[key]:
            x.append(item[0])
            if item[0] > max_x:
                max_x = item[0]
            elif item[0] < min_x:
                min_x = item[0]
            y.append(item[1])
            if item[1] > max_y:
                max_y = item[1]
            elif item[0] < min_y:
                min_y = item[0]
            color.append(colors[int(key) - 1])
            marker.append(markers[int(key)-1])

    x_range = max_x - min_x
    y_range = max_y - min_y
    x = (2 * (x - min_x) / x_range) - 1
    y = (2 * (y - min_y) / y_range) - 1
    legend_info = []
    for counter in range(0, len(colors)):
        legend_info.append(Line2D([0], [0], color='w', label=labels[counter], markerfacecolor=colors[counter],
                                  marker=markers[counter], markersize=8))

    fig, ax = plt.subplots()
    flag = 0
    for key in vector_2d_formulas.keys():
        key = int(key)
        print(key)
        start = tabstart[flag]
        end = tabend[flag]
        ax.scatter(x[start:end], y[start:end], c=color[start:end], marker=markers[key-1], alpha=0.9)
        flag+=1
    ax.legend(handles=legend_info, loc='upper right', ncol=2, prop={'size': 6})
    plt.savefig("pic4.svg",format="svg")
    plt.show()

def touch_vector(formula_dict,formula_name):
    f = open(formula_name+".txt",'w')
    f.write(formula_dict[formula_name])
    return 0

def read_formulas(file_path_formula_categories):
    with open('vector_file/vector_dict.pickle', 'rb') as file:
        vector_dict = pickle.load(file)
    file = open(file_path_formula_categories)
    line = file.readline().strip("\n")
    f = open("new_tsne",'w')
    while line:
        sections = line.split("/")
        # print(sections[1])
        point_key = sections[0].split(",")[0]
        f.write(point_key)
        f.write(',')
        f.write(sections[1])
        f.write('\n')
        line = file.readline().strip("\n")
    f.close()
    return 0

def read_data_points(file_path_formula_categories, vector_dict):
    file = open(file_path_formula_categories)
    line = file.readline().strip("\n")
    data_points = {}

    flag = 0
    while line:
        sections = line.split(",")

        #获取向量文件
        # file2 = open(file_path_2d_vectors_of_formulas + sections[1] + ".txt")
        #从txt中提取向量
        # vector = np.fromfile(file2, sep=" ")

        #获取向量
        try:
            vector = vector_dict[sections[1]]
            # 如果该公式的分类号 sec[0] 在点集中，则加入该分类子集
            if sections[0] in data_points.keys():
                data_points[sections[0]].append(vector)

            # 否则，增加新的子类，初始为这个vector
            else:
                lst = [vector]
                data_points[sections[0]] = lst

        except:
            print("eRRor:",end=':')
            print(sections[0])
            print(sections[1])

        line = file.readline().strip("\n")
    for key,value in data_points.items():
        print(key)
    return data_points

def vector_select(vector_dict):

    file = open('new_tsne')
    line = file.readline().strip("\n")

    while line:
        sections = line.split(",")
        # 获取向量
        try:
            vector = vector_dict[sections[1]]
            if sections[1] == 'Hurwitz_matrix:2':
                vector_for_draw = vector
            vector_for_draw = np.concatenate((vector_for_draw,vector),axis=0)

        except:
            print("eRRor:", end=':')
            print(sections[1])
        line = file.readline().strip("\n")

    return vector_for_draw

def set_new_dict(vector_dict,deom_2_vect):
    file = open('new_tsne')
    line = file.readline().strip("\n")
    flag = 0
    vect_2demon = {}
    while line:
        sections = line.split(",")
        # 获取向量
        if sections[1] in vector_dict.keys():
            vect_2demon[sections[1]] = deom_2_vect[flag]
            flag += 1
        # try:
        #     vector = vector_dict[sections[1]]
        #     vect_2demon[sections] =
        #     # if sections[1] == 'Hurwitz_matrix:2':
        #     #     vector_for_draw = vector
        #     # vector_for_draw = np.concatenate((vector_for_draw, vector), axis=0)
        #
        # except:
        #     print("eRRor:", end=':')
        #     print(sections[1])
        line = file.readline().strip("\n")

    file2 = open('2demon_vect.pickle', 'wb')
    pickle.dump(vect_2demon, file2)
    # print(vect_2demon)
    return vect_2demon

def main():
    temp_file = "new_tsne"
    with open('/home/aragorn/mywork/tangent/vector_file/vector_dict.pickle','rb') as file:
        vector_dict = pickle.load(file)
    # print(vector_dict['Hurwitz_matrix:2'])
    # read_formulas(temp_file)
    #test
    # tsne = TSNE(n_components=2,init='pca',random_state=0)
    # X = np.array([[1,2,3],[1,2,3],[1,7,8]])
    # result = tsne.fit_transform(X)
    # print(result)
    #转换数据格式
    # print(get_2demon_data(vector_select(vector_dict))[1])
    demon2_data = set_new_dict(vector_dict,get_2demon_data(vector_select(vector_dict)))
    draw_map(temp_file,demon2_data)

if __name__ == '__main__':
    main()