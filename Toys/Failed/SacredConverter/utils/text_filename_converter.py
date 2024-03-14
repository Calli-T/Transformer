import os


# 신약의 넘버링을 구약에 이어서
def ad2bcad():
    path = "C:\\Users\\joy14\\PycharmProjects\\Transformer\\Toys\\SacredConverter\\source\\source\\쉬운성경\\신약"
    name_list = os.listdir(path)
    old_name_list = os.listdir(path)

    for idx, name in enumerate(name_list):
        name_list[idx] = str(idx + 40) + " " + name.split(" ")[1]

    os.chdir(path)

    for idx, name in enumerate(old_name_list):
        os.rename(old_name_list[idx], name_list[idx])


def ad2bcad2():
    path = "/Toys/SacredConverter/source/개역개정"
    # print(os.listdir(path))
    name_list = os.listdir(path)

    os.chdir(path)

    for idx, name in enumerate(name_list):
        os.rename(name_list[idx], str(idx + 1) + " " + name[4:])
        
# 2자리 정수로 1-> 01과 같이 표현해야하나 빼먹은건 귀찮으니 수작업으로함

ad2bcad2()
