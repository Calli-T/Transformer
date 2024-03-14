import os
from re import sub

os.chdir('../')
source_root = os.getcwd() + '\\source'
after_dir1 = source_root + '\\from'
after_dir2 = source_root + '\\to'

name_list1 = os.listdir(after_dir1)
name_list2 = os.listdir(after_dir2)

for name in name_list1:
    fr1 = open(after_dir1 + "\\" + name, mode='r')
    fr2 = open(after_dir2 + "\\" + name, mode='r')

    count1 = 0
    count2 = 0

    while True:
        line1 = fr1.readline()
        if line1 == '':
            break
        else:
            count1 += 1

    while True:
        line2 = fr2.readline()
        if line2 == '':
            break
        else:
            count2 += 1

    if count1 == count2:
        print(f"{name}은 서로 줄 수가 같음")
    else:
        break
