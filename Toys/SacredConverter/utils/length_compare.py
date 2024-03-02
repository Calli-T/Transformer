import os
from re import sub

os.chdir("../")
source_root = os.getcwd() + '\\source'
before_dir1 = source_root + '\\쉬운'
before_dir2 = source_root + '\\개역개정'
after_dir1 = source_root + '\\from'
after_dir2 = source_root + '\\to'

print(os.listdir(before_dir1))
print(os.listdir(before_dir2))
name_list1 = os.listdir(before_dir1)
name_list2 = os.listdir(before_dir2)

max_seq = 0

for name in name_list1:
    fr1 = open(before_dir1 + "\\" + name, mode='r')
    fr2 = open(before_dir2 + "\\" + name, mode='r')
    fw1 = open(after_dir1 + "\\" + name, mode='w')
    fw2 = open(after_dir2 + "\\" + name, mode='w')

    pattern = r'\<[^)]*\>'

    count1 = 0
    count2 = 0

    while True:
        line1 = fr1.readline()
        line2 = fr2.readline()
        if line1 == '':
            break
        if line2 == '':
            break

        tag1 = line1.split(" ")[:1]
        tag2 = line2.split(" ")[:1]
        if tag1 != tag2:
            print(tag1, tag2)
            break

        new_line1 = sub(pattern=pattern, repl='', string=" ".join(line1.rstrip("\n").split(" ")[1:])).lstrip(" ")
        new_line2 = sub(pattern=pattern, repl='', string=" ".join(line2.rstrip("\n").split(" ")[1:])).lstrip(" ")

        fw1.write(new_line1+"\n")
        fw2.write(new_line2+"\n")

        if max_seq < len(new_line1.split(" ")):
            max_seq = len(new_line1.split(" "))
        if max_seq < len(new_line2.split(" ")):
            max_seq = len(new_line2.split(" "))

    fr1.close()
    fr2.close()
    fw1.close()
    fw2.close()

print(max_seq)

'''
    if name == "01 창세기.txt":
    break
'''
