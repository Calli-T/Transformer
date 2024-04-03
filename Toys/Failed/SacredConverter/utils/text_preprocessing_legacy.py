import os
from re import sub

'''
f = open(file='./source/개역개정/1-01창세기.txt', mode='r')
print(f.readline())
'''

source_root = os.getcwd() + '\\source'
before_dir1 = source_root + '\\쉬운'
before_dir2 = source_root + '\\개역개정'
after_dir1 = source_root + '\\from'
after_dir2 = source_root + '\\to'

print(os.listdir(before_dir1))
print(os.listdir(before_dir2))
name_list1 = os.listdir(before_dir1)
name_list2 = os.listdir(before_dir2)


'''
f = open(before_dir1 + "\\" + name_list1[0], mode='r')
while True:
    line = f.readline()
    if line == '':
        break
    print(line.rstrip("\n"))
'''
max_from_seq = 0
for name in name_list1:
    fr = open(before_dir1 + "\\" + name, mode='r')
    fw = open(after_dir1 + "\\" + name, mode='w')
    pattern = r'\<[^)]*\>'

    while True:
        line = fr.readline()
        if line == '':
            break
        new_line = sub(pattern=pattern, repl='', string=" ".join(line.rstrip("\n").split(" ")[1:])).lstrip(" ")

        #print(new_line)

        if max_from_seq < len(new_line.split(" ")):
            max_from_seq = len(new_line.split(" "))

        fw.write(new_line+"\n")

    fr.close()
    fw.close()

    '''
        if name == "02 출애굽기.txt":
        break
    '''

print(max_from_seq)

max_to_seq = 0
for name in name_list2:
    fr = open(before_dir2 + "\\" + name, mode='r')
    fw = open(after_dir2 + "\\" + name, mode='w')
    pattern = r'\<[^)]*\>'

    while True:
        line = fr.readline()
        if line == '':
            break
        new_line = sub(pattern=pattern, repl='', string=" ".join(line.rstrip("\n").split(" ")[1:])).lstrip(" ")

        #print(new_line)
        if max_to_seq < len(new_line.split(" ")):
            max_to_seq = len(new_line.split(" "))

        fw.write(new_line+"\n")

    fr.close()
    fw.close()

    '''
        if name == "02 출애굽기.txt":
        break
    '''

print(max_to_seq)