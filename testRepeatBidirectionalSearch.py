import numpy as np

raw_v =  np.array(["a", 0, "p", "a", 1, "p", "a", 0, "p", "a", 2, "p"])
print("raw_v:", raw_v)
print("len of raw v:", len(raw_v))

repeat3Rawv = np.tile(raw_v, 3)
print("repeat3Rawv:", repeat3Rawv)


for i in range(int(len(raw_v)/3)):
    angle  = raw_v[i*3 +1]

    if angle ==  '0' :
        print("selected angle in raw v:", angle)
        print( "current index in raw v：", i*3 +1 )

        print("try to find right nonzero")
        for right_index in range(len(raw_v) + (i+1)*3 +1, len(repeat3Rawv),3):
            print("right search value:", repeat3Rawv[right_index])
            if repeat3Rawv[right_index] != '0':
                print("searched right nonzero value:", repeat3Rawv[right_index] )
                break
        print("try to find left nonzero")
        for left_index in range(len(raw_v) + (i-1)*3 +1, 0, -1):
            print("left search value:", repeat3Rawv[left_index])
            if repeat3Rawv[left_index] != '0':
                print("searched left nonzero value:", repeat3Rawv[left_index] )
                break