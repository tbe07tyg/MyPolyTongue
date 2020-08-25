import os

try:
    def run_case(file_dir):
        fileIdx =1
        for root, dirs, files in os.walk(file_dir):
            print(files, "to be run")
            for i in files:

                if i.endswith('{}.py'.format(fileIdx)):
                    cmd = 'python ' + root + '/' + i + ' {}'.format(fileIdx)
                    print(cmd)
                    os.system(cmd)
                    fileIdx += 1
                if fileIdx>5:
                    break
        print("total run {} training scripts!".format(fileIdx))
except Exception as e:
    print("Has some error", e)

# Case 1  BaseExp
run_case('E:\\Projects\\MyPolyTongue\\TonguePlusData\\EXP_1_base_RunningScripts')

# # Case 2  Exp Mish
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_2_Mish')