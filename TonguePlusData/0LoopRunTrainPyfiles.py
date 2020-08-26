import os
from glob import glob
try:
    def run_case(file_dir, max_run=5):
        fileIdx =1
        # for root, dirs, files in os.walk(file_dir):
        #     print(files, "to be run")
        #     for i in files:
        all_files = glob(file_dir + '/*')
        print("all files:", all_files)
        for file in all_files:
            if file.endswith('Train.py'):
                for i in range(max_run):
                    print("run file: {}  {} ".format(file, fileIdx))
                    cmd = 'python ' + file + ' {}'.format(fileIdx)
                    print(cmd)
                    os.system(cmd)
                    fileIdx += 1
                    if fileIdx > max_run:
                        break
        print("total run {} training scripts!".format(fileIdx))
except Exception as e:
    print("Has some error", e)
#
# Case 1  BaseExp
run_case('E:\\Projects\\MyPolyTongue\\TonguePlusData\\EXP_5_CSP_SAE_FrontNeck')

# # Case 2  Exp Mish
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_2_Mish')


# # Case 4  EXP_4_SAE_MidNeck
# run_case('C:\\myProjects\\MyPolyTongue\\TonguePlusData\\EXP_4_SAE_MidNeck')
