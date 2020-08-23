import os

try:
    def run_case(file_dir):
        for root, dirs, files in os.walk(file_dir):
            for i in files:
                if i.endswith('.py'):
                    cmd = 'python ' + root + '/' + i
                    os.system(cmd)
except Exception as e:
    print("Has some error", e)

# Case 1
run_case('E:\\Projects\\MyPolyTongue\\TonguePlusData\\EXP_1_base_RunningScripts')