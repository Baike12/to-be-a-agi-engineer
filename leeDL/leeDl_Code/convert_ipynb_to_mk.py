import os
import subprocess
import glob

def nbconvert():
    # 源目录和目标目录
    source_dir = r'D:\Code\Python\leedl-tutorial-1.1.8\leedl-tutorial-1.1.8\Homework'
    target_dir = r'D:\Note\to-be-a-agi-engineer\Code\LeeDl\HomeWork'

    # 创建目标目录如果不存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源目录下的所有文件夹
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # 检查是否是以HW开头的文件夹
        if os.path.isdir(folder_path) and folder_name.startswith('HW'):
            # 构造ipynb文件路径
            ipynb_file = glob.glob(os.path.join(folder_path, '*.ipynb'))

            # 检查ipynb文件是否存在
            if ipynb_file:
                ipynb_file = ipynb_file[0]
                if os.path.isfile(ipynb_file):
                    # 构造目标markdown文件路径
                    md_file = os.path.join(target_dir, f'{folder_name}.md')

                    # 执行nbconvert命令
                    command = ['jupyter', 'nbconvert', '--to', 'script', ipynb_file, '--output', md_file]
                    print("ipynbfile", ipynb_file)
                    subprocess.run(command)
                    print(f'Converted {ipynb_file} to {md_file}')
                else:
                    print(f'{ipynb_file} does not exist')

def main():
    nbconvert()

if __name__ == "__main__":
    main()
