import os
from glob import glob

directory='/mnt/hdd/jyshim/workspace/Datasets/Ship/3d/20230526/snapshot/image'

# 작업 디렉토리를 이 스크립트의 디렉토리로 고정
#os.chdir(os.path.dirname(__file__))
#json file location fixed

main_block_dict={}
def extract_number_block_names(image_path):
    image_name = os.path.basename(image_path)
    s = image_name.split('-')
    block_name = s[1]
    block_len = len(block_name)
    
    return block_len

#./*/s[0]-s[1]-*-*.jpg -> s[0]-s[1]-*-*.jpg -> (s[0],s[1])
def call_main_block(directory):

    #main_block=[]
    modes=['8020','8021','8060','8061','8100','8101','8103','8104','8106','8107','8121','8122','8125','8126'
    ,'8127','8128','8129','8132','8134','8138','8140','8141','8148','8149','8150','8151','8159','8160','8164','8165','8173']

    images_list = []

    for modes in modes:
        images_list.extend(glob(f'{directory}/{modes}/*.jpg'))

    block_len = [extract_number_block_names(image_path) for image_path in images_list]

    block_size = len(block_len)

    for i in range(block_size):
        if block_len[i-1] == 4:
            os.remove(images_list[i-1])
            

def main():

    main_block_dict = call_main_block(directory)

if __name__ =='__main__':
    main()

