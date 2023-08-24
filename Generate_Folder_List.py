# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import os

def init():
    fp = (r"/Volumes/RSaleeb_2TB/2023-06-19_SAVE_pmCSF/")
    print_dirs(fp)

def print_dirs(fp):

    for root, dirs, files in os.walk(fp):
        for name in dirs:

            for_py = 'pathList.append(r"{0}")'.format(os.path.join(root + "/" + name))
            print(for_py)    


#    for root, dirs, files in os.walk(fp):
#        for name in dirs:
#
#                if 'ThT.tif' in name:
        
#                        for_py = 'pathList.append(r"{0}")'.format(os.path.join(root))
#                        print(for_py)
                        

        
init()