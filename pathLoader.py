# -*- coding: utf-8 -*-
"""
    @File: pathLoader.py
    @Author: Chaos
    @Date: 2023/6/23
    @Description: 读取路径文件的路径，返回给定路径字符串
"""
import os
import re


def getpath(keyword):
    """
    获取关键词所代表的存储路径
    :param keyword: 请求的关键词，目前包括dataRawSavedPath，netModelPath，datasetPath，testSetPath，不区分大小写。
    :return: 存储路径的字符串
    """
    # 路径存放目录
    txtPathFile = ".//path.txt"
    try:
        with open(txtPathFile, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print('"\"path.txt\" file does not exist! "')
    except Exception as err:
        print("Fatal Error! Error Type:\n\t" + repr(err))

    returnPath = ""
    for line in lines:
        if re.match(r'^[\s#]+', line):
            continue
        if keyword.lower() in line.lower():
            line = line.rstrip('\n')  # 去掉换行符
            line = re.sub(r'[\\]+', r'\\', line)
            returnPath = re.split(r'[\s\=]+', line)[1]
            if os.path.exists(returnPath):
                return returnPath
            else:
                raise ValueError("The path corresponding to the requested keyword does not exist, Please check the path file!")

    raise ValueError("requested path keyword \"%s\" does not exist! " % keyword)


if __name__ == '__main__':
    print("================== pathLoad ==================")
    print(getpath("datasetPath"))
