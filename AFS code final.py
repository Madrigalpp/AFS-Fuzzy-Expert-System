from scipy.special import comb
from scipy.stats import pearsonr

import numpy as np
import pandas as pd
from tkinter import filedialog
import tkinter as tk
import os
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

################################################忽略警告模块##############################################################
import warnings
warnings.filterwarnings("ignore")
################################################数据导入导出模块##############################################################
def AFS_readxlsx():
    # 设置文件对话框会显示的文件类型
    application_window = tk.Tk()
    application_window.withdraw()
    # 设置文件对话框会显示的文件类型
    filetypes = [('all files', '.*'), ('xlsx files', '.xlsx')]
    # 返回选择文件路径
    filepath = filedialog.askopenfilename(parent=application_window,
                                          initialdir=os.getcwd(),
                                          title="请选择一个文件:", filetypes=filetypes)
    """
    if messagebox.askokcancel("Quit", "请确认是否退出？"):
        return application_window.destroy()
    else:
        xlsx = pd.ExcelFile(filepath, engine="openpyxl")
        mcc = pd.read_excel(xlsx, 'Sheet1',index_col=None,header=None)
        mcc = np.array(mcc)
        M = np.mat(mcc)
        return  M
    """
    xlsx = pd.ExcelFile(filepath, engine="openpyxl")
    mcc = pd.read_excel(xlsx, 'Sheet1', header=0, index_col=0)
    return mcc
def AFS_input():
    k = input('\n 是否输入样本(yes-1\\no-0)：')
    k = int(k)
    arr = []
    if k == 1:
        for i in range(99):
            pt = input('\n 请输入模糊概念：')
            pt = int(pt)
            arr.append(pt)
            if pt == -1:
                arr.remove(-1)
                return arr
    else:
        judge = input('\n 是否确定退出(yes-1\\no-0)：')
        judge = int(judge)
        if judge == 1:
            for i in range(99):
                pt = input('\n 请输入模糊概念：')
                pt = int(pt)
                arr.append(pt)
                if pt == -1:
                    arr.remove(-1)
                    return arr
        else:
            exit()
def AFS_writexlsx(data):
    # 创建一个窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏窗口
    # 调用文件保存窗口, 打开指定的初始路径
    filetypes = [('all files', '.*'), ('xlsx files', '.xlsx')]
    filepath = filedialog.asksaveasfilename(initialdir="C:\\Users\\DELL\\Desktop", title="保存文件",
                                            filetypes= filetypes)
    xlsx = pd.ExcelFile(filepath, engine="xlsxwriter")
    mcc = pd.read_excel(xlsx, 'Sheet1', header=0, index_col=0)
    return  xlsx
#############################################简单数据集关联性算子###########################################################
def AFS_res(m):
    m_isnull = m.loc[~(m == 0).all(axis=1)]
    if len(m_isnull) == 1:
        AFS_SN = m_isnull
        return AFS_SN
    else:
        index_label = m.index.values
        header_label = m.columns.values
        m_arr = np.array(m)
        m_mat = np.mat(m_arr)
        row, col = m_mat.shape
        MZE = np.zeros((1, col))
        boundary = 0
        ###################挑选具备工程意义集合#####################
        for i in range(0, np.size(m, axis=0)):
            if np.sum(m_mat[i, :]) != 0:
                MB = m_mat[i, :]
                MC = m_mat[i, :]
                boundary = i
                break
        ###################局部元素关联性初步筛选，并形成两大结构#####################
        for j in range(boundary + 1, row):
            TR = np.min([MB[0, :], m_mat[j, :]], 0)
            if np.sum(TR, axis=1) != 0:
                MB = np.vstack((MB, m_mat[j, :]))
                m_mat[j, :] = MZE
        ###################计算上述两个结构体的关联性##############################
        while np.size(MC, 0) != np.size(MB, 0):
            TRAN = MB
            MC = MB
            for j in range(boundary + 1, np.size(m, axis=0)):
                rr = np.min([np.max(TRAN, 0), m_mat[j]], 0)
                if np.sum(rr) != 0:
                    TRAN = np.vstack((TRAN, m_mat[j]))
                    m_mat[j] = MZE
            MB = TRAN
        m_mat[i] = MZE
        AFS_S1_ind = []
        for j in range(np.size(MC, axis=0)):
            for i in index_label:
                if np.all((MC[j, :] == m.loc[[i], :]).astype("int")) != 0:
                    AFS_S1_ind.append(i)
        AFS_S1 = pd.DataFrame(MC, columns=header_label, index=AFS_S1_ind)
        del_num = AFS_S1.loc[:, (AFS_S1 == 0).all()].columns
        for i in del_num:
            del AFS_S1[i]
        m_red = pd.DataFrame(m_mat, columns=header_label, index=index_label)
        return AFS_S1, m_red
def AFS_res_mul(m):
    AFS_S = []
    m_isnull = m.loc[~(m == 0).all(axis=1)]
    if len(m_isnull) == 1:
        AFS_SN = m_isnull
        AFS_S.append(AFS_SN)
        return AFS_S[0]
    else:
        AFS_S1, m_red = AFS_res(m)
        AFS_S.append(AFS_S1)
        m_red_isnull = m_red.loc[~(m_red == 0).all(axis=1)]
        i = len(m_red_isnull)
        if i != 1:
            while np.sum(np.mat(m_red)) != 0:
                AFS_SN, m_redn = AFS_res(m_red_isnull)
                AFS_S.append(AFS_SN)
                m_red = m_redn
            for k in range(len(AFS_S)):
                print(f'第{k + 1}结构:\n', AFS_S[k])
            return AFS_S
############################################复杂数据集关联性算子############################################################
def AFS_res_per(m):
    m_isnull = m.loc[~(m == 0).all(axis=1)]
    if len(m_isnull) == 1:
        AFS_SN = m_isnull
        return AFS_SN
    else:
        index_label = m.index.values
        header_label = m.columns.values
        m_arr = np.array(m)
        m_mat = np.mat(m_arr)
        row, col = m_mat.shape
        MZE = np.zeros((1, col))
        boundary = 0
        ###################挑选具备工程意义集合#####################
        for i in range(0, np.size(m, axis=0)):
            if np.sum(m_mat[i, :]) != 0:
                MB = m_mat[i, :]
                MC = m_mat[i, :]
                boundary = i
                break
        ###################局部元素关联性初步筛选，并形成两大结构#####################
        k = float(input('\n 输入阈值:'))
        for j in range(boundary + 1, row):
            TR = pearsonr(np.array(MB[0, :]).flatten().transpose(), np.array(m_mat[j, :]).flatten().transpose())
            if TR[0] > k:
                MB = np.vstack((MB, m_mat[j, :]))
                m_mat[j, :] = MZE
        ###################计算上述两个结构体的关联性##############################
        AFS_S = []
        AFS_SN = []
        m_redn = []
        MB_2 = np.zeros((1, np.size(MB, 1)))
        while np.size(MC, 0) != np.size(MB, 0):
            TRAN = MB
            MC = MB
            TRAN_2 = MB_2
            for j in range(boundary + 1, np.size(m, axis=0)):
                rr = pearsonr(np.array(np.max(TRAN, 0)).flatten(), np.array(m_mat[j]).flatten())
                if rr[0] > k:
                    TRAN = np.vstack((TRAN, m_mat[j]))
                    m_mat[j, :] = MZE

                elif rr[0] <= k:
                    TRAN_2 = np.vstack((TRAN_2, m_mat[j]))
                    TRAN_2 = TRAN_2[[not np.all(TRAN_2[i, :] == 0) for i in range(np.size(TRAN_2, 0))], :]
                    m_mat[j, :] = MZE
            MB = TRAN
            MB_2 = TRAN_2
        m_mat[i] = MZE
        if np.any(MB_2[0, :]) == 0:
            AFS_S.append(MC)
            for k in range(len(AFS_S)):
                AFS_S1_ind = []
                for j in range(np.size(AFS_S[k], axis=0)):
                    for i in index_label:
                        if np.all((AFS_S[k][j, :] == m.loc[[i], :]).astype("int")) != 0:
                            AFS_S1_ind.append(i)
                AFS_S1 = pd.DataFrame(AFS_S[k], columns=header_label, index=AFS_S1_ind)
                del_num = AFS_S1.loc[:, (AFS_S1 == 0).all()].columns
                for i in del_num:
                    del AFS_S1[i]
                m_red = pd.DataFrame(m_mat, columns=header_label, index=index_label)
                AFS_SN.append(AFS_S1)
                m_redn.append(m_red)
        else:
            AFS_S.append(MC)
            AFS_S.append(MB_2)
            for k in range(len(AFS_S)):
                AFS_S1_ind = []
                for j in range(np.size(AFS_S[k], axis=0)):
                    for i in index_label:
                        if np.all((AFS_S[k][j, :] == m.loc[[i], :]).astype("int")) != 0:
                            AFS_S1_ind.append(i)
                cc = header_label
                cd = AFS_S1_ind
                ce = AFS_S[k]
                AFS_S1 = pd.DataFrame(AFS_S[k], columns=header_label, index=AFS_S1_ind)
                del_num = AFS_S1.loc[:, (AFS_S1 == 0).all()].columns
                for i in del_num:
                    del AFS_S1[i]
                m_red = pd.DataFrame(m_mat, columns=header_label, index=index_label)
                AFS_SN.append(AFS_S1)
                m_redn.append(m_red)
        return AFS_SN, m_redn
def AFS_res_per_mul(m):
    AFS_S = []
    m_isnull = m.loc[~(m == 0).all(axis=1)]
    if len(m_isnull) == 1:
        AFS_SN = m_isnull
        AFS_S.append(AFS_SN)
        return AFS_S[0]
    else:
        AFS_SN, m_redn = AFS_res_per(m)
        for i in range(len(AFS_SN)):
            if i < len(AFS_SN):
                AFS_S.append(AFS_SN[i])
        m_red_isnull = m_redn[len(AFS_SN) - 1].loc[~(m_redn[len(AFS_SN) - 1] == 0).all(axis=1)]
        j = len(m_red_isnull)
        mred = m_redn[len(AFS_SN) - 1]
        if j != 1:
            while np.sum(np.mat(mred)) != 0:
                AFS_SN2, m_redn2 = AFS_res_per(m_red_isnull)
                for k in range(len(AFS_SN2)):
                    if k < len(AFS_SN2):
                        AFS_S.append(AFS_SN2[k])
                mred = m_redn2[len(AFS_SN2) - 1]
        for h in range(len(AFS_S)):
            print(f'第{h + 1}结构:\n', AFS_S[h])
        return AFS_S
class AFS:

    # def __init__(self,filename):
    #    self.filename = filename
    #############################################单个复杂模糊概念转换及约简##########################################
    def tran_bool(self, m):
        Num_Row = int(m.max())  # 矩阵最大值求解
        row, col = np.shape(m)  # 获取矩阵行数与列数
        receive = np.zeros((Num_Row, col))  # 以矩阵最大值作为行数，列数不变，组建0值矩阵
        for i in range(col):
            No_Zero_Row = np.argwhere(m[:, i] != 0)  # 以矩阵最大值作为行数，列数不变，组建0值矩阵
            for j in range(len(No_Zero_Row)):
                No_Zero_Row_num = No_Zero_Row[:, 0][j]  # 返回每列非零元素索引值
                m_tran_a = np.array(m)
                No_Zero_Elem = m_tran_a[No_Zero_Row_num:, i][0]  # 返回每列非零元素索引值对应的数值
                receive[int(No_Zero_Elem - 1), i] = 1  # 将对应的数值作为行数添加于bool数组，由于原始数组最小值为，所以为保证索引值不超限必须-1
        return receive

    def negateEI(self, m):
        Bool_M = AFS.tran_bool(m)  # 数值转换为bool数据类型
        Bool_M = AFS.reduceEI_bool(Bool_M)  # 约减EI代数
        m_row, m_col = m.shape  # 获取数值矩阵大小
        Bool_M_row, Bool_M_col = Bool_M.shape  # 获取数值矩阵大小
        for i in range(m_col):  # 循环逻辑判断语句模块子结构
            for j in range(m_row):
                if (m[j, i] == 0):
                    continue
                elif (m[j, i] % 2 == 1):
                    m[j, i] = m[j, i] + 1
                else:
                    m[j, i] = m[j, i] - 1
        T = np.transpose(m[:, 0])  # 矩阵首列转置
        Del_EI = np.argwhere(T != 0)  # 找到判断为1的列索引
        # 找到矩阵中每列不为0的数值，并转置后进行降维
        Del_EI = np.transpose(Del_EI[:, [not np.all(Del_EI[:, i] == 0) for i in range(Del_EI.shape[1])]]).flatten()
        Bool_N = AFS.tran_bool(T[:, Del_EI])  # 数值转换为bool数据类型
        for j in range(1, Bool_M_col):
            T = np.transpose(m[:, j])  # 矩阵除首列外其他列转置
            Del_EI = np.argwhere(T != 0)  # 找到判断为1的列索引
            # 找到矩阵中每列不为0的数值，并转置后进行降维
            Del_EI = np.transpose(Del_EI[:, [not np.all(Del_EI[:, i] == 0) for i in range(Del_EI.shape[1])]]).flatten()
            # 将多个bool矩阵组合
            Bool_N = AFS.mult_bool(AFS.reduceEI_bool(Bool_N), AFS.tran_bool(T[:, Del_EI]), 1)
        return (Bool_N)

    def reduceEI_bool(self, m):
        row, col = np.shape(m)  # 获取矩阵行数与列数
        sum_sort = np.sort(np.sum(np.mat(m), axis=0)).astype("int")  # 按列求取元素和，输出为行向量
        sum_sort_index = np.argsort(np.sum(np.mat(m), axis=0))  # 输出原始元素所在索引位置
        Reserve_EI = np.mat(np.zeros((row, col)))
        m_tran_c = np.mat(np.zeros((row, col)))
        if sum_sort[0, 0] == 0:
            N = 0
        else:
            for i in range(col):
                m_tran_c[:, i] = m[:, sum_sort_index[0, i]].reshape(-1,
                                                                    1)  # 按照顺序排列索引位置重组原始矩阵,此语句如果使用和reduceeico_bool-run同样的
                # 表述形式会出现异常计算错误，因此假定了一个中间变量m_tran_c作为传递变量
            if col == 1:  # 判定条件，单列数据不参与运算
                N = m_tran_c
            else:
                for i in range(col):
                    if np.all(m_tran_c[:, i] == 0):  # 判定条件，搜索矩阵每列全部元素是否为0
                        # np.all(np.array)对矩阵所有元素做与操作，所有为True则返回True
                        # np.any(np.array)对矩阵所有元素做或运算，存在True则返回True
                        continue
                    else:
                        Reserve_EI[:, i] = m_tran_c[:, i]  # 提取到导入数据按列赋值中间变量
                        Nnz_Element = np.argwhere((Reserve_EI[:, i] != 0))  # 返回该列每个元素为零的索引值
                        Nnz_Sum_mat = np.mat(np.zeros((len(Nnz_Element), col)))  # 创建0行向量计算矩阵空间
                        for k in range(len(Nnz_Element)):
                            Nnz_Sum_mat[k, :] = (m_tran_c[Nnz_Element[:, 0][k], :])  # 矩阵空间填充
                        Nnz_Sum = np.sum(Nnz_Sum_mat, axis=0)  # 计算该列所有为零位置在原有数据结构中每一列的元素和
                        Judge_EI = np.where(np.array(Nnz_Sum) == sum_sort[0, i], 1, 0)  # 将具体数据转换为bool类型进行判断
                        Del_EI = np.argwhere(Judge_EI == 1)  # 找到判断为1的列索引
                        for j in range(len(Del_EI)):
                            Del_EI_num = Del_EI[:, 1][j]
                            m_tran_c[:, Del_EI_num] = 0  # 将上述判断为真的列向量修正为0，即初始化
                N = Reserve_EI[:, [not np.all(Reserve_EI[:, i] == 0) for i in range(Reserve_EI.shape[1])]]
        return (N)

    def reduceEI(self, m):
        m_tran_a = np.array(m)
        if np.sum(m_tran_a[0]) != 0:  # 若第一行不为0 ， 则不存在更简单的矩阵
            print('the first row of input matrix must be zeros.')
        else:
            m_tran_a_cpt = AFS.compactei(m)  ## 先对EI进行紧缩，除去多余的0
            m_tran_a_cpt_bool = AFS.tran_bool(m_tran_a_cpt)  ## 转换成布尔型矩阵
            c = np.mat(m_tran_a_cpt_bool)
            Reserve_EI = AFS.reduceEI_bool(c)
            d = np.mat(Reserve_EI)
            N = AFS.tran_EI_mat(d)  ## 转换为EI形式的向量
        return (N)

    def compactei(self, m):
        if np.any(m[0, :] != 0):
            print("矩阵第一行必须为0")
        else:
            m_sort = np.sort(m, axis=0)  # 矩阵按列排序
            Max_Num_Column = 0  # 初始化中间比较变量
            row, col = m_sort.shape  # 返回矩阵行列数
            for i in range(col):
                comp = np.unique(m_sort[:, i], axis=0)  # 按列剔除矩阵重复元素
                C_size = np.count_nonzero(comp)  # 返回非零元素个数
                Max_Num_Column = max(Max_Num_Column, C_size)  # 返回每列具有实际意义的数值个数
                cc = np.zeros((row - C_size, 1))  # 创建重复元素与原有第一行元素为0的矩阵的新一轮为0矩阵
                m_sort[:, i] = np.concatenate((cc, comp[1:]), axis=0)  # 按列拼接数组
            n = m_sort[row - Max_Num_Column - 1:][:]
            return n

    def tran_EI_mat(self, m):
        m_tran = np.mat(m)
        r, c = np.shape(m_tran)
        Set_M = np.mat(range(1, r + 1))
        Repmat_Set_M = np.mat(np.tile(Set_M.transpose(), (1, c)))
        x1 = np.zeros((1, c))  ##建立一个空矩阵
        Repmat_Set_M_f = np.multiply(Repmat_Set_M, m_tran)
        N = np.vstack([x1, Repmat_Set_M_f])
        N = AFS.compactei(N)
        return (N)

    #############################################两个复杂模糊概念比较分析###########################################
    def less_ei(self, x, y):
        Bool_x = AFS.tran_bool(x)  # 转换为bool数值类型
        Bool_y = AFS.tran_bool(y)  # 转换为bool数值类型

        Bool_x_re = AFS.reduceEI_bool(Bool_x)  # 约简用布尔矩阵表示的EM中的一个元素
        Bool_y_re = AFS.reduceEI_bool(Bool_y)  # 约简用布尔矩阵表示的EM中的一个元素

        Bool_x_row, Bool_x_col = Bool_x_re.shape  # 获取矩阵大小
        Bool_y_row, Bool_y_col = Bool_y_re.shape  # 获取矩阵大小

        Size_Row = max(Bool_x_row, Bool_y_row)  # 比较两矩阵获取最大行数

        Bool_x_zero = np.zeros((Size_Row - Bool_x_row, Bool_x_col))  # 初始化0矩阵
        Bool_x = np.vstack((Bool_x_re, Bool_x_zero))  # 拼接0矩阵
        Bool_y_zero = np.zeros((Size_Row - Bool_y_row, Bool_y_col))  # 初始化0矩阵
        Bool_y = np.vstack((Bool_y_re, Bool_y_zero))  # 拼接0矩阵
        for i in range(Bool_x_col):
            Tran = np.tile(Bool_x[:, i], (1, Bool_y_col))  # 按列遍历矩阵扩充，行数不变，列数扩大
            # 判断语句，获取布尔类型比较结果且强制转化为0/1数值
            if (((np.sum((Tran >= Bool_y).astype(int), axis=0)) == Size_Row).any().astype(int) == 0):
                f = 0
                break
            else:
                f = 1
                continue
        return f

    def equal_ei(self, x, y):
        if AFS.less_ei(x, y) == 1 and AFS.less_ei(y, x) == 1:
            f = 1
        else:
            f = 0
        return f

    ############################################两个复杂模糊概念的EI代数系统#########################################
    def EIsum(self, x, y):
        rowm, colm = np.shape(x)
        rown, coln = np.shape(y)  # 先找到两个size
        CC = np.vstack([np.zeros((max(rowm, rown) - rowm, colm)), x])
        CD = np.vstack([np.zeros((max(rowm, rown) - rown, coln)), y])
        EI_Elem = np.hstack([CC, CD])
        final = AFS.reduceEI(EI_Elem)
        return (final)

    def EImult(self, x, y):
        boolm = AFS.tran_bool(x)
        booln = AFS.tran_bool(y)
        rbm = AFS.reduceEI_bool(np.mat(boolm))
        rbn = AFS.reduceEI_bool(np.mat(booln))
        rowm, colm = np.shape(rbm)
        rown, coln = np.shape(rbn)
        dima = np.zeros((max(rowm, rown) - rowm, colm))
        dimb = np.zeros((max(rowm, rown) - rown, coln))
        boolmf = np.concatenate((rbm, dima), axis=0)
        boolnf = np.concatenate((rbn, dimb), axis=0)
        rowbn, colbn = boolnf.shape
        T = colm - coln
        Logicor = np.empty((rowbn, colbn), dtype=object)
        if T > 0:
            for i in range(coln):
                repmatm = np.tile(boolnf[:, i], (1, colm))
                matrepmatm = np.mat(repmatm)
                cc = np.mat(boolmf + matrepmatm)
                cc[cc > 0] = 1
                Logicor = np.concatenate((Logicor, cc), axis=1)
                Logicorf = Logicor[:, colbn:]
                Logicorfm = np.mat(Logicorf)
                final = AFS.reduceEI_bool(Logicorfm)
        else:
            for i in range(colm):
                repmatm = np.tile(boolmf[:, i], (1, coln))
                matrepmatm = np.mat(repmatm)
                cc = np.mat(boolnf + matrepmatm)
                cc[cc > 0] = 1
                Logicor = np.concatenate((Logicor, cc), axis=1)
                Logicorf = Logicor[:, colbn:]
                Logicorfm = np.mat(Logicorf)
                final = AFS.reduceEI_bool(Logicorfm)
        ans = AFS.tran_EI_mat(final)
        return ans

    ############################################两个复杂模糊概念的EII代数系统#########################################

    def compacteii(self, m):
        row, col = m.shape  # 矩阵转置
        k = -1  # 设置中间变量控制循环体

        for i in range(row):
            if (np.sum(m[i, :]) != 0):  # 逐行求和，执行判断语句，观察是否符合循环跳出条件
                break
            k = k + 1
        if k == -1:
            print('the first row of input matrix must be zeros.')

        mat = m[k:, :]  # 矩阵切片，切片首行自动保留
        EI_Num = np.where(np.array(np.sum(mat, axis=1)) == 0, 1, 0)  # 矩阵遍历求和，返回bool类型文件，将符合条件数值得到
        EI_Num = np.argwhere(EI_Num == 1)  # 获取上述条件索引值
        EI_Num = EI_Num[:,
                 [not np.all(EI_Num[:, i] == 0) for i in range(EI_Num.shape[1])]]  # 由于np.argwhere函数返回是数值的坐标，需要剔除零列
        if np.size(EI_Num, axis=0) == 1:
            print('the concept represented by input matrix must be a EII element.')

        EI_Elem = mat[0:EI_Num[1, 0], :]  # 矩阵切片，切片尾行不算数值
        Simple = mat[EI_Num[1, 0]:, :]  # 矩阵切片，切片首行自动保留
        N = np.vstack((AFS.compactei(EI_Elem), AFS.compactei(Simple)))  # 矩阵拼接
        return N

    def reduceEII(self, m):
        if np.any(m[0, :] != 0):
            print('the first row of input matrix must be zeros.')
        M = AFS.compacteii(m)
        if np.size(M, axis=1) == 1:
            N = M
        else:
            Num_EI_Elem = np.where(np.array(np.sum(M, axis=1)) == 0, 1, 0)  # 矩阵遍历求和，返回bool类型文件，将符合条件数值得到
            Num_EI_Elem = np.argwhere(Num_EI_Elem == 1)  # 获取上述条件索引值
            # 由于np.argwhere函数返回是数值的坐标，需要剔除零列
            Num_EI_Elem = Num_EI_Elem[:, [not np.all(Num_EI_Elem[:, i] == 0) for i in range(Num_EI_Elem.shape[1])]]
            Bool_EI_Elem = AFS.tran_bool(M[0:Num_EI_Elem[1, 0], :])  # 将数值文件转换为bool类型
            Bool_EI_Elem_row, Bool_EI_Elem_col = Bool_EI_Elem.shape  # 获取数值矩阵的行数与列数
            sum_sort = np.sort(np.sum(Bool_EI_Elem, axis=0))  # 按列求取元素和，输出为行向量
            sum_sort_index = np.argsort(np.sum(Bool_EI_Elem, axis=0))  # 输出原始元素所在索引位置
            if sum_sort[0] == 0:
                N = M[:, sum_sort_index[0]]
            else:
                Bool_EI_Elem = Bool_EI_Elem[:, sum_sort_index]
                Simple = M[Num_EI_Elem[1, 0]:, sum_sort_index]  # 数据切片
                Simple_row, simple__col = Simple.shape  # 获取数值矩阵的行数与列数
                Reserve_EI = np.mat(np.zeros((Bool_EI_Elem_row, Bool_EI_Elem_col)))  # 创建0矩阵作为中间变量用于数据传递
                Reduced_Simple = np.mat(np.zeros((Simple_row, simple__col)))  # 创建0矩阵作为中间变量用于数据传递
                for i in range(Bool_EI_Elem_col):
                    if np.all(Bool_EI_Elem[:, i] == 0):  # 判定条件，搜索矩阵每列全部元素是否为0
                        # np.all(np.array)对矩阵所有元素做与操作，所有为True则返回True
                        # np.any(np.array)对矩阵所有元素做或运算，存在True则返回True
                        continue
                    else:
                        Reserve_EI[:, i] = Bool_EI_Elem[:, i].reshape(Bool_EI_Elem_row, 1)  # numpy做矩阵运算会导致维度丢失，此处为修复
                        Reduced_Simple[:, i] = Simple[:, i].reshape(Simple_row, 1)  # numpy做矩阵运算会导致维度丢失，此处为修复
                        Nnz_Element = np.where(np.array(Bool_EI_Elem[:, i]) != 0, 1, 0)  # 矩阵遍历求和，返回bool类型文件，将符合条件数值得到
                        Nnz_Element = np.argwhere(Nnz_Element == 1)  # 获取上述条件索引值
                        # 由于np.argwhere函数返回是数值的坐标，需要剔除零列
                        Nnz_Element = Nnz_Element[:,
                                      [not np.all(Num_EI_Elem[:, i] == 0) for i in range(Num_EI_Elem.shape[1])]]
                        Nnz_Sum = np.sum(Bool_EI_Elem[Nnz_Element, :], axis=0)
                        Judge_EI = np.where(np.array(Nnz_Sum) == sum_sort[i], 1, 0)  # 将具体数据转换为bool类型进行判断
                        Del_EI = np.argwhere(Judge_EI == 1)  # 找到判断为1的列索引
                        for j in range(len(Del_EI)):
                            Del_EI_num = Del_EI[:, 1][j]
                            Bool_EI_Elem[:, Del_EI_num] = 0  # 将上述判断为真的列向量修正为0，即初始化
                N = np.vstack([AFS.tran_EI_mat(Reserve_EI), Reduced_Simple])
                # 由于np.vstack函数只能完成两个非零矩阵的拼接，因此需要在建立矩阵后进行批量清除
                N = N[:, [not np.all(N[:, i] == 0) for i in range(N.shape[1])]]
        return N

    def EIISUM(self, m, n):
        m = AFS.reduceEII(m)
        n = AFS.reduceEII(n)

        m_Sum = np.sum(m, axis=1)  # 按列计算矩阵元素和
        M_EI_Num = np.argwhere(m_Sum == 0)  # 找到元素和为0的索引值
        M_EI_Num = M_EI_Num[:, [not np.all(M_EI_Num[:, i] == 0) for i in range(M_EI_Num.shape[1])]].flatten()

        n_Sum = np.sum(n, axis=1)  # 按列计算矩阵元素和
        N_EI_Num = np.argwhere(n_Sum == 0)  # 找到元素和为0的索引值
        N_EI_Num = N_EI_Num[:, [not np.all(N_EI_Num[:, i] == 0) for i in range(N_EI_Num.shape[1])]].flatten()

        M_EI_Elem = m[:M_EI_Num[1], :]
        N_EI_Elem = n[:N_EI_Num[1], :]
        M_Simple = m[M_EI_Num[1]:, :]
        N_Simple = n[N_EI_Num[1]:, :]

        M_EI_Elem_row, M_EI_Elem_col = M_EI_Elem.shape
        N_EI_Elem_row, N_EI_Elem_col = N_EI_Elem.shape
        M_Simple_row, M_Simple_col = M_Simple.shape
        N_Simple_row, N_Simple_col = N_Simple.shape

        CC = np.vstack([np.zeros((max(M_EI_Elem_row, N_EI_Elem_row) - M_EI_Elem_row, M_EI_Elem_col)), M_EI_Elem])
        CD = np.vstack([np.zeros((max(M_EI_Elem_row, N_EI_Elem_row) - N_EI_Elem_row, N_EI_Elem_col)), N_EI_Elem])
        EI_Elem = np.hstack([CC, CD])

        CE = np.vstack([np.zeros((max(M_Simple_row, N_Simple_row) - M_Simple_row, M_Simple_col)), M_Simple])
        CF = np.vstack([np.zeros((max(M_Simple_row, N_Simple_row) - N_Simple_row, N_Simple_col)), N_Simple])
        Simple = np.hstack([CE, CF])

        COMBINE = np.vstack([EI_Elem, Simple]).astype(int)
        f = AFS.reduceEII(COMBINE)
        return f

    def mult_bool(self, m, n, Logic_Ind):
        m_row, m_col = m.shape  # 获取矩阵行数与列数
        n_row, n_col = n.shape  # 获取矩阵行数与列数
        ############################解决矩阵大小不一致问题，找到后创建0矩阵进行补充######################################################
        m_zero = np.zeros((max(m_row, n_row) - m_row, m_col))
        m_new = np.vstack((m, m_zero))
        n_zero = np.zeros((max(m_row, n_row) - n_row, n_col))
        n_new = np.vstack((n, n_zero))
        row, col = np.shape(n_new)
        ########################################################################################################################
        t = m_row - n_row  # 矩阵大小判断基准
        logic = np.mat(np.zeros((row, col)))  # 创建矩阵，用0填充，防止拼接矩阵函数报错
        if Logic_Ind == 1:
            if t >= 0:
                for i in range(n_col):
                    logic1 = np.mat(np.zeros((m_row, m_col)))  # 创建矩阵，用0填充，防止拼接矩阵函数报错
                    RepNi = np.tile(n_new[:, i], (1, m_col))  # 按列遍历矩阵扩充，行数不变，列数扩大
                    for j in range(m_row):  # 遍历矩阵元素进行替换
                        for k in range(m_col):
                            if RepNi[j, k] == 0 and m_new[j, k] == 0:
                                logic1[j, k] = 0
                            else:
                                logic1[j, k] = 1
                    logic = np.hstack((logic, logic1))  # 拼接矩阵
                    logic = logic[:, [not np.all(logic[:, i] == 0) for i in range(logic.shape[1])]]  # 将元素均为0的列删除
            else:
                for i in range(m_col):
                    logic1 = np.mat(np.zeros((n_row, n_col)))  # 创建矩阵，用0填充，防止拼接矩阵函数报错
                    RepMi = np.tile(m_new[:, i], (1, n_col))  # 按列遍历矩阵扩充，行数不变，列数扩大
                    for j in range(n_row):  # 遍历矩阵元素进行替换
                        for k in range(n_col):
                            if RepMi[j, k] == 0 and n_new[j, k] == 0:
                                logic1[j, k] = 0
                            else:
                                logic1[j, k] = 1
                    logic = np.hstack((logic, logic1))
                    logic = logic[:, [not np.all(logic[:, i] == 0) for i in range(logic.shape[1])]]  # 将元素均为0的列删除
        elif Logic_Ind == 2:
            if t >= 0:
                for i in range(n_col):
                    RepNi = np.tile(n_new[:, i], (1, m_col))  # 按列遍历矩阵扩充，行数不变，列数扩大
                    cc = np.mat(RepNi + m_new)
                    cc[cc > 1] = 1
                    cc[RepNi == 0] = 0
                    cc[m_new == 0] = 0
                    logic = np.hstack((logic, cc))
                    logic = logic[:, [not np.all(logic[:, i] == 0) for i in range(logic.shape[1])]]  # 将元素均为0的列删除
            else:
                for i in range(m_col):
                    RepMi = np.tile(m_new[:, i], (1, n_col))  # 按列遍历矩阵扩充，行数不变，列数扩大
                    cc = np.mat(RepMi + n_new)
                    cc[cc > 1] = 1
                    cc[RepMi == 0] = 0
                    cc[n_new == 0] = 0
                    logic = np.concatenate((np.mat(logic), np.mat(cc)), axis=1)
                    logic = np.mat(logic)
            logic = logic[:, col:]
        return logic

    def EIImult(self, m, n):
        m = AFS.reduceEII(m)
        n = AFS.reduceEII(n)
        m_Sum = np.sum(m, axis=1)  # 按列计算矩阵元素和
        M_EI_Num = np.argwhere(m_Sum == 0)  # 找到元素和为0的索引值
        M_EI_Num = M_EI_Num[:, [not np.all(M_EI_Num[:, i] == 0) for i in range(M_EI_Num.shape[1])]].flatten()
        n_Sum = np.sum(n, axis=1)  # 按列计算矩阵元素和
        N_EI_Num = np.argwhere(n_Sum == 0)  # 找到元素和为0的索引值
        N_EI_Num = N_EI_Num[:, [not np.all(N_EI_Num[:, i] == 0) for i in range(N_EI_Num.shape[1])]].flatten()
        M_EI_Elem = m[:M_EI_Num[1], :]
        N_EI_Elem = n[:N_EI_Num[1], :]
        M_Simple = m[M_EI_Num[1]:, :]
        N_Simple = n[N_EI_Num[1]:, :]
        Bool_M_EI_Elem = AFS.tran_bool(M_EI_Elem)
        Bool_N_EI_Elem = AFS.tran_bool(N_EI_Elem)
        Bool_M_Simple = AFS.tran_bool(np.mat(M_Simple))
        Bool_N_Simple = AFS.tran_bool(N_Simple)
        Logic_Elem = AFS.mult_bool(np.mat(Bool_M_EI_Elem), np.mat(Bool_N_EI_Elem), 1)
        Logic_Simple = AFS.mult_bool(np.mat(Bool_M_Simple), np.mat(Bool_N_Simple), 2)
        Logic_Elem = AFS.tran_EI_mat(Logic_Elem)
        Logic_Simple = AFS.tran_EI_mat(Logic_Simple)
        f = np.concatenate((Logic_Elem, Logic_Simple), axis=0)
        ans = AFS.reduceEII(f)
        return (ans)

    #############################################两个模糊复杂集E#I代数系统##############################################
    def reduceEIco_bool(self, m):
        row, col = np.shape(m)  # 获取矩阵行数与列数
        sum_sort = np.sum(-m, axis=0)  # 按列求取元素和，输出为行向量
        sum_sort = -sum_sort  # 倒序排列

        sum_sort_index = np.argsort(-sum_sort)  # 输出原始元素所在索引位置

        Reserve_EI = np.mat(np.zeros((row, col)))  # 创建与原始数据大小相同的0矩阵

        m = np.mat(m[:, sum_sort_index])
        m = np.reshape(m, (row, col))
        if col == 1:  # 判定条件，单列数据不参与运算
            N = m

        else:
            for i in range(col):
                if np.all(m[:, i] == 0):  # 判定条件，搜索矩阵每列全部元素是否为0
                    # np.all(np.array)对矩阵所有元素做与操作，所有为True则返回True
                    # np.any(np.array)对矩阵所有元素做或运算，存在True则返回True
                    continue
                else:
                    Reserve_EI[:, i] = m[:, i]  # 提取到导入数据按列赋值中间变量
                    Zero_Element = np.argwhere((Reserve_EI[:, i] == 0))  # 返回该列每个元素为零的索引值
                    Zero_Sum_mat = np.mat(np.zeros((len(Zero_Element), col)))  # 创建0行向量计算矩阵空间
                    for k in range(len(Zero_Element)):
                        Zero_Sum_mat[k, :] = (m[Zero_Element[:, 0][k], :])  # 矩阵空间填充
                    Zero_Sum = np.sum(Zero_Sum_mat, axis=0)  # 计算该列所有为零位置在原有数据结构中每一列的元素和
                    Judge_EI = np.where(np.array(Zero_Sum) == 0, 1, 0)  # 将具体数据转换为bool类型进行判断
                    Del_EI = np.argwhere(Judge_EI == 1)  # 找到判断为1的列索引
                    for j in range(len(Del_EI)):
                        Del_EI_num = Del_EI[:, 1][j]
                        m[:, Del_EI_num] = 0  # 将上述判断为真的列向量修正为0，即初始化
                    N = Reserve_EI[:, [not np.all(Reserve_EI[:, i] == 0) for i in range(Reserve_EI.shape[1])]]

        return N

    def EIcomult(self, x, y):
        boolm = AFS.tran_bool(x)
        booln = AFS.tran_bool(y)
        rbm = AFS.reduceEIco_bool(np.mat(boolm))
        rbn = AFS.reduceEIco_bool(np.mat(booln))
        rowm, colm = np.shape(rbm)
        rown, coln = np.shape(rbn)
        dima = np.zeros((max(rowm, rown) - rowm, colm))
        dimb = np.zeros((max(rowm, rown) - rown, coln))
        boolmf = np.concatenate((rbm, dima), axis=0)
        boolnf = np.concatenate((rbn, dimb), axis=0)
        rowbn, colbn = boolnf.shape
        T = colm - coln
        Logicor = np.empty((rowbn, colbn), dtype=object)
        if T > 0:
            for i in range(coln):
                repmatm = np.tile(boolnf[:, i], (1, colm))
                matrepmatm = np.mat(repmatm)
                cc = np.mat(boolmf + matrepmatm)
                cc[cc > 1] = 1
                cc[boolmf == 0] = 0
                cc[matrepmatm == 0] = 0
                Logicor = np.concatenate((Logicor, cc), axis=1)
                Logicorf = Logicor[:, colbn:]
                Logicorfm = np.mat(Logicorf)
                final = AFS.reduceEIco_bool(Logicorfm)
        else:
            for i in range(colm):
                repmatm = np.tile(boolmf[:, i], (1, coln))
                matrepmatm = np.mat(repmatm)
                cc = np.mat(boolnf + matrepmatm)
                cc[cc > 1] = 1
                cc[boolnf == 0] = 0
                cc[matrepmatm == 0] = 0
                Logicor = np.concatenate((Logicor, cc), axis=1)
                Logicorf = Logicor[:, colbn:]
                Logicorfm = np.mat(Logicorf)
                final = AFS.reduceEIco_bool(Logicorfm)
        ans = AFS.tran_EI_mat(final)
        return ans

    def EIcosum(self, m, n):
        m_row, m_col = m.shape
        n_row, n_col = n.shape

        CC = np.vstack([np.zeros((max(m_row, n_row) - m_row, m_col)), m]).astype(int)
        CD = np.vstack([np.zeros((max(m_row, n_row) - n_row, n_col)), n]).astype(int)

        COMBINE = np.hstack([CC, CD])
        f = AFS.reduceEIco(COMBINE)

        return (f)

    def less_eico(self, m, n):
        Bool_m = AFS.tran_bool(m)  # 转换为bool数值类型
        Bool_n = AFS.tran_bool(n)  # 转换为bool数值类型

        Bool_m_re = AFS.reduceEIco_bool(Bool_m)  # 约简用布尔矩阵表示的EM中的一个元素
        Bool_n_re = AFS.reduceEIco_bool(Bool_n)  # 约简用布尔矩阵表示的EM中的一个元素

        Bool_m_row, Bool_m_col = Bool_m_re.shape  # 获取矩阵大小
        Bool_n_row, Bool_n_col = Bool_n_re.shape  # 获取矩阵大小

        Size_Row = max(Bool_m_row, Bool_n_row)  # 比较两矩阵获取最大行数

        Bool_m_zero = np.zeros((Size_Row - Bool_m_row, Bool_m_col))  # 初始化0矩阵
        Bool_m = np.vstack((Bool_m_re, Bool_m_zero))  # 拼接0矩阵

        Bool_n_zero = np.zeros((Size_Row - Bool_n_row, Bool_n_col))  # 初始化0矩阵
        Bool_n = np.vstack((Bool_n_re, Bool_n_zero))  # 拼接0矩阵

        for i in range(Bool_m_col):
            Tran = np.tile(Bool_m[:, i], (1, Bool_n_col))  # 按列遍历矩阵扩充，行数不变，列数扩大
            if (((np.sum((Tran <= Bool_n).astype(int), axis=0)) == Size_Row).any().astype(
                    int) == 0):  # 判断语句，获取布尔类型比较结果且强制转化为0/1数值
                f = 0
                break
            else:
                f = 1
                continue
        return f

    def equalEI_co(self, m, n):
        if AFS.less_eico(m, n) == 1 and AFS.less_eico(n, m) == 1:
            f = 1
        else:
            f = 0
        return (f)

    def reduceEIco(self, m):
        if np.any(m[0, :] != 0):
            print('the first row of input matrix must be zeros.')
        M = AFS.compactei(m)  # 矩阵压缩
        m = AFS.tran_bool(M)  # 数值转化为bool类型
        Reserve_EI = AFS.reduceEIco_bool(m)  # 矩阵缩减
        N = AFS.tran_EI_mat(Reserve_EI)  # 矩阵缩减
        return N

    #############################################数据集AFS三维结构生成##############################################
    def tran_simple_concept_index(self, m):
        row, col = m.shape
        m_pool = pd.isnull(m)
        for i in range(row):
            for j in range(col):
                m_pool[i, j] = not m_pool[i, j] == True
        m_pool_int = m_pool.astype("int")
        Nnz_Sum = np.sum(m_pool_int)
        SC_Index = np.argwhere(m_pool_int == 1)  # 找到判断为1的列索引
        SC_Index_sort = SC_Index[np.lexsort(SC_Index.T)]
        for k in range(Nnz_Sum):
            SC_Index_row = SC_Index_sort[k, 0]
            SC_Index_col = SC_Index_sort[k, 1]
            m_pool_int[SC_Index_row, SC_Index_col] = 2 * (k + 1) - 1
        return (m_pool_int)

    def gen_structure(self, Parameter_Mat, Data_Mat):
        P_Mat_Size = np.shape(Parameter_Mat)
        D_Mat_Size = np.shape(Data_Mat)
        if P_Mat_Size[1] != D_Mat_Size[1]:
            print('the column numbers of the matrices do not match.')
        else:
            Feature_Parameter_Index = ~ pd.isnull(Parameter_Mat)
            Feature_Parameter_Index = Feature_Parameter_Index.astype(int)
            SC_Index = AFS.tran_simple_concept_index(Parameter_Mat)
            Str_Mat = np.zeros([D_Mat_Size[0], np.max(SC_Index) + 1, D_Mat_Size[0]])
            for i in range(D_Mat_Size[0]):
                Repxi = np.tile(Data_Mat[i, :], (D_Mat_Size[0], 1))
                for p in range(np.shape(Feature_Parameter_Index)[0]):
                    Is1_Index = np.nonzero(Feature_Parameter_Index[p, :])
                    Rep_PMatp = np.tile(Parameter_Mat[p, Is1_Index[-1]], (D_Mat_Size[0], 1))
                    Logicv = [abs(np.array(Repxi[:, Is1_Index[-1]]) - np.array(Rep_PMatp)) <= abs(
                        np.array(Data_Mat[:, Is1_Index[-1]]) - np.array(Rep_PMatp))]
                    Neg_Logicv = [abs(np.array(Repxi[:, Is1_Index[-1]]) - np.array(Rep_PMatp)) >= abs(
                        np.array(Data_Mat[:, Is1_Index[-1]]) - np.array(Rep_PMatp))]
                    Str_Mat[:, SC_Index[p, Is1_Index[-1]] - 1, i] = np.array(Logicv, dtype="int").swapaxes(0, 1)
                    Str_Mat[:, SC_Index[p, Is1_Index], i] = np.array(Neg_Logicv, dtype="int").swapaxes(0, 1)
        return Str_Mat

    #############################################格值隶属度计算函数##############################################
    def under_A_xi(self, Str_Mat, Set_Simple_Concept, Index_Xi):
        x = np.flatnonzero(Set_Simple_Concept)  # 注意，此处只能将简单概念放置， 不存在复杂概念， 就是交的运算, 函数意思返回非零元素位置
        #ans = Set_Simple_Concept.ravel()[x]  # 转为一维数组
        ans = x
        midindex = Set_Simple_Concept[ans[0] - 1]
        before_sum = Str_Mat[:, midindex - 1, Index_Xi]
        sum = np.sum((before_sum), axis=1)
        f = len(midindex)
        x = np.mat(np.where(f == sum)).transpose()
        final = x.ravel()
        return final
    def low_mat(self,m):
        m1 = m.copy(deep=True)
        m2 = m.copy(deep=True)
        m3 = m.copy(deep=True)
        m4 = m.copy(deep=True)
        for i in m.columns:
            hj1_max = np.max(np.mat(m.loc[:, [i]]))
            hj2_min = np.min(np.mat(m.loc[:, [i]]))
            hj3_adv = np.sum(np.mat(m.loc[:, [i]]), axis=0) / len(m.index)
            hj_list = []
            for j in m.index:
                hj = abs(np.mat(m.loc[[j], [i]]) - hj3_adv)
                hj_list.append(hj)
            hj4_max_num = np.max(hj_list)
            hj5_min_num = np.min(hj_list)
            for k in m.index:
                low_mi1_xj = (hj1_max - np.mat(m.loc[[k], [i]])) / (hj1_max - hj2_min)
                low_mi2_xj = (hj4_max_num - abs(np.mat(m.loc[[k], [i]]) - hj3_adv)) / (hj4_max_num - hj5_min_num)
                low_mi3_xj = (abs(np.mat(m.loc[[k], [i]]) - hj3_adv) - hj5_min_num) / (hj4_max_num - hj5_min_num)
                low_mi4_xj = (np.mat(m.loc[[k], [i]]) - hj2_min) / (hj1_max - hj2_min)
                m1.loc[[k], [i]] = low_mi1_xj
                m2.loc[[k], [i]] = low_mi2_xj
                m3.loc[[k], [i]] = low_mi3_xj
                m4.loc[[k], [i]] = low_mi4_xj
        return m4
    def degree_xi(self, Lou_Mat, Str_Mat, Fuzzy_Set, Index_Xi, Logic_Ind):
        Total = np.sum((Lou_Mat), axis=0)
        Result = 0
        row, col = np.shape(Fuzzy_Set)
        for k in range(col):
            Concept = Fuzzy_Set[:, k]
            f = np.argwhere(Concept != 0)
            Concept = Concept[f[0][0], :]
            rowC, colC = np.shape(Concept)
            if Logic_Ind == 1:
                A_Tao = AFS.under_A_xi(Str_Mat, np.array(Concept), Index_Xi)
                Sum = np.sum((Lou_Mat[A_Tao, Concept - 1]), axis=1)
                To = Total[0, Concept - 1]
                TT = Sum / To
                Result = np.max((Result, np.prod(TT)))
            elif Logic_Ind == 2:
                A_Tao = AFS.under_A_xi(Str_Mat, np.array(Concept), Index_Xi)
                Sum = np.sum((Lou_Mat[A_Tao, Concept - 1]), axis=1)
                To = Total[0, Concept - 1]
                TT = Sum / To
                Result = np.max((Result, np.min(TT)))
            else:
                TT = 1
                for i in range(rowC):
                    T = np.argwhere(Str_Mat[Index_Xi - 1, :, Concept[i] - 1])
                    Sum = np.sum((Lou_Mat[T, Concept[i] - 1]), axis=1)
                    Sumf = Sum[0, 2]
                    To = Total[0, Concept[i] - 1]
                    so = Sumf / To
                    TT = np.min((TT, so))
                Result = np.max((Result, TT))
        return Result
    def EIgen_degreelist(self, Lou_mat, Str_Mat, Fuzzy_Set, Logic_Ind):
        M = np.zeros((np.shape(Lou_mat)[0], 1))
        for i in range(np.shape(Lou_mat)[0]):
            M[i, 0] = AFS.degree_xi(Lou_mat, Str_Mat, Fuzzy_Set, i, Logic_Ind)
        return M
    def EIIdegree(self,EIlist, Logi_index, conbi_num):
        def listgen(n):
            return map(int, ''.join(map(str, range(1, n + 1))))  # 生成列表
        if Logi_index == 1:
            row, col = np.shape(EIlist)
            #xlist = list(listgen(row))
            numbers = list(range(1,row+1))
            twocombi = combinations(numbers, conbi_num)
            T = np.mat([list(item) for item in twocombi])
            sum = 0
            for i in range(int(comb(row, conbi_num))):
                c = (np.min((EIlist[T[i] - 1]), axis=1))
                sum = sum + c
            degree = np.zeros((1, int(comb(row, conbi_num))))
            for m in range(int(comb(row, conbi_num))):
                c = (np.min((EIlist[T[m] - 1]), axis=1))
                degree[0][m] = c[0][0] / sum[0][0]
            index_all = []
            for h in range(len( degree[0])):
                index_all.append(T[h])
            index = np.where( degree[0] == np.max( degree[0]))
            return index_all,  degree[0], T[index[0]],  degree[0][index]
        else:
            row, col = np.shape(EIlist)
            xlist = list(listgen(row))
            twocombi = combinations(xlist, conbi_num)
            T = np.mat([list(item) for item in twocombi])
            c = []
            for i in range(int(comb(row, conbi_num))):
                c = np.append(c, [np.min((EIlist[T[i] - 1]), axis=1)])
            list2 = []
            for j in range(int(comb(row, conbi_num))):
                x = [m for m in c if m < (np.min((EIlist[T[j] - 1]), axis=1))]
                xlen = len(x)
                list2 = np.append(list2, [xlen / comb(row, conbi_num)])
            degree = list2
            index_all = []
            for h in range(len(degree)):
                index_all.append(T[h])
            index = np.where(degree == np.max(degree))
            return index_all, degree[0], T[index[0]], degree[0][index]
    #  注： 隶属度函数计算示例
    # set valuable
    # ##define Str_Mat
    # Str_Mat = np.zeros([4, 4, 6])
    # Str_Mat[0, :, :] = np.mat([[1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0]])
    # Str_Mat[1, :, :] = np.mat([[1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0]])
    # Str_Mat[2, :, :] = np.mat([[1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 1, 0]])
    # Str_Mat[3, :, :] = np.mat([[1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]])
    # # define Lou_Mat
    # Lou_Mat = np.mat([[0, 1, 1, 0, 0.1667, 0.83333], [0.2727, 0.7273, 0.7273, 0.2727, 0.6667, 0.3333],
    #                   [0.5455, 0.4545, 0.4545, 0.5455, 0.8333, 0.1667], [1, 0, 0, 1, 0, 1]])
    #
    # Fuzzy_Set = np.mat([[0, 1], [0, 3]]).transpose()
    # Index_Xi = 2
    # Logic_Ind = 1
    "注：Index_Xi-1和Set_Simple_Concept-1是由于matlab和python的索引值不一致导致的，在python中需要从零0开始，"
    "所以需要相减，因此结果也会变小，结果会-1"
    '已经在代码中完善， under_A_xi 将比标准index索引-1'


if __name__ == "__main__":
    #################################################测试数据集###############################################################
    """手动输入函数的调用
    m1 = np.mat(AFS_input())
    m2 = np.mat(AFS_input())
    m3 = np.mat(AFS_input())
    m = np.vstack((m1,m2,m3)).transpose()
    n1 = np.mat(AFS_input())
    n2 = np.mat(AFS_input())
    """
    ##代码输入数值示例##
    """两个复杂模糊概念比较分析
    m = np.mat([[0, 1, 4, 7], [0, 0, 1, 4], [0, 2, 5, 5]]).transpose()  # 创建示例矩阵
    n = np.mat([[0, 0, 1, 4], [0, 2, 5, 5]]).transpose()  # 创建示例矩阵
    """
    """两个复杂模糊概念的EI代数系统
        m = np.mat([[0, 1, 2, 3, 3, 4], [0, 0, 3, 2, 5, 4], [0, 1, 1, 6, 7, 4], [0, 0, 0, 2, 4, 5]]).transpose()
        n = np.mat([[0, 0, 2, 2, 3, 4], [0, 1, 4, 5, 6, 4], [0, 0, 0, 7, 6, 6]]).transpose()
        m = np.mat([[0, 0, 1,3,4], [0, 0, 0,3,2], [0, 0, 2,6,8], [0, 1, 4,5,8]]).transpose()
        n = np.mat([[0, 1, 3], [0, 0, 2], [0, 0, 8]]).transpose()
    """
    """两个复杂模糊概念的EII代数系统
        m = np.mat([[0, 1, 2, 3, 4, 0, 4], [0, 2, 3, 4, 5, 0, 5]]).transpose()  # 创建示例矩阵
        n = np.mat([[0, 1, 2, 3, 0, 0, 3, 4], [0, 0, 2, 3, 0, 3, 4, 5], [0, 0, 1, 3, 0, 3, 4, 7]]).transpose()  # 创建示例矩阵
    """
    """AFS三维结构
    Parameter_Mat=np.mat([5.4,6.5,6.0]).transpose()
    Data_Mat=np.mat([6.5,6.2,5.9,5.4]).transpose()
    """
    """格值隶属度
    Str_Mat = np.zeros([4,4,6])
    Str_Mat [0,:,:] = np.mat([[1,1,1,1,1,1],[0,1,1,0,0,1],[0,1,1,0,0,1],[0,1,1,0,1,0]])
    Str_Mat [1,:,:] = np.mat([[1,0,0,1,1,0],[1,1,1,1,1,1],[0,1,1,0,0,1],[0,1,1,0,1,0]])
    Str_Mat [2,:,:] = np.mat([[1,0,0,1,1,0],[1,0,0,1,1,0],[1,1,1,1,1,1],[0,1,1,0,1,0]])
    Str_Mat [3,:,:] = np.mat([[1,0,0,1,0,1],[1,0,0,1,0,1],[1,0,0,1,0,1],[1,1,1,1,1,1]])
    Set_Simple_Concept = np.mat([0,2]).transpose()
    Index_Xi = 1
    """
    ''' 两个模糊复杂集E#I代数系统
    m = np.mat([[0, 1, 2, 3, 3, 4], [0, 0, 3, 2, 5, 4], [0, 1, 1, 6, 7, 4], [0, 0, 0, 2, 4, 5]]).transpose()  # 创建示例矩阵
    n = np.mat([[0, 0, 2, 2, 3, 4], [0, 1, 4, 5, 6, 4], [0, 0, 0, 7, 6, 6]]).transpose()  # 创建示例矩阵
    '''
    #############################################单个复杂模糊概念转换及约简示例##########################################

    # tran_bool
    '''

    m = np.mat([[0, 1, 3, 6], [0, 0, 5, 9]]).transpose()
    AFS = AFS()
    print(AFS.tran_bool(m))
    '''
    '''
    m = np.mat([[0, 1, 2, 3, 3, 4], [0, 0, 3, 2, 5, 4], [0, 1, 1, 6, 7, 4], [0, 0, 0, 2, 4, 5]]).transpose()
    AFS = AFS()
    print(AFS.reduceEI(m))
    '''
    '''
    m = np.mat([[0,1,2,3,3,4],[0,0,3,2,5,4],[0,1,1,6,7,4],[0,0,0,2,4,5]]).transpose()  # 创建示例矩阵
    AFS = AFS()
    print(AFS.compactei(m))
    '''
    '''
    m = np.mat([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0]]) .transpose()
    AFS=AFS()
    print(AFS.reduceEI_bool(m))
    '''
    '''
    m = np.mat([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0]]).transpose()
    AFS = AFS()
    print(AFS.tran_EI_mat(m))
    # compacteii
    '''
    '''
       m = np.mat([[0, 1, 3, 6], [0, 0, 5, 9]]).transpose()
       AFS = AFS()
       print(AFS.negateEI(m))
    '''
    #################################################两个复杂模糊概念的EI代数系统###############################################
    # negateEI
    # reduceEI_bool
    # reduceEI
    # compactei
    # tran_EI_mat
    #############################################两个复杂模糊概念比较分析###########################################
    # less_ei
    '''
    m = np.mat([[0,1,4,7],[0,2,5,5]]).transpose() #创建示例矩阵
    n = np.mat([[0,0,1,4],[0,2,5,5]]).transpose() #创建示例矩阵
    AFS=AFS()
    print(AFS.less_ei(m, n))
    '''
    # equal_ei
    '''
    m = np.mat([[0, 1, 4, 7], [0, 0, 1, 4], [0, 2, 5, 5]]).transpose()  # 创建示例矩阵
    n = np.mat([[0, 0, 1, 4], [0, 2, 5, 5]]).transpose()  # 创建示例矩阵
    AFS=AFS()
    print(AFS.equal_ei(m, n))
    '''
    #################################################两个复杂模糊概念的EII代数系统###############################################
    # EIsum
    '''
    m = np.mat([[0, 0, 1, 3 ,4 ],[0 ,0, 0, 3, 2],[0, 0, 2 ,6 ,8],[0, 1, 4 ,5 ,8]]).transpose()
    n = np.mat([[0 ,1 ,3] ,[0 ,0 ,2],[ 0 ,0, 8]]).transpose()
    AFS = AFS()
    print(AFS.EIsum(m, n))
    '''
    # EImult
    '''
    m = np.mat([[0, 1, 2, 3, 3, 4], [0, 0, 3, 2, 5, 4], [0, 1, 1, 6, 7, 4], [0, 0, 0, 2, 4, 5]]).transpose()
    n = np.mat([[0, 0, 2, 2, 3, 4], [0, 1, 4, 5, 6, 4], [0, 0, 0, 7, 6, 6]]).transpose()
    AFS = AFS()
    print(AFS.EImult(m, n))
    '''
    ###############################################两个复杂模糊概念的EII代数系统#################################################
    # compacteii
    '''
    m = np.mat([[0, 1, 2, 3, 3, 4, 0, 3, 3, 4], [0, 0, 3, 2, 5, 4, 0, 0, 2, 5], [0, 1, 1, 6, 7, 4, 0, 1, 4, 7],
                [0, 0, 0, 2, 4, 5, 0, 1, 2, 5]]).transpose()
    AFS = AFS()
    print(AFS.compacteii(m))

    '''
    # reduceEII
    '''
    m = np.mat([[0, 1, 2, 3, 3, 4, 0, 3, 3, 4], [0, 0, 3, 2, 5, 4, 0, 0, 2, 5], [0, 1, 1, 6, 7, 4, 0, 1, 4, 7],
                [0, 0, 0, 2, 4, 5, 0, 1, 2, 5]]).transpose()
    AFS = AFS()
    print(AFS.reduceEII(m))
    '''
    # EIISUM
    '''
    m = np.mat([[0,1,2,3,4,0,4],[0,2,3,4,5,0,5]]).transpose() #创建示例矩阵
    n = np.mat([[0,1,2,3,0,0,3,4],[0,0,2,3,0,3,4,5],[0,0,1,3,0,3,4,7]]).transpose() #创建示例矩阵
    AFS = AFS()
    print(AFS.EIISUM(m,n))
    '''
    # mult_bool
    '''
    m = np.mat([[0,0,0,1,0],[0,0,0,0,1]]).transpose() #创建示例矩阵
    n = np.mat([[0,0,1,1,1,0,0],[0,0,1,1,0,0,1]]).transpose() #创建示例矩阵
    AFS = AFS()
    print(AFS.mult_bool(m,n,2))
    '''
    # EIImult
    '''
    m = np.mat([[0,1,2,3,4,0,4],[0,2,3,4,5,0,5]]).transpose() #创建示例矩阵
    n = np.mat([[0,1,2,3,0,0,3,4],[0,0,2,3,0,3,4,5],[0,0,1,3,0,3,4,7]]).transpose() #创建示例矩阵
    AFS = AFS()
    print(AFS.EIImult(m,n))
    '''
    ###################################################两个模糊复杂集E#I代数系统################################################
    # reduceEIco_bool
    '''
    AFS = AFS()

    a = np.mat([[0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1]]).transpose()
    b = np.mat([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0]]).transpose()
    print(AFS.reduceEIco_bool(a))
    '''
    # EIcomult
    '''
    AFS = AFS()
    
    m = np.mat([[0, 1, 2, 3, 3, 4], [0, 0, 3, 2, 5, 4], [0, 1, 1, 6, 7, 4], [0, 0, 0, 2, 4, 5]]).transpose()
    n = np.mat([[0, 0, 2, 2, 3, 4], [0, 1, 4, 5, 6, 4], [0, 0, 0, 7, 6, 6]]).transpose()

    print(AFS.EIcomult(m, n))
    '''
    # EIcosum
    '''
    AFS = AFS()

    m = np.mat([[0, 1, 2, 3, 3, 4, 1], [0, 0, 3, 2, 5, 4, 1], [0, 1, 1, 6, 7, 4, 1], [0, 0, 0, 2, 4, 5, 1]]).transpose()
    n = np.mat([[0, 0, 2, 2, 3, 4], [0, 1, 4, 5, 6, 4], [0, 0, 0, 7, 6, 6]]).transpose()
    print(AFS.EIcosum(m,n))
    '''
    # less_eico
    '''
    AFS=AFS()
    m = np.mat([[0, 1, 4, 7], [0, 2, 5, 5]]).transpose()
    n = np.mat([[0, 0, 1, 4], [0, 2, 5, 5]]).transpose()
    print(AFS.less_eico(m,n))
    '''
    # equalEI_co
    ''' 
    AFS = AFS()
    m = np.mat([[0, 1, 4, 7], [0, 0, 1, 4], [0, 2, 5, 5]]).transpose()  # 创建示例矩阵
    n = np.mat([[0, 1, 4, 7], [0, 2, 5, 5]]).transpose()  # 创建示例矩阵
    print(AFS.equalEI_co(m, n))

    
    
    '''
    # reduceEIco
    '''
    AFS=AFS()
    m = np.mat([[0,1,2,3,3,4], [0,0,3,2,5,4], [0,1,1,6,7,4], [0,0,0,2,4,5]]).transpose()

    print(AFS.reduceEIco(m))
    '''
    ########################################################数据集AFS三维结构生成##############################################
    # tran_simple_concept_index
    '''
    m = np.mat([[4.3,5.8433,7.9],[math.nan,3.054,4.4],[math.nan,math.nan,6.9],[math.nan,1.1987,2.5]]).transpose() #创建示例矩阵
    AFS = AFS()
    print(AFS.tran_simple_concept_index(m))
    '''
    # gen_structure
    '''

    Parameter_Mat = np.mat([5.4, 6.5, 6.0]).transpose()

    Data_Mat = np.mat([6.5, 6.2, 5.9, 5.4]).transpose()

    AFS = AFS()
    print(AFS.gen_structure( Parameter_Mat, Data_Mat))
    
    Parameter_Mat = np.mat([[None,None,5],[None,None,5]]).astype(np.float64).transpose()
    参数矩阵的建立
    '''
    ###########################################################格值隶属度计算函数##############################################
    # under_A_xi
    '''
    AFS=AFS()
    Str_Mat = np.zeros([4, 4, 6])
    Str_Mat[0, :, :] = np.mat([[1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0]])
    Str_Mat[1, :, :] = np.mat([[1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0]])
    Str_Mat[2, :, :] = np.mat([[1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 1, 0]])
    Str_Mat[3, :, :] = np.mat([[1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]])

    Set_Simple_Concept = np.mat([[1]])
    Index_Xi = 3
    ans = AFS.under_A_xi(Str_Mat, Set_Simple_Concept, Index_Xi)
    print(ans)
    '''
    # degree_xi
    '''
    # set valuable
    #define Str_Mat
    AFS=AFS()
    Str_Mat = np.zeros([4, 4, 6])
    Str_Mat[0, :, :] = np.mat([[1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0]])
    Str_Mat[1, :, :] = np.mat([[1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 0]])
    Str_Mat[2, :, :] = np.mat([[1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 1, 0]])
    Str_Mat[3, :, :] = np.mat([[1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]])
    # define Lou_Mat
    Lou_Mat = np.mat([[0, 1, 1, 0, 0.1667, 0.83333], [0.2727, 0.7273, 0.7273, 0.2727, 0.6667, 0.3333],
                      [0.5455, 0.4545, 0.4545, 0.5455, 0.8333, 0.1667], [1, 0, 0, 1, 0, 1]])

    Fuzzy_Set = np.mat([[0, 1], [0, 3]]).transpose()
    Index_Xi = 2
    Logic_Ind = 2
    print(AFS.degree_xi(Lou_Mat, Str_Mat, Fuzzy_Set, Index_Xi, Logic_Ind))
    '''
    ###################################################################权重计算##############################################
    # low_mat
    """
    filename = "C:\\Users\\DELL\\Desktop\\low_mat.xlsx"
    xlsx = pd.ExcelFile(filename, engine="openpyxl")
    mcc = pd.read_excel(xlsx, 'Sheet1',index_col=0,header=0)
    m1,m2,m3,m4 =  AFS().low_mat(mcc)
    print(m4)
    """
    ###################################################################数据分解##############################################
    # deco
    """
    m = AFS_readxlsx()
    AFS_S = AFS_res(m)
    AFS_S = AFS_res_mul(m)
    """
    # deco-pearson
#m = AFS_readxlsx()
# EII代数函数说明
'''
EII代数第二个自变量为EIgenlist 返回值
'''
#AFS_SN = AFS_res_per_mul(m)
# AFS_SN = AFS_res咸蛋超人发表是咸蛋超人发表我色弱不能      去去q_mul(m)

#声明类
AFS = AFS()
#数据读取，数值提取
m = AFS_readxlsx()
mcc = np.mat(m)
#权重换算，数值提取
low_mat = AFS.low_mat(m)
weight = np.mat(low_mat)
#参数矩阵导入
parameter_mat = np.mat([[5],[5],[5],[5],[5],[5],[5]]).astype(np.float64).transpose()#第一结构
#parameter_mat = np.mat([[5],[5],[5],[5],[4],[4]]).astype(np.float64).transpose() #第二结构
#AFS结构生成
afsstructure = AFS.gen_structure(parameter_mat, mcc)
Str_Mat = afsstructure
#模糊概念集合，即M集合元素个数
Fuzzy_Set= np.mat([[0,3]]).transpose()
#算子选择
Logic_Ind= 1
#ei组合生成
EIlist=AFS.EIgen_degreelist(weight, Str_Mat, Fuzzy_Set, Logic_Ind)
#eii隶属度计算
for i in range(1,mcc.shape[0]):
    EIIdegree = AFS.EIIdegree(EIlist, 1, i)
    #print(EIIdegree[0])
    #数据导出
    #file_path = AFS_writexlsx()
    #print(file_path)
    file_path = 'C:\\Users\\DELL\\Desktop\\'
    file_name = f'匹配结果：{i}.xlsx'
    EIIdegree = pd.DataFrame(data = EIIdegree).transpose()
    #归一化
    min_max_scaler = MinMaxScaler()
    EIIdegree.loc[:, [1]] = min_max_scaler.fit_transform(np.mat(EIIdegree.loc[:, [1]]))
    #EIIdegree.to_excel(file_path+file_name,index=False)
    EIIdegree.to_csv(file_path + file_name, index=False)
    #数据结果还需要归一化，具体公式可以表示为：第一步：X_std = (X - X.min)/ (X.max(axis=0) - X.min
    # 第二步：X_scaled = X_std * (max - min) + min

