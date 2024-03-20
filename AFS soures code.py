import numpy as np
import pandas as pd
from tkinter import filedialog
import tkinter as tk
import os
from scipy.special import comb
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
def AFS_readxlsx():

    application_window = tk.Tk()
    application_window.withdraw()
    filetypes = [('all files', '.*'), ('xlsx files', '.xlsx')]
    filepath = filedialog.askopenfilename(parent=application_window,
                                          initialdir=os.getcwd(),
                                          title="select a file:", filetypes=filetypes)
    """
    if messagebox.askokcancel("Quit", "confirm to quit？"):
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
    k = input('\n in put a sample or not (yes-1\\no-0)：')
    k = int(k)
    arr = []
    if k == 1:
        for i in range(99):
            pt = input('\n input fuzzy term：')
            pt = int(pt)
            arr.append(pt)
            if pt == -1:
                arr.remove(-1)
                return arr
    else:
        judge = input('\n confirm to quit(yes-1\\no-0)：')
        judge = int(judge)
        if judge == 1:
            for i in range(99):
                pt = input('\n input fuzzy term：')
                pt = int(pt)
                arr.append(pt)
                if pt == -1:
                    arr.remove(-1)
                    return arr
        else:
            exit()
def AFS_writexlsx(data):
    root = tk.Tk()
    root.withdraw()
    filetypes = [('all files', '.*'), ('xlsx files', '.xlsx')]
    filepath = filedialog.asksaveasfilename(initialdir="C:\\Users\\DELL\\Desktop", title="save_file",
                                            filetypes= filetypes)
    xlsx = pd.ExcelFile(filepath, engine="xlsxwriter")
    mcc = pd.read_excel(xlsx, 'Sheet1', header=0, index_col=0)
    return  xlsx
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
        for i in range(0, np.size(m, axis=0)):
            if np.sum(m_mat[i, :]) != 0:
                MB = m_mat[i, :]
                MC = m_mat[i, :]
                boundary = i
                break
        for j in range(boundary + 1, row):
            TR = np.min([MB[0, :], m_mat[j, :]], 0)
            if np.sum(TR, axis=1) != 0:
                MB = np.vstack((MB, m_mat[j, :]))
                m_mat[j, :] = MZE
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
        for i in range(0, np.size(m, axis=0)):
            if np.sum(m_mat[i, :]) != 0:
                MB = m_mat[i, :]
                MC = m_mat[i, :]
                boundary = i
                break
        k = float(input('\n input threshold:'))
        for j in range(boundary + 1, row):
            TR = pearsonr(np.array(MB[0, :]).flatten().transpose(), np.array(m_mat[j, :]).flatten().transpose())
            if TR[0] > k:
                MB = np.vstack((MB, m_mat[j, :]))
                m_mat[j, :] = MZE
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
            print(f'the {h + 1} structure :\n', AFS_S[h])
        return AFS_S
class AFS:

    def tran_bool(self, m):
        Num_Row = int(m.max())
        row, col = np.shape(m)
        receive = np.zeros((Num_Row, col))
        for i in range(col):
            No_Zero_Row = np.argwhere(m[:, i] != 0)
            for j in range(len(No_Zero_Row)):
                No_Zero_Row_num = No_Zero_Row[:, 0][j]
                m_tran_a = np.array(m)
                No_Zero_Elem = m_tran_a[No_Zero_Row_num:, i][0]
                receive[int(No_Zero_Elem - 1), i] = 1
        return receive

    def negateEI(self, m):
        Bool_M = AFS.tran_bool(m)
        Bool_M = AFS.reduceEI_bool(Bool_M)
        m_row, m_col = m.shape
        Bool_M_row, Bool_M_col = Bool_M.shape
        for i in range(m_col):
            for j in range(m_row):
                if (m[j, i] == 0):
                    continue
                elif (m[j, i] % 2 == 1):
                    m[j, i] = m[j, i] + 1
                else:
                    m[j, i] = m[j, i] - 1
        T = np.transpose(m[:, 0])
        Del_EI = np.argwhere(T != 0)

        Del_EI = np.transpose(Del_EI[:, [not np.all(Del_EI[:, i] == 0) for i in range(Del_EI.shape[1])]]).flatten()
        Bool_N = AFS.tran_bool(T[:, Del_EI])
        for j in range(1, Bool_M_col):
            T = np.transpose(m[:, j])
            Del_EI = np.argwhere(T != 0)
            Del_EI = np.transpose(Del_EI[:, [not np.all(Del_EI[:, i] == 0) for i in range(Del_EI.shape[1])]]).flatten()
            Bool_N = AFS.mult_bool(AFS.reduceEI_bool(Bool_N), AFS.tran_bool(T[:, Del_EI]), 1)
        return (Bool_N)

    def reduceEI_bool(self, m):
        row, col = np.shape(m)
        sum_sort = np.sort(np.sum(np.mat(m), axis=0)).astype("int")
        sum_sort_index = np.argsort(np.sum(np.mat(m), axis=0))
        Reserve_EI = np.mat(np.zeros((row, col)))
        m_tran_c = np.mat(np.zeros((row, col)))
        if sum_sort[0, 0] == 0:
            N = 0
        else:
            for i in range(col):
                m_tran_c[:, i] = m[:, sum_sort_index[0, i]].reshape(-1,
                                                                    1)
            if col == 1:
                N = m_tran_c
            else:
                for i in range(col):
                    if np.all(m_tran_c[:, i] == 0):
                        continue
                    else:
                        Reserve_EI[:, i] = m_tran_c[:, i]
                        Nnz_Element = np.argwhere((Reserve_EI[:, i] != 0))
                        Nnz_Sum_mat = np.mat(np.zeros((len(Nnz_Element), col)))
                        for k in range(len(Nnz_Element)):
                            Nnz_Sum_mat[k, :] = (m_tran_c[Nnz_Element[:, 0][k], :])
                        Nnz_Sum = np.sum(Nnz_Sum_mat, axis=0)
                        Judge_EI = np.where(np.array(Nnz_Sum) == sum_sort[0, i], 1, 0)
                        Del_EI = np.argwhere(Judge_EI == 1)
                        for j in range(len(Del_EI)):
                            Del_EI_num = Del_EI[:, 1][j]
                            m_tran_c[:, Del_EI_num] = 0
                N = Reserve_EI[:, [not np.all(Reserve_EI[:, i] == 0) for i in range(Reserve_EI.shape[1])]]
        return (N)

    def reduceEI(self, m):
        m_tran_a = np.array(m)
        if np.sum(m_tran_a[0]) != 0:
            print('the first row of input matrix must be zeros.')
        else:
            m_tran_a_cpt = AFS.compactei(m)
            m_tran_a_cpt_bool = AFS.tran_bool(m_tran_a_cpt)
            c = np.mat(m_tran_a_cpt_bool)
            Reserve_EI = AFS.reduceEI_bool(c)
            d = np.mat(Reserve_EI)
            N = AFS.tran_EI_mat(d)
        return (N)

    def compactei(self, m):
        if np.any(m[0, :] != 0):
            print("矩阵第一行必须为0")
        else:
            m_sort = np.sort(m, axis=0)
            Max_Num_Column = 0
            row, col = m_sort.shape
            for i in range(col):
                comp = np.unique(m_sort[:, i], axis=0)
                C_size = np.count_nonzero(comp)
                Max_Num_Column = max(Max_Num_Column, C_size)
                cc = np.zeros((row - C_size, 1))
                m_sort[:, i] = np.concatenate((cc, comp[1:]), axis=0)
            n = m_sort[row - Max_Num_Column - 1:][:]
            return n

    def tran_EI_mat(self, m):
        m_tran = np.mat(m)
        r, c = np.shape(m_tran)
        Set_M = np.mat(range(1, r + 1))
        Repmat_Set_M = np.mat(np.tile(Set_M.transpose(), (1, c)))
        x1 = np.zeros((1, c))
        Repmat_Set_M_f = np.multiply(Repmat_Set_M, m_tran)
        N = np.vstack([x1, Repmat_Set_M_f])
        N = AFS.compactei(N)
        return (N)

    def less_ei(self, x, y):
        Bool_x = AFS.tran_bool(x)
        Bool_y = AFS.tran_bool(y)

        Bool_x_re = AFS.reduceEI_bool(Bool_x)
        Bool_y_re = AFS.reduceEI_bool(Bool_y)

        Bool_x_row, Bool_x_col = Bool_x_re.shape
        Bool_y_row, Bool_y_col = Bool_y_re.shape

        Size_Row = max(Bool_x_row, Bool_y_row)

        Bool_x_zero = np.zeros((Size_Row - Bool_x_row, Bool_x_col))
        Bool_x = np.vstack((Bool_x_re, Bool_x_zero))
        Bool_y_zero = np.zeros((Size_Row - Bool_y_row, Bool_y_col))
        Bool_y = np.vstack((Bool_y_re, Bool_y_zero))
        for i in range(Bool_x_col):
            Tran = np.tile(Bool_x[:, i], (1, Bool_y_col))
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

    def EIsum(self, x, y):
        rowm, colm = np.shape(x)
        rown, coln = np.shape(y)
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

    def compacteii(self, m):
        row, col = m.shape
        k = -1

        for i in range(row):
            if (np.sum(m[i, :]) != 0):
                break
            k = k + 1
        if k == -1:
            print('the first row of input matrix must be zeros.')

        mat = m[k:, :]
        EI_Num = np.where(np.array(np.sum(mat, axis=1)) == 0, 1, 0)
        EI_Num = np.argwhere(EI_Num == 1)
        EI_Num = EI_Num[:,
                 [not np.all(EI_Num[:, i] == 0) for i in range(EI_Num.shape[1])]]
        if np.size(EI_Num, axis=0) == 1:
            print('the concept represented by input matrix must be a EII element.')

        EI_Elem = mat[0:EI_Num[1, 0], :]
        Simple = mat[EI_Num[1, 0]:, :]
        N = np.vstack((AFS.compactei(EI_Elem), AFS.compactei(Simple)))
        return N

    def reduceEII(self, m):
        if np.any(m[0, :] != 0):
            print('the first row of input matrix must be zeros.')
        M = AFS.compacteii(m)
        if np.size(M, axis=1) == 1:
            N = M
        else:
            Num_EI_Elem = np.where(np.array(np.sum(M, axis=1)) == 0, 1, 0)
            Num_EI_Elem = np.argwhere(Num_EI_Elem == 1)
            Num_EI_Elem = Num_EI_Elem[:, [not np.all(Num_EI_Elem[:, i] == 0) for i in range(Num_EI_Elem.shape[1])]]
            Bool_EI_Elem = AFS.tran_bool(M[0:Num_EI_Elem[1, 0], :])
            Bool_EI_Elem_row, Bool_EI_Elem_col = Bool_EI_Elem.shape
            sum_sort = np.sort(np.sum(Bool_EI_Elem, axis=0))
            sum_sort_index = np.argsort(np.sum(Bool_EI_Elem, axis=0))
            if sum_sort[0] == 0:
                N = M[:, sum_sort_index[0]]
            else:
                Bool_EI_Elem = Bool_EI_Elem[:, sum_sort_index]
                Simple = M[Num_EI_Elem[1, 0]:, sum_sort_index]
                Simple_row, simple__col = Simple.shape
                Reserve_EI = np.mat(np.zeros((Bool_EI_Elem_row, Bool_EI_Elem_col)))
                Reduced_Simple = np.mat(np.zeros((Simple_row, simple__col)))
                for i in range(Bool_EI_Elem_col):
                    if np.all(Bool_EI_Elem[:, i] == 0):
                        continue
                    else:
                        Reserve_EI[:, i] = Bool_EI_Elem[:, i].reshape(Bool_EI_Elem_row, 1)
                        Reduced_Simple[:, i] = Simple[:, i].reshape(Simple_row, 1)
                        Nnz_Element = np.where(np.array(Bool_EI_Elem[:, i]) != 0, 1, 0)
                        Nnz_Element = np.argwhere(Nnz_Element == 1)
                        Nnz_Element = Nnz_Element[:,
                                      [not np.all(Num_EI_Elem[:, i] == 0) for i in range(Num_EI_Elem.shape[1])]]
                        Nnz_Sum = np.sum(Bool_EI_Elem[Nnz_Element, :], axis=0)
                        Judge_EI = np.where(np.array(Nnz_Sum) == sum_sort[i], 1, 0)
                        Del_EI = np.argwhere(Judge_EI == 1)
                        for j in range(len(Del_EI)):
                            Del_EI_num = Del_EI[:, 1][j]
                            Bool_EI_Elem[:, Del_EI_num] = 0
                N = np.vstack([AFS.tran_EI_mat(Reserve_EI), Reduced_Simple])
                N = N[:, [not np.all(N[:, i] == 0) for i in range(N.shape[1])]]
        return N

    def EIISUM(self, m, n):
        m = AFS.reduceEII(m)
        n = AFS.reduceEII(n)

        m_Sum = np.sum(m, axis=1)
        M_EI_Num = np.argwhere(m_Sum == 0)
        M_EI_Num = M_EI_Num[:, [not np.all(M_EI_Num[:, i] == 0) for i in range(M_EI_Num.shape[1])]].flatten()

        n_Sum = np.sum(n, axis=1)
        N_EI_Num = np.argwhere(n_Sum == 0)
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
        m_row, m_col = m.shape
        n_row, n_col = n.shape
        m_zero = np.zeros((max(m_row, n_row) - m_row, m_col))
        m_new = np.vstack((m, m_zero))
        n_zero = np.zeros((max(m_row, n_row) - n_row, n_col))
        n_new = np.vstack((n, n_zero))
        row, col = np.shape(n_new)
        t = m_row - n_row
        logic = np.mat(np.zeros((row, col)))
        if Logic_Ind == 1:
            if t >= 0:
                for i in range(n_col):
                    logic1 = np.mat(np.zeros((m_row, m_col)))
                    RepNi = np.tile(n_new[:, i], (1, m_col))
                    for j in range(m_row):
                        for k in range(m_col):
                            if RepNi[j, k] == 0 and m_new[j, k] == 0:
                                logic1[j, k] = 0
                            else:
                                logic1[j, k] = 1
                    logic = np.hstack((logic, logic1))
                    logic = logic[:, [not np.all(logic[:, i] == 0) for i in range(logic.shape[1])]]
            else:
                for i in range(m_col):
                    logic1 = np.mat(np.zeros((n_row, n_col)))
                    RepMi = np.tile(m_new[:, i], (1, n_col))
                    for j in range(n_row):
                        for k in range(n_col):
                            if RepMi[j, k] == 0 and n_new[j, k] == 0:
                                logic1[j, k] = 0
                            else:
                                logic1[j, k] = 1
                    logic = np.hstack((logic, logic1))
                    logic = logic[:, [not np.all(logic[:, i] == 0) for i in range(logic.shape[1])]]
        elif Logic_Ind == 2:
            if t >= 0:
                for i in range(n_col):
                    RepNi = np.tile(n_new[:, i], (1, m_col))
                    cc = np.mat(RepNi + m_new)
                    cc[cc > 1] = 1
                    cc[RepNi == 0] = 0
                    cc[m_new == 0] = 0
                    logic = np.hstack((logic, cc))
                    logic = logic[:, [not np.all(logic[:, i] == 0) for i in range(logic.shape[1])]]
            else:
                for i in range(m_col):
                    RepMi = np.tile(m_new[:, i], (1, n_col))
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
        m_Sum = np.sum(m, axis=1)
        M_EI_Num = np.argwhere(m_Sum == 0)
        M_EI_Num = M_EI_Num[:, [not np.all(M_EI_Num[:, i] == 0) for i in range(M_EI_Num.shape[1])]].flatten()
        n_Sum = np.sum(n, axis=1)
        N_EI_Num = np.argwhere(n_Sum == 0)
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
    def reduceEIco_bool(self, m):
        row, col = np.shape(m)
        sum_sort = np.sum(-m, axis=0)
        sum_sort = -sum_sort

        sum_sort_index = np.argsort(-sum_sort)
        Reserve_EI = np.mat(np.zeros((row, col)))

        m = np.mat(m[:, sum_sort_index])
        m = np.reshape(m, (row, col))
        if col == 1:
            N = m

        else:
            for i in range(col):
                if np.all(m[:, i] == 0):

                    continue
                else:
                    Reserve_EI[:, i] = m[:, i]
                    Zero_Element = np.argwhere((Reserve_EI[:, i] == 0))
                    Zero_Sum_mat = np.mat(np.zeros((len(Zero_Element), col)))
                    for k in range(len(Zero_Element)):
                        Zero_Sum_mat[k, :] = (m[Zero_Element[:, 0][k], :])
                    Zero_Sum = np.sum(Zero_Sum_mat, axis=0)
                    Judge_EI = np.where(np.array(Zero_Sum) == 0, 1, 0)
                    Del_EI = np.argwhere(Judge_EI == 1)
                    for j in range(len(Del_EI)):
                        Del_EI_num = Del_EI[:, 1][j]
                        m[:, Del_EI_num] = 0
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
        Bool_m = AFS.tran_bool(m)
        Bool_n = AFS.tran_bool(n)

        Bool_m_re = AFS.reduceEIco_bool(Bool_m)
        Bool_n_re = AFS.reduceEIco_bool(Bool_n)

        Bool_m_row, Bool_m_col = Bool_m_re.shape
        Bool_n_row, Bool_n_col = Bool_n_re.shape

        Size_Row = max(Bool_m_row, Bool_n_row)

        Bool_m_zero = np.zeros((Size_Row - Bool_m_row, Bool_m_col))
        Bool_m = np.vstack((Bool_m_re, Bool_m_zero))

        Bool_n_zero = np.zeros((Size_Row - Bool_n_row, Bool_n_col))
        Bool_n = np.vstack((Bool_n_re, Bool_n_zero))

        for i in range(Bool_m_col):
            Tran = np.tile(Bool_m[:, i], (1, Bool_n_col))
            if (((np.sum((Tran <= Bool_n).astype(int), axis=0)) == Size_Row).any().astype(
                    int) == 0):
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
        M = AFS.compactei(m)
        m = AFS.tran_bool(M)
        Reserve_EI = AFS.reduceEIco_bool(m)
        N = AFS.tran_EI_mat(Reserve_EI)
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
        SC_Index = np.argwhere(m_pool_int == 1)
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

    def under_A_xi(self, Str_Mat, Set_Simple_Concept, Index_Xi):
        x = np.flatnonzero(Set_Simple_Concept)
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
            return map(int, ''.join(map(str, range(1, n + 1))))
        if Logi_index == 1:
            row, col = np.shape(EIlist)
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


if __name__ == "__main__":
      test
