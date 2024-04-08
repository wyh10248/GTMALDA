# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:06:56 2023

@author: 28473
"""
from scipy.io import loadmat
import numpy as np
import torch
import numpy as np

def l21_norm(Mat):
    rows, cols = Mat.shape
    diagList = np.zeros(rows)

    for i in range(rows):
        add = np.sum(Mat[i, :]**2)
        diagList[i] = 1 / (2 * np.sqrt(add))

    diagTemp = np.diag(diagList)
    return diagTemp


def nmf(A, lncSimi, disSimi, beita, k, iterate):
    rows, cols = A.shape
    C = np.abs(np.random.rand(rows, k))
    D = np.abs(np.random.rand(cols, k))

    diag_cf = np.diag(np.sum(lncSimi, axis=1))
    diag_df = np.diag(np.sum(disSimi, axis=1))

    A1 = np.dot(lncSimi, A)
    A2 = np.dot(A, disSimi)
    A = np.maximum(A1, A2)

    AA = np.zeros_like(A)
    for j in range(cols):
        colList = A[:, j]
        AA[:, j] = (A[:, j] - np.min(colList)) / (np.max(colList) - np.min(colList))
    A = AA

    for step in range(iterate):
        Y = A - np.dot(C, D.T)
        B = l21_norm(Y)

        if beita > 0:
            BAD = np.dot(B, A) @ D + beita * lncSimi @ C
            BCDD = np.dot(B, C @ D.T @ D) + beita * diag_cf @ C
            C = np.multiply(C, BAD / BCDD)

        if beita > 0:
            ABC = A.T @ B @ C + beita * disSimi @ D
            DCBC = np.dot(D, C.T @ B @ C) + beita * diag_df @ D
            D = np.multiply(D, ABC / DCBC)

        scoreMat_NMF = A - np.dot(C, D.T)
        error = np.mean(np.abs(scoreMat_NMF)) / np.mean(A)
        #print(f'step={step}  error={error}')
    return C, D

# Example usage:
# if__name__="main"
# data_path = '../dataset/'
# data_set = 'data-R-D447-218/'
# A = np.loadtxt(data_path + data_set + 'disease-lncRNA.csv', delimiter=',')
# A=torch.tensor(A.T)
# disSimi1 = loadmat(data_path + data_set + 'diease_similarity_kernel-k=20-t=20-ALPHA=0.1.mat') 
# disSimi = disSimi1['WM']
# RNASimi1 = loadmat(data_path + data_set + 'RNA_s_kernel-k=20-t=20-ALPHA=0.1.mat')
# lncSimi = RNASimi1['WM']
# R,D=nmf(A, lncSimi, disSimi, 0.01, 64, 500)
# print(R.shape)
#print(D)
