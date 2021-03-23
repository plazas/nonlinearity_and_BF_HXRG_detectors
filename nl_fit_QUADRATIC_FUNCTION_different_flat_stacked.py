# coding=utf-8
from multiprocessing import Pool
from multiprocessing import cpu_count
import matplotlib as mpl
import sigma_clip
import glob
import badger
import matplotlib.patches as patches
from scipy import stats
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import optimize
import collections
import astropy.io.fits as pf
import os
import sys
import subprocess as S
import numpy as np
import matplotlib
matplotlib.use('Pdf')


# Derive NL correction from flat at one illumination level; apply it to other at a different illumination level


# Do the plotting here
plt.minorticks_on()
# plt.tight_layout()

# We do not have matplotlib 1.1, with the 'style' package. Modify the matplotlibrc file parameters instead
mpl.rc('lines', linewidth=1, color='black', linestyle='-')
mpl.rc('font', family='serif', weight='normal', size=10.0)
mpl.rc('text',  color='black', usetex=False)
mpl.rc('axes',  edgecolor='black', linewidth=1, grid=False, titlesize='x-large',
       labelsize='x-large', labelweight='normal', labelcolor='black')
mpl.rc('axes.formatter', limits=[-4, 4])

mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0

mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['ytick.minor.pad'] = 8
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rc('legend', numpoints=1, fontsize='x-large', shadow=False, frameon=False)


sigma_cut = 3.0
prop = fm.FontProperties(size=7)
loc_label = 'upper right'
gain = 2.7  # e/ADU


# QUADRATIC POLYNOMIAL
fit_order = 2


def fitfunc(x, p0, p1, p2):
    x = np.array(x)
    return p0 + p1*x + p2*(p1*x)**2


#pinit=np.repeat([1.0], fit_order + 1)
pinit = [10, 10, -7e-7]
pinit2 = [1, 1, -7e-10]


# MORE GENERAL FUNCTION
# V=(1/2Vbi) (ft)**2 (-1 + sqrt(1 + (2ft)**2 Vbi )  )

# def fitfunc (t, f, Vbi):
#    t=np.array(t)
#    return (1/2*Vbi)*(f*t)**2*(-1+np.sqrt(1+4*(f*t)**2*Vbi))
#pinit=[1.0, 1.0]


def fit_pixel_ramp(ramp='', time='', counter=0, pinit=[1, 1, 1e-6]):
    time_vec, signal_vec, signal_vec_err = [], [], []
    if not len(time) == len(ramp):
        print("len(time), len(ramp): ", len(time), len(ramp))
        print("len(time)==len(ramp) not satisfied")
        sys.exit(1)
    for t, sample_array in zip(time, ramp):

        index_x, index_y = np.unravel_index(counter, sample_array.shape)
        s = sample_array[index_x, index_y]

        time_vec.append(t)
        #signal_vec.append( 2**16-1-s )
        signal_vec.append(s)  # s=ADU_dark - ADU_data
        signal_vec_err.append(np.sqrt(1))

    time_vec = np.array(time_vec)
    signal_vec = np.array(signal_vec)
    signal_vec_err = np.array(signal_vec_err)

    pfinal, covar, chi2_red = get_final_parameters_first_fit(
        x=time_vec, y=signal_vec, y_err=signal_vec_err, pinit=pinit)
    p0, p1, p2 = pfinal[0], pfinal[1], pfinal[2]
    p0_err = np.sqrt(np.diag(covar))[0]
    p1_err = np.sqrt(np.diag(covar))[1]
    p2_err = np.sqrt(np.diag(covar))[2]
    # print "pfinal: ", pfinal
    # print "pfinal_err: ", np.diag(covar)
    string_label = "c2: %g +/- %g" % (p2, p2_err)
    # return p2, p2_err, chi2_red
    return pfinal, covar, chi2_red


def correct_and_fit_pixel_ramp(ramp='', dark_ramp='', time='', counter=0, pinit=[1, 1, -1e-6], c0=0, c0_dark=0, c2=1):
    flag = False
    time_vec, signal_vec, signal_vec_err = [], [], []
    if not len(time) == len(ramp):
        print("len(time)==len(ramp) not satisfied")
        sys.exit(1)

    print("SHAPES: ", ramp.shape, dark_ramp.shape, time.shape)

    for t, sample_array, sample_dark in zip(time, ramp, dark_ramp):

        index_x, index_y = np.unravel_index(counter, sample_array.shape)
        s_temp = sample_array[index_x, index_y]
        # Apply the correction here!
        s = (1/(2*c2))*(-1+np.sqrt(1-4*c2*(c0-s_temp)))
        print("Corrected S: ", s)
        # Same pixels, in dark fields
        d_temp = sample_dark[index_x, index_y]
        d = (1/(2*c2))*(-1+np.sqrt(1-4*c2*(c0_dark-d_temp)))

        # Subtract dark from data
        s -= d
        print("Dark Subtracted S: ", s)
        time_vec.append(t)
        #signal_vec.append( 2**16-1-s )
        signal_vec.append(s)  # s=ADU_dark - ADU_data
        signal_vec_err.append(np.sqrt(1))

    time_vec = np.array(time_vec)
    signal_vec = np.array(signal_vec)
    signal_vec_err = np.array(signal_vec_err)

    print("After loop of corrections ")
    if np.isnan(signal_vec).any():
        flag = True
        return [0, 0, 0], 0, 0, flag
    else:
        print("Before get_final_parameters: ")
        pfinal, covar, chi2_red = get_final_parameters_first_fit(
            x=time_vec, y=signal_vec, y_err=signal_vec_err, pinit=pinit)
        p0, p1, p2 = pfinal[0], pfinal[1], pfinal[2]
        p0_err = np.sqrt(np.diag(covar))[0]
        p1_err = np.sqrt(np.diag(covar))[1]
        p2_err = np.sqrt(np.diag(covar))[2]
        # print "pfinal: ", pfinal
        # print "pfinal_err: ", np.diag(covar)
        #string_label="c2: %g +/- %g" %(p2, p2_err)
        # return p2, p2_err, chi2_red
        return pfinal, covar, chi2_red, flag


def get_final_parameters_first_fit(x=[], y=[], y_err=[], pinit=pinit):
    # this function gets all the parameters according to the polynomial order: alpha, beta, gamma, delta, etc
    x, y, y_err = np.array(x), np.array(y), np.array(y_err)
    pfinal, covar = optimize.curve_fit(
        fitfunc, x, y, p0=pinit, sigma=y_err,  maxfev=100000000)
    chi2 = np.power((fitfunc(x, *pfinal) - y)/y_err,  2).sum()
    n, d = len(x), len(pinit)
    nu = n-d
    chi2_red = chi2*1./nu
    return pfinal, covar, chi2_red


# Eric's data Feb 22
# files=glob.glob("/data-acq/WFIRST/2017-02-22/raw/flats-*")
# files_darks=glob.glob("/data-acq/WFIRST/2017-02-22/raw/darks-*")


# Eric's data, Feb 23
# files=glob.glob("/data-acq/WFIRST/2017-02-23/raw/focus-000[3-9]_*.fits") # spots
# files=glob.glob("/data-acq/WFIRST/2017-02-23/raw/flats-*") # flats
# files_darks=glob.glob("/data-acq/WFIRST/2017-02-23/raw/focus-000[012]_*.fits")

# Chaz's data, Feb 28. Don't use badger to read the ramps
# files_darks=glob.glob("/data-acq/WFIRST/2017-02-28/raw/spots_fixed-100[0-9]_0000.fits")
# files=glob.glob("/data-acq/WFIRST/2017-02-28/raw/spots_fixed-101[0-9]_0000.fits") #Flats 80


# Andres's data, 03-02-17
files_darks = glob.glob(
    "/projector/aplazas/data/WFIRST/2017-03-02/raw/andres-000[0-9]*.fits")  # 10 ramps of darks

# files1=glob.glob("/projector/aplazas/data/WFIRST/2017-03-02/raw/andres-00[1-9][0-9]*.fits")  # 100 ramps of flats, wlamp=120
# files2=glob.glob("/projector/aplazas/data/WFIRST/2017-03-02/raw/andres-010[0-9]*.fits")
# files_data1=files1+files2

# ramp_string=sys.argv[1].zfill(4)
# files=glob.glob("/projector/aplazas/data/WFIRST/2017-03-02/raw/andres-%s*.fits"%ramp_string)
# print "File: ", files

#temp=int(sys.argv[1]) + 100
# print temp
# ramp_string2="%s"%temp
# print ramp_string2
# ramp_string2=ramp_string2.zfill(4)
# print ramp_string2
# files2=glob.glob("/projector/aplazas/data/WFIRST/2017-03-02/raw/andres-%s*.fits"%ramp_string2)
# print "File number 2: ", files2

# files1=glob.glob("/projector/aplazas/data/WFIRST/2017-03-02/raw/andres-01[1-9][0-9]*.fits")  # 100 ramps of flats, wlamp=100
# files2=glob.glob("/projector/aplazas/data/WFIRST/2017-03-02/raw/andres-020[0-9]*.fits")
# files_data2=files1+files2

# files1=glob.glob("/data-acq/WFIRST/2017-03-02/raw/andres-02[1-9][0-9]*.fits")  # 100 ramps of spots, wlamp=180
# files2=glob.glob("/data-acq/WFIRST/2017-03-02/raw/andres-030[0-9]*.fits")
# files=files1+files2

# files1=glob.glob("/data-acq/WFIRST/2017-03-02/raw/andres-03[1-9][0-9]*.fits")  # 100 ramps of spots, wlamp=20
# files2=glob.glob("/data-acq/WFIRST/2017-03-02/raw/andres-040[0-9]*.fits")
# files=files1+files2


# Use Stacked (100 files) file
files_data1 = glob.glob(
    "/projector/aplazas/stacked/stacked100FlatsWLAMP120W-*.fits")
#files_data2=glob.glob ("/projector/aplazas/stacked/stacked100FlatsWLAMP100W-*.fits")
files_data_A = glob.glob(
    "/projector/aplazas/data/WFIRST/2017-05-10/raw/f11cal000[5-9].fits")  # Chaz
files_data_B = glob.glob(
    "/projector/aplazas/data/WFIRST/2017-05-10/raw/f11cal001[0-4].fits")  # Chaz
files_data2 = files_data_A + files_data_B
print(files_data2)

# To read stacked files; for Chaz's data use Badger
dict_data1 = {'0000': [], '0001': [], '0002': [],
              '0003': [], '0004': [], '0005': []}
#dict_data2={'0000':[], '0001':[], '0002':[], '0003':[], '0004':[], '0005':[]}


for file in files_data1:
    number = file.split('_')[-1].split('.')[0]
    dict_data1[number] = file

# for file in files_data2:
#    number=file.split('_')[-1].split('.')[0]
#    dict_data2[number]=file


print(dict_data1)
# print dict_data2

data_data1, data_data2 = [], []
darks2 = []

for k in ['0000', '0001', '0002', '0003', '0004', '0005']:
    s = dict_data1[k]
    d = pf.open(s)[0].data
    data_data1.append(d)

# for k in ['0005', '0006', '0007', '0008', '0009', '00010','00011', '00012', '00013', '00014']:

#    f=dict_data2[k]
#    d=pf.open(f)[0].data
#    data_data2.append(d)


temp_data1 = np.array(data_data1)
# temp_data2=np.array(data_data2)


# Use Chaz's data for the ramp with lower illumination
files_darks2 = glob.glob(
    "/projector/aplazas/data/WFIRST/2017-05-10/raw/f11cal000[01234].fits")
files_data_A = glob.glob(
    "/projector/aplazas/data/WFIRST/2017-05-10/raw/f11cal000[5-9].fits")
files_data_B = glob.glob(
    "/projector/aplazas/data/WFIRST/2017-05-10/raw/f11cal001[0-4].fits")
files_data2 = files_data_A + files_data_B


allData2, infoData2 = badger.getRampsFromFiles(files_data2)
print(infoData2)
print(infoData2.dtype.names)

# print temp_data2.shape
# sys.exit()

allDarks, infoDarks = badger.getRampsFromFiles(files_darks)
# this is just to get the info. data 1, darks1: taken by me, several ramps.
allDarks2, infoDarks2 = badger.getRampsFromFiles(files_darks2)

print(infoDarks.dtype.names)


print("Number of dark ramps, infoDarks['EXPTIME'][0], infoDarks['sample'][0]: ", len(
    allDarks), infoDarks['EXPTIME'][0], infoDarks['sample'][0])
print("Number of dark ramps2, infoDarks2['EXPTIME'][0], infoDarks2['sample'][0]: ", len(
    allDarks2), infoDarks2['EXPTIME'][0], infoDarks2['sample'][0])
temp_darks2 = np.median(allDarks2, axis=0)
temp_darks = np.median(allDarks, axis=0)  # Use median of darks
# Stack Chaz's ramps, filmstrip; each single FITS file is a full ramp
temp_data2 = np.median(allData2, axis=0)


time2 = np.linspace(0.0, infoDarks2['EXPTIME'][0], infoDarks2['sample'][0])[1:]
#time=np.linspace(0.0, infoDarks['EXPTIME'][0], infoDarks['sample'][0] +1)[1:]
time = np.linspace(0.0, infoDarks['EXPTIME'][0], len(dict_data1))[1:-1]
print("Time: ", time, len(time))
print("Time2: ", time2, len(time2))
# sys.exit()

shapes = temp_data2.shape
data_data2 = np.zeros(shapes)
data_darks2 = np.zeros(shapes)
print(shapes)

# Reference pixels correction
for i in range(shapes[0]):
    for j in range(shapes[1]):
        data_data2[i, j, :] = (2**16-1-temp_data2[i, j, :]) - \
            (2**16-1-np.median(temp_data2[i, j, 2044:]))
        data_darks2[i, j, :] = (2**16-1-temp_darks2[i, j, :]) - \
            (2**16-1-np.median(temp_darks2[i, j, 2044:]))


shapes = temp_data1.shape
data_data1 = np.zeros(shapes)
data_darks = np.zeros(shapes)
print(shapes)


# Reference pixels correction
for i in range(shapes[0]):
    for j in range(shapes[1]):
        data_data1[i, j, :] = (2**16-1-temp_data1[i, j, :]) - \
            (2**16-1-np.median(temp_data1[i, j, 2044:]))
        data_darks[i, j, :] = (2**16-1-temp_darks[i, j, :]) - \
            (2**16-1-np.median(temp_darks[i, j, 2044:]))


data_data1 = data_data1*gain
data_data2 = data_data2*gain
data_darks = data_darks*gain
data_darks2 = data_darks2*gain

# Reference frame. Discard frame 0.
B_data1 = data_data1[1]
B_data2 = data_data2[1]
B_darks = data_darks[1]
B_darks2 = data_darks2[1]

print(np.mean(B_data1), np.mean(B_data2), np.mean(B_darks), np.mean(B_darks2))
# sys.exit(1)

data_data1 = data_data1[1:]
data_data2 = data_data2[1:]
data_darks = data_darks[1:]
data_darks2 = data_darks2[1:]


M = np.array([[0, 0.007, 0], [0.009, -0.032, 0.009], [0, 0.007, 0]])
I = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
K = I-M
print("K: ")
print(K)


# Loop over each sample, correct for IPC, bias, and Reference pixels
GLOBAL_RAMP, GLOBAL_RAMP2 = [], []
darks, darks2 = [], []
for sample_f1, sample_d1 in zip(data_data1, data_darks):
    con_f1 = ndimage.filters.convolve(
        sample_f1 - B_data1, K, mode='constant', cval=0.0)
    con_d1 = ndimage.filters.convolve(
        sample_d1 - B_darks, K, mode='constant', cval=0.0)
    GLOBAL_RAMP.append(con_f1 + B_data1)
    darks.append(con_d1 + B_darks)

for sample_f2, sample_d2 in zip(data_data2, data_darks2):
    con_f2 = ndimage.filters.convolve(
        sample_f2 - B_data2, K, mode='constant', cval=0.0)
    con_d2 = ndimage.filters.convolve(
        sample_d2 - B_darks2, K, mode='constant', cval=0.0)
    GLOBAL_RAMP2.append(con_f2 + B_data2)
    darks2.append(con_d2 + B_darks2)


GLOBAL_RAMP = np.array(GLOBAL_RAMP)[:-1]
GLOBAL_RAMP2 = np.array(GLOBAL_RAMP2)
darks = np.array(darks)[:-1]
darks2 = np.array(darks2)


print("ALL SHAPES: ")
print("time.shape: ", time.shape)
print("time2.shape: ", time2.shape)
print("GLOBAL_RAMP.shape: ", GLOBAL_RAMP.shape)
print("GLOBAL_RAMP2.shape: ", GLOBAL_RAMP2.shape)
print("darks.shape: ", darks.shape)
print("darks2.shape: ", darks2.shape)

# sys.exit()

title_string = ''
title_string2 = ''  # infoRamps2['fileName'][0]


ramp_number = 100
dir = sys.argv[1]
outdir = '/projector/aplazas/'+dir+"/"
cmd = "mkdir -v %s" % (outdir)
S.Popen([cmd], shell=True, stdout=S.PIPE).communicate()[0].split()
#S.Popen([cmd], shell=True, stdout=open("hola", "w")).communicate()[0].split()

print("OUTPUT DIRECTORY: ", outdir)


#darks2=np.median(allDarks, axis=0)
# darks=np.median(allDarks, axis=0)[:-1]  # Use median of darks


# data1, darks, and, time are the higher illumination level


#allRamps=darks - allRamps[0:30]
#data=np.mean(allRamps, axis=0)
# allRamps=0
# GLOBAL_RAMP=(darks-data1)*gain
# GLOBAL_RAMP2=(darks2-data2)*gain

# GLOBAL_RAMP=data*gain
MASK = pf.open("/projector/aplazas/master-euclid-mask.fits")[0].data
fill_value = -999999


x_cut = 1550
y_cut = 1850
# Use only pixels for which x > 1000
GLOBAL_RAMP = GLOBAL_RAMP[:, y_cut:, x_cut:]
GLOBAL_RAMP2 = GLOBAL_RAMP2[:, y_cut:, x_cut:]
MASK = MASK[y_cut:, x_cut:]

npix_total = MASK.shape[0]*MASK.shape[1]


def aux_quadratic_fit(t_vec, s_vec, s_e_vec, pinit=pinit, label="_"):
    p, c, chi2 = get_final_parameters_first_fit(
        x=t_vec, y=s_vec, y_err=s_e_vec, pinit=pinit)
    pe = np.sqrt(np.diag(c))
    print("p, pe, chi2: ", p, pe, chi2)
    p_all = np.array([p, pe])
    #np.savetxt (outdir+"parameters_quadratic_%s.txt"%label, p_all)
    return p, pe, chi2, c


def get_residual_error(l='', s='', varc0='', varc1='', covc0c1='', t=''):
    # t=t/1000
    # return 100*np.sqrt((s/l**2)+ s**2*(varc0+t**2*varc1 + 2*t*covc0c1)/l**2)
    return 100*np.sqrt(s)/l


def quadratic_fit(GLOBAL_RAMP, GLOBAL_RAMP2, time, time2):
    t_vec, s_vec, se_vec = [], [], []
    s1, s2, s3, s4 = [], [], [], []
    se1, se2, se3, se4 = [], [], [], []

    t_vec2, s_vec2, se_vec2 = [], [], []
    s1_2, s2_2, s3_2, s4_2 = [], [], [], []
    se1_2, se2_2, se3_2, se4_2 = [], [], [], []

    # x1,y1=500,500
    # x2,y2=1500,1500
    #x3,y3=1500, 500
    # x4,y4=1000,1500

    # x1,y1=500,500
    # x2,y2=900,900
    #x3,y3=900, 500
    # x4,y4=700,700

    x1, y1 = 100, 100
    x2, y2 = 150, 150
    x3, y3 = 125, 125
    x4, y4 = 50, 50

    counter = 0
    for (sample, sample2, t, t2) in zip(GLOBAL_RAMP, GLOBAL_RAMP2, time, time2):
        if counter == 0:
            counter += 1
            continue
        # mean
        temp = sample*MASK
        index, = np.where(temp.flatten() == 0)
        masked_sample = sample.flatten()[index]
        mean, scatter, indices = sigma_clip.sigma_clip(
            masked_sample, niter=10, nsig=sigma_cut, get_indices=True)
        temp_data = sample.flatten()[indices]
        s_vec.append(mean)
        se_vec.append(scatter/np.sqrt(len(temp_data)))
        t_vec.append(t)
        # pixel 1
        s1.append(sample[x1][y1])
        se1.append(np.sqrt(sample[x1][y1]))
        # pixel 2
        s2.append(sample[x2][y2])
        se2.append(np.sqrt(sample[x2][y2]))
        # pixel 3
        s3.append(sample[x3][y3])
        se3.append(np.sqrt(sample[x3][y3]))
        # pixel 4
        s4.append(sample[x4][y4])
        se4.append(np.sqrt(sample[x4][y4]))

        # GLOBAL_RAMP2
        # mean
        temp = sample2*MASK
        index, = np.where(temp.flatten() == 0)
        masked_sample2 = sample2.flatten()[index]
        mean, scatter, indices = sigma_clip.sigma_clip(
            masked_sample2, niter=10, nsig=sigma_cut, get_indices=True)
        temp_data = sample2.flatten()[indices]
        s_vec2.append(mean)
        se_vec2.append(scatter/np.sqrt(len(temp_data)))
        t_vec2.append(t2)
        # pixel 1
        s1_2.append(sample2[x1][y1])
        se1_2.append(np.sqrt(sample2[x1][y1]))
        # pixel 2
        s2_2.append(sample2[x2][y2])
        se2_2.append(np.sqrt(sample2[x2][y2]))
        # pixel 3
        s3_2.append(sample2[x3][y3])
        se3_2.append(np.sqrt(sample2[x3][y3]))
        # pixel 4
        s4_2.append(sample2[x4][y4])
        se4_2.append(np.sqrt(sample2[x4][y4]))

    p_mean, pe_mean, chi2_mean, cov_mean = aux_quadratic_fit(
        t_vec, s_vec, se_vec, pinit=pinit, label="mean")
    p_pix1, pe_pix1, chi2_pix1, cov_pix1 = aux_quadratic_fit(
        t_vec, s1, se1, pinit=pinit, label="pix1")
    p_pix2, pe_pix2, chi2_pix2, cov_pix2 = aux_quadratic_fit(
        t_vec, s2, se2, pinit=pinit, label="pix2")
    p_pix3, pe_pix3, chi2_pix3, cov_pix3 = aux_quadratic_fit(
        t_vec, s3, se3, pinit=pinit, label="pix3")
    p_pix4, pe_pix4, chi2_pix4, cov_pix4 = aux_quadratic_fit(
        t_vec, s4, se4, pinit=pinit, label="pix4")

    p_mean_2, pe_mean_2, chi2_mean_2, cov_mean_2 = aux_quadratic_fit(
        t_vec2, s_vec2, se_vec2, pinit=pinit, label="mean")
    p_pix1_2, pe_pix1_2, chi2_pix1_2, cov_pix1_2 = aux_quadratic_fit(
        t_vec2, s1_2, se1_2, pinit=pinit, label="pix1")
    p_pix2_2, pe_pix2_2, chi2_pix2_2, cov_pix2_2 = aux_quadratic_fit(
        t_vec2, s2_2, se2_2, pinit=pinit, label="pix2")
    p_pix3_2, pe_pix3_2, chi2_pix3_2, cov_pix3_2 = aux_quadratic_fit(
        t_vec2, s3_2, se3_2, pinit=pinit, label="pix3")
    p_pix4_2, pe_pix4_2, chi2_pix4_2, cov_pix4_2 = aux_quadratic_fit(
        t_vec2, s4_2, se4_2, pinit=pinit, label="pix4")

    # APPLY CORRECTION FROM STACKED FLATS AT W120. HARDCODED!!!!!!
    #p_mean, pe_mean=[76.0894, 6.14945, -7.20641e-07], [53.9008, 0.805332, 6.67043e-08]
    #p_pix1, pe_pix1=p_mean, pe_mean
    #p_pix2, pe_pix2=p_mean, pe_mean
    #p_pix3, pe_pix3=p_mean, pe_mean
    #p_pix4, pe_pix4=p_mean, pe_mean

    t_vec, t_vec2 = np.array(t_vec), np.array(t_vec2)
    s_vec, se_vec = np.array(s_vec), np.array(se_vec)

    s1, se1 = np.array(s1), np.array(se1)
    s2, se2 = np.array(s2), np.array(se2)
    s3, se3 = np.array(s3), np.array(se3)
    s4, se4 = np.array(s4), np.array(se4)

    linear = p_mean[0] + p_mean[1]*t_vec
    quad = linear + p_mean[2]*(p_mean[1]*t_vec)**2
    res = 100*(-s_vec+linear)/linear
    res_err = get_residual_error(
        l=linear, s=s_vec, varc0=pe_mean[0]**2, varc1=pe_mean[1]**2, covc0c1=cov_mean[0][1], t=t_vec)

    linear_pix1 = p_pix1[0] + p_pix1[1]*t_vec
    quad_pix1 = linear_pix1 + p_pix1[2]*(p_pix1[1]*t_vec)**2
    res_pix1 = 100*(-s1+linear_pix1)/linear_pix1
    res_pix1_err = get_residual_error(
        l=linear_pix1, s=s1, varc0=pe_pix1[0]**2, varc1=pe_pix1[1]**2, covc0c1=cov_pix1[0][1], t=t_vec)

    linear_pix2 = p_pix2[0] + p_pix2[1]*t_vec
    quad_pix2 = linear_pix2 + p_pix2[2]*(p_pix2[1]*t_vec)**2
    res_pix2 = 100*(-s2+linear_pix2)/linear_pix2
    res_pix2_err = get_residual_error(
        l=linear_pix2, s=s2, varc0=pe_pix2[0]**2, varc1=pe_pix2[1]**2, covc0c1=cov_pix2[0][1], t=t_vec)

    linear_pix3 = p_pix3[0] + p_pix3[1]*t_vec
    quad_pix3 = linear_pix3 + p_pix3[2]*(p_pix3[1]*t_vec)**2
    res_pix3 = 100*(-s3+linear_pix3)/linear_pix3
    res_pix3_err = get_residual_error(
        l=linear_pix3, s=s3, varc0=pe_pix3[0]**2, varc1=pe_pix3[1]**2, covc0c1=cov_pix3[0][1], t=t_vec)

    linear_pix4 = p_pix4[0] + p_pix4[1]*t_vec
    quad_pix4 = linear_pix4 + p_pix4[2]*(p_pix4[1]*t_vec)**2
    res_pix4 = 100*(-s4+linear_pix4)/linear_pix4
    res_pix4_err = get_residual_error(
        l=linear_pix4, s=s4, varc0=pe_pix4[0]**2, varc1=pe_pix4[1]**2, covc0c1=cov_pix4[0][1], t=t_vec)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.errorbar(t_vec, res, yerr=res_err, fmt='k-o', alpha=1.0, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_mean[0], pe_mean[0], p_mean[1], pe_mean[1], p_mean[2], pe_mean[2]))
    ax.errorbar(t_vec, res_pix1, yerr=res_pix1_err, fmt='r-o', alpha=0.6, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_pix1[0], pe_pix1[0], p_pix1[1], pe_pix1[1], p_pix1[2], pe_pix1[2]))
    ax.errorbar(t_vec, res_pix2, yerr=res_pix2_err, fmt='b-o', alpha=0.6, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_pix2[0], pe_pix2[0], p_pix2[1], pe_pix2[1], p_pix2[2], pe_pix2[2]))
    ax.errorbar(t_vec, res_pix3, yerr=res_pix3_err, fmt='g-o', alpha=0.6, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_pix3[0], pe_pix3[0], p_pix3[1], pe_pix3[1], p_pix3[2], pe_pix3[2]))
    ax.errorbar(t_vec, res_pix4, yerr=res_pix4_err, fmt='y-o', alpha=0.6, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_pix4[0], pe_pix4[0], p_pix4[1], pe_pix4[1], p_pix4[2], pe_pix4[2]))
    plt.ylim([-3, 12])
    plt.legend(loc='upper right', fancybox=True,
               ncol=1, numpoints=1, prop=prop)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Fractional NL (%)')

    ax = fig.add_subplot(212)
    ax.errorbar(t_vec, s_vec, yerr=se_vec, fmt='k.', alpha=1.0, label="Mean")
    ax.errorbar(t_vec, quad, yerr=None, fmt='k-', alpha=1.0)

    ax.errorbar(t_vec, s1, yerr=se1, fmt='r.', alpha=0.6,
                label="(x1,y1)=(%g,%g)" % (x1, y1))
    ax.errorbar(t_vec, quad_pix1, yerr=None, fmt='r-', alpha=0.6)

    ax.errorbar(t_vec, s2, yerr=se2, fmt='b.', alpha=0.6,
                label="(x2,y2)=(%g,%g)" % (x2, y2))
    ax.errorbar(t_vec, quad_pix2, yerr=None, fmt='b-', alpha=0.6)

    ax.errorbar(t_vec, s3, yerr=se3, fmt='g.', alpha=0.6,
                label="(x3,y3)=(%g,%g)" % (x3, y3))
    ax.errorbar(t_vec, quad_pix3, yerr=None, fmt='g-', alpha=0.6)

    ax.errorbar(t_vec, s4, yerr=se4, fmt='y.', alpha=0.6,
                label="(x4,y4)=(%g,%g)" % (x4, y4))
    ax.errorbar(t_vec, quad_pix4, yerr=None, fmt='y-', alpha=0.6)

    plt.legend(loc='upper left', fancybox=True, ncol=1, numpoints=1, prop=prop)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal (e-)')

    fig.suptitle(title_string + " " + " Ramp Number: %g \n" %
                 ramp_number + title_string2)
    plt.tight_layout()
    pp.savefig(fig)

    # Now do the correction using the coefficients from above (GLOBAL_RAMP)

    t_vec, s_vec, se_vec = [], [], []
    s1, s2, s3, s4 = [], [], [], []
    se1, se2, se3, se4 = [], [], [], []

    counter = 0
    for sample, t in zip(GLOBAL_RAMP2, time2):
        if counter == 0:
            counter += 1
            continue
        # mean
        temp = sample*MASK
        index = np.where(temp.flatten() == 0)
        masked_sample = sample.flatten()[index]
        mean, scatter, indices = sigma_clip.sigma_clip(
            masked_sample, niter=10, nsig=sigma_cut, get_indices=True)
        temp_data = sample.flatten()[indices]
        corrected = (1/(2*p_mean[2])) * \
            (-1+np.sqrt(1-4*p_mean[2]*(p_mean_2[0]-mean)))
        print("mean, mean corrected, c0 + c_1*t: ",
              mean, corrected, p_mean[0] + p_mean[1]*t)
        s_vec.append(corrected)
        se_vec.append(scatter/np.sqrt(len(temp_data)))
        t_vec.append(t)

        corrected = (1/(2*p_pix1[2]))*(-1+np.sqrt(1 -
                                                  4*p_pix1[2]*(p_pix1_2[0]-sample[x1][y1])))
        s1.append(corrected)
        se1.append(np.sqrt(corrected))
        # pixel 2
        corrected = (1/(2*p_pix2[2]))*(-1+np.sqrt(1 -
                                                  4*p_pix2[2]*(p_pix2_2[0]-sample[x2][y2])))
        s2.append(corrected)
        se2.append(np.sqrt(corrected))
        # pixel 3
        corrected = (1/(2*p_pix3[2]))*(-1+np.sqrt(1 -
                                                  4*p_pix3[2]*(p_pix3_2[0]-sample[x3][y3])))
        s3.append(corrected)
        se3.append(np.sqrt(corrected))
        # pixel 4
        corrected = (1/(2*p_pix4[2]))*(-1+np.sqrt(1 -
                                                  4*p_pix4[2]*(p_pix4_2[0]-sample[x4][y4])))
        s4.append(corrected)
        se4.append(np.sqrt(corrected))

    p_mean_corrected, pe_mean_corrected, chi2_mean_corrected, cov_mean_corrected = aux_quadratic_fit(
        t_vec, s_vec, se_vec, pinit=pinit2, label="mean")
    p_pix1_c, pe_pix1_c, chi2_pix1_c, cov_pix1_c = aux_quadratic_fit(
        t_vec, s1, se1, pinit=pinit2, label="pix1")
    p_pix2_c, pe_pix2_c, chi2_pix2_c, cov_pix2_c = aux_quadratic_fit(
        t_vec, s2, se2, pinit=pinit2, label="pix2")
    p_pix3_c, pe_pix3_c, chi2_pix3_c, cov_pix3_c = aux_quadratic_fit(
        t_vec, s3, se3, pinit=pinit2, label="pix3")
    p_pix4_c, pe_pix4_c, chi2_pix4_c, cov_pix4_c = aux_quadratic_fit(
        t_vec, s4, se4, pinit=pinit2, label="pix4")

    t_vec = np.array(t_vec)
    s_vec, se_vec = np.array(s_vec), np.array(se_vec)

    s1, se1 = np.array(s1), np.array(se1)
    s2, se2 = np.array(s2), np.array(se2)
    s3, se3 = np.array(s3), np.array(se3)
    s4, se4 = np.array(s4), np.array(se4)

    linear_pix1_c = p_pix1_c[0] + p_pix1_c[1]*t_vec
    quad_pix1_c = linear_pix1_c + p_pix1_c[2]*(p_pix1_c[1]*t_vec)**2
    res_pix1_c = 100*(-s1+linear_pix1_c)/linear_pix1_c
    res_pix1_err_c = get_residual_error(
        l=linear_pix1_c, s=s1, varc0=pe_pix1_c[0]**2, varc1=pe_pix1_c[1]**2, covc0c1=cov_pix1_c[0][1], t=t_vec)

    linear_pix2_c = p_pix2_c[0] + p_pix2_c[1]*t_vec
    quad_pix2_c = linear_pix2_c + p_pix2_c[2]*(p_pix2_c[1]*t_vec)**2
    res_pix2_c = 100*(-s2+linear_pix2_c)/linear_pix2_c
    res_pix2_err_c = get_residual_error(
        l=linear_pix2_c, s=s2, varc0=pe_pix2_c[0]**2, varc1=pe_pix2_c[1]**2, covc0c1=cov_pix2_c[0][1], t=t_vec)

    linear_pix3_c = p_pix3_c[0] + p_pix3_c[1]*t_vec
    quad_pix3_c = linear_pix3_c + p_pix3_c[2]*(p_pix3_c[1]*t_vec)**2
    res_pix3_c = 100*(-s3+linear_pix3_c)/linear_pix3_c
    res_pix3_err_c = get_residual_error(
        l=linear_pix3_c, s=s3, varc0=pe_pix3_c[0]**2, varc1=pe_pix3_c[1]**2, covc0c1=cov_pix3_c[0][1], t=t_vec)

    linear_pix4_c = p_pix4_c[0] + p_pix4_c[1]*t_vec
    quad_pix4_c = linear_pix4_c + p_pix4_c[2]*(p_pix4_c[1]*t_vec)**2
    res_pix4_c = 100*(-s4+linear_pix4_c)/linear_pix4_c
    res_pix4_err_c = get_residual_error(
        l=linear_pix4_c, s=s4, varc0=pe_pix4_c[0]**2, varc1=pe_pix4_c[1]**2, covc0c1=cov_pix4_c[0][1], t=t_vec)

    linear_corrected = p_mean_corrected[0] + p_mean_corrected[1]*t_vec
    quad_corrected = linear_corrected + \
        p_mean_corrected[2]*(p_mean_corrected[1]*t_vec)**2
    res_corrected = 100*(-s_vec+linear_corrected)/linear_corrected
    res_corrected_err = get_residual_error(
        l=linear_corrected, s=s_vec, varc0=pe_mean_corrected[0]**2, varc1=pe_mean_corrected[1]**2, covc0c1=cov_mean_corrected[0][1], t=t_vec)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.errorbar(t_vec, res_corrected, yerr=res_corrected_err, fmt='k-o', alpha=1.0, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_mean_corrected[0], pe_mean_corrected[0], p_mean_corrected[1], pe_mean_corrected[1], p_mean_corrected[2], pe_mean_corrected[2]))

    ax.errorbar(t_vec, res_pix1_c, yerr=res_pix1_err_c, fmt='r-o', alpha=0.6, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_pix1_c[0], pe_pix1_c[0], p_pix1_c[1], pe_pix1_c[1], p_pix1_c[2], pe_pix1_c[2]))
    ax.errorbar(t_vec, res_pix2_c, yerr=res_pix2_err_c, fmt='b-o', alpha=0.6, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_pix2_c[0], pe_pix2_c[0], p_pix2_c[1], pe_pix2_c[1], p_pix2_c[2], pe_pix2_c[2]))
    ax.errorbar(t_vec, res_pix3_c, yerr=res_pix3_err_c, fmt='g-o', alpha=0.6, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_pix3_c[0], pe_pix3_c[0], p_pix3_c[1], pe_pix3_c[1], p_pix3_c[2], pe_pix3_c[2]))
    ax.errorbar(t_vec, res_pix4_c, yerr=res_pix4_err_c, fmt='y-o', alpha=0.6, label=" $C_0$:(%.3g$\pm$%.3g),$C_1$:(%.3g$\pm$%.3g),$C_2$:(%.4g$\pm$%.3g)" %
                (p_pix4_c[0], pe_pix4_c[0], p_pix4_c[1], pe_pix4_c[1], p_pix4_c[2], pe_pix4_c[2]))
    plt.ylim([-2.5, 2.5])

    plt.legend(loc='upper right', fancybox=True,
               ncol=1, numpoints=1, prop=prop)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Fractional NL (%)')

    ax = fig.add_subplot(212)
    ax.errorbar(t_vec, s_vec, yerr=se_vec, fmt='k.', alpha=1.0, label="Mean")
    ax.errorbar(t_vec, quad_corrected, yerr=None, fmt='k-', alpha=1.0)

    ax.errorbar(t_vec, s1, yerr=se1, fmt='r.', alpha=0.6,
                label="(x1,y1)=(%g,%g)" % (x1, y1))
    ax.errorbar(t_vec, quad_pix1_c, yerr=None, fmt='r-', alpha=0.6)

    ax.errorbar(t_vec, s2, yerr=se2, fmt='b.', alpha=0.6,
                label="(x2,y2)=(%g,%g)" % (x2, y2))
    ax.errorbar(t_vec, quad_pix2_c, yerr=None, fmt='b-', alpha=0.6)

    ax.errorbar(t_vec, s3, yerr=se3, fmt='g.', alpha=0.6,
                label="(x3,y3)=(%g,%g)" % (x3, y3))
    ax.errorbar(t_vec, quad_pix3_c, yerr=None, fmt='g-', alpha=0.6)

    ax.errorbar(t_vec, s4, yerr=se4, fmt='y.', alpha=0.6,
                label="(x4,y4)=(%g,%g)" % (x4, y4))
    ax.errorbar(t_vec, quad_pix4_c, yerr=None, fmt='y-', alpha=0.6)

    plt.legend(loc='upper left', fancybox=True, ncol=1, numpoints=1, prop=prop)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal (e-)')

    fig.suptitle("Corrected")
    plt.tight_layout()
    pp.savefig(fig)


pp = PdfPages(outdir+"nl_ramp_%g_diff.pdf" % ramp_number)
# Next function: for mean and 4 pixels ramps. For the 4 million pixels; parallelization below.
# GLOBAL_RAMP: derive NL, GLOBAL_RAMP2: where NL correction from ramp 1 is applied
quadratic_fit(GLOBAL_RAMP, GLOBAL_RAMP2, time, time2)
# pp.close()
# sys.exit(1)


def nl_function(index):
    # Unique index to some dictionary that gives you the input values
    index_x, index_y = np.unravel_index(index, GLOBAL_RAMP[0].shape)
    if MASK[index_x, index_y] == 16:
        return fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, index_x, index_y

    else:
        # Get coefficients GLOBAL_RAMP. I will use c2 from here in 'correction'. Ramp used to derive main correction
        p, cov, chi2 = fit_pixel_ramp(
            ramp=GLOBAL_RAMP, time=time, counter=index, pinit=pinit)
        c0, c1, c2 = p[0], p[1], p[2]
        c0_err, c1_err, c2_err = np.sqrt(np.diag(cov))[0], np.sqrt(np.diag(cov))[
            1], np.sqrt(np.diag(cov))[2]
        # Get coefficients GLOBAL_RAMP2. I will use c0 from here in 'correction'
        p2, cov2, chi2 = fit_pixel_ramp(
            ramp=GLOBAL_RAMP2, time=time2, counter=index, pinit=pinit)
        c0_2, c1_2, c2_2 = p2[0], p2[1], p2[2]
        c0_err2, c1_err2, c2_err2 = np.sqrt(np.diag(cov2))[0], np.sqrt(np.diag(cov2))[
            1], np.sqrt(np.diag(cov2))[2]

        # Get coefficients dark_2. I will use c0_dark2 from here in 'correction'
        p2_darks2, cov2_darks2, chi2_darks2 = fit_pixel_ramp(
            ramp=darks2, time=time2, counter=index, pinit=pinit)
        c0_darks2, c1_darks2, c2_darks2 = p2_darks2[0], p2_darks2[1], p2_darks2[2]
        c0_darks2_err2, c1_darks2_err2, c2_darks2_err2 = np.sqrt(np.diag(cov2_darks2))[
            0], np.sqrt(np.diag(cov2_darks2))[1], np.sqrt(np.diag(cov2_darks2))

        if np.isnan(c0) or np.isnan(c2) or np.isinf(c0) or np.isinf(c2):
            return fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, index_x, index_y
        else:
            # Apply the correction per ramp and Fit again
            p, cov, chi2_corr, flag = correct_and_fit_pixel_ramp(
                ramp=GLOBAL_RAMP2, dark_ramp=darks2, time=time2, counter=index, pinit=pinit2, c0=c0_2, c0_dark=c0_darks2, c2=c2)
            if flag:
                return fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, index_x, index_y
            else:
                c0_corr, c1_corr, c2_corr = p[0], p[1], p[2]
                c0_err_corr, c1_err_corr, c2_err_corr = np.sqrt(
                    np.diag(cov))[0], np.sqrt(np.diag(cov))[1], np.sqrt(np.diag(cov))[2]

                if np.isnan(c0_corr) or np.isnan(c2_corr) or np.isinf(c0_corr) or np.isinf(c2_corr):
                    return fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, fill_value, index_x, index_y
                else:
                    return c0, c0_err, c1, c1_err, c2, c2_err, chi2, c0_corr, c0_err_corr, c1_corr, c1_err_corr, c2_corr, c2_err_corr, chi2_corr, index_x, index_y


processes = cpu_count()
use = 8
print("I have ", processes, "cores here. Using: %g" % use)
pool = Pool(processes=use)
results = pool.map(nl_function, list(range(npix_total)))
pool.close()
pool.join()


c0_vec, c0_err_vec, chi2_vec = [], [], []
c1_vec, c1_err_vec = [], []
c2_vec, c2_err_vec = [], []

c0_mat = 0*GLOBAL_RAMP[0]
c1_mat = 0*GLOBAL_RAMP[0]
c2_mat = 0*GLOBAL_RAMP[0]
chi2_mat = 0*GLOBAL_RAMP[0]


c0_corr_vec, c0_err_corr_vec, chi2_corr_vec = [], [], []
c1_corr_vec, c1_err_corr_vec = [], []
c2_corr_vec, c2_err_corr_vec = [], []

c0_corr_mat = 0*GLOBAL_RAMP[0]
c1_corr_mat = 0*GLOBAL_RAMP[0]
c2_corr_mat = 0*GLOBAL_RAMP[0]
chi2_corr_mat = 0*GLOBAL_RAMP[0]


print("ramp[0].shape: ", GLOBAL_RAMP[0].shape)

for line in results:
    c0_vec.append(line[0])
    c0_err_vec.append(line[1])
    c1_vec.append(line[2])
    c1_err_vec.append(line[3])
    c2_vec.append(line[4])
    c2_err_vec.append(line[5])
    chi2_vec.append(line[6])

    c0_corr_vec.append(line[7])
    c0_err_corr_vec.append(line[8])
    c1_corr_vec.append(line[9])
    c1_err_corr_vec.append(line[10])
    c2_corr_vec.append(line[11])
    c2_err_corr_vec.append(line[12])
    chi2_corr_vec.append(line[13])

    i, j = line[14], line[15]

    c0 = line[0]
    c1 = line[2]
    c2 = line[4]
    chi2 = line[6]

    c0_corr = line[7]
    c1_corr = line[9]
    c2_corr = line[11]
    chi2_corr = line[13]

    c0_mat[i, j] = c0
    c1_mat[i, j] = c1
    c2_mat[i, j] = c2
    chi2_mat[i, j] = chi2

    c0_corr_mat[i, j] = c0_corr
    c1_corr_mat[i, j] = c1_corr
    c2_corr_mat[i, j] = c2_corr
    chi2_corr_mat[i, j] = chi2_corr


print("Before sigma clipping ")

np.savetxt(outdir+'c0_ramp_%g.txt' % ramp_number, c0_mat)
np.savetxt(outdir+'c1_ramp_%g.txt' % ramp_number, c1_mat)
np.savetxt(outdir+'c2_ramp_%g.txt' % ramp_number, c2_mat)
np.savetxt(outdir+'chi2_ramp_%g.txt' % ramp_number, chi2_mat)

np.savetxt(outdir+'c0_ramp_%g_corr.txt' % ramp_number, c0_corr_mat)
np.savetxt(outdir+'c1_ramp_%g_corr.txt' % ramp_number, c1_corr_mat)
np.savetxt(outdir+'c2_ramp_%g_corr.txt' % ramp_number, c2_corr_mat)
np.savetxt(outdir+'chi2_ramp_%g_corr.txt' % ramp_number, chi2_corr_mat)

c0_vec = np.array(c0_vec)
c0_err_vec = np.array(c0_err_vec)
c1_vec = np.array(c1_vec)
c1_err_vec = np.array(c1_err_vec)
c2_vec = np.array(c2_vec)
c2_err_vec = np.array(c2_err_vec)
chi2_vec = np.array(chi2_vec)

c0_corr_vec = np.array(c0_corr_vec)
c0_err_corr_vec = np.array(c0_err_corr_vec)
c1_corr_vec = np.array(c1_corr_vec)
c1_err_corr_vec = np.array(c1_err_corr_vec)
c2_corr_vec = np.array(c2_corr_vec)
c2_err_corr_vec = np.array(c2_err_corr_vec)
chi2_corr_vec = np.array(chi2_corr_vec)


mean_c0, scatter_c0, indices = sigma_clip.sigma_clip(
    c0_vec, niter=10, nsig=sigma_cut, get_indices=True)
c0_vec = c0_vec[indices]

mean_c1, scatter_c1, indices = sigma_clip.sigma_clip(
    c1_vec, niter=10, nsig=sigma_cut, get_indices=True)
c1_vec = c1_vec[indices]

mean_c2, scatter_c2, indices = sigma_clip.sigma_clip(
    c2_vec, niter=10, nsig=sigma_cut, get_indices=True)
c2_vec = c2_vec[indices]

mean_chi2, scatter_chi2, indices = sigma_clip.sigma_clip(
    chi2_vec, niter=10, nsig=sigma_cut, get_indices=True)
chi2_vec = chi2_vec[indices]


mean_c0_corr, scatter_c0_corr, indices = sigma_clip.sigma_clip(
    c0_corr_vec, niter=10, nsig=sigma_cut, get_indices=True)
c0_corr_vec = c0_corr_vec[indices]

mean_c1_corr, scatter_c1_corr, indices = sigma_clip.sigma_clip(
    c1_corr_vec, niter=10, nsig=sigma_cut, get_indices=True)
c1_corr_vec = c1_corr_vec[indices]

mean_c2_corr, scatter_c2_corr, indices = sigma_clip.sigma_clip(
    c2_corr_vec, niter=10, nsig=sigma_cut, get_indices=True)
c2_corr_vec = c2_corr_vec[indices]

mean_chi2_corr, scatter_chi2_corr, indices = sigma_clip.sigma_clip(
    chi2_corr_vec, niter=10, nsig=sigma_cut, get_indices=True)
chi2_corr_vec = chi2_corr_vec[indices]


#pp=PdfPages(outdir+"nl_ramp_%g.pdf" %ramp_number)

fig = plt.figure()
ax = fig.add_subplot(221)
n, bins, patches_out = ax.hist(c0_vec, 50, normed=True, facecolor='green',
                               alpha=0.75, label='Mean: %g \n Scatter:%g' % (mean_c0, scatter_c0))
ax.set_title('Histogram of c0 after %g-sigma clipping' % sigma_cut, size=8)
ax.legend(loc=loc_label, fancybox=True, ncol=1, numpoints=1, prop=prop)
ax.tick_params(axis='both', which='major', labelsize=7)


ax = fig.add_subplot(222)
n, bins, patches_out = ax.hist(c1_vec, 50, normed=True, facecolor='yellow',
                               alpha=0.75, label='Mean: %g \n Scatter:%g' % (mean_c1, scatter_c1))
ax.set_title('Histogram of c1 after %g-sigma clipping' % sigma_cut, size=8)
ax.legend(loc=loc_label, fancybox=True, ncol=1, numpoints=1, prop=prop)
ax.tick_params(axis='both', which='major', labelsize=7)

ax = fig.add_subplot(223)
n, bins, patches_out = ax.hist(c2_vec, 50, normed=True, facecolor='red',
                               alpha=0.75, label='Mean: %g \n Scatter:%g' % (mean_c2, scatter_c2))
ax.set_title('Histogram of c2 after %g-sigma clipping' % sigma_cut, size=8)
ax.legend(loc=loc_label, fancybox=True, ncol=1, numpoints=1, prop=prop)
ax.tick_params(axis='both', which='major', labelsize=7)

# with open("all_c2.txt", "a") as myfile:
#    myfile.write("%g %g \n"%(mean_c2, scatter_c2))


ax = fig.add_subplot(224)
n, bins, patches_out = ax.hist(chi2_vec, 50, normed=True, facecolor='blue',
                               alpha=0.75, label='Mean: %g \n Scatter:%g' % (mean_chi2, scatter_chi2))
ax.set_title('Histogram of chi2 after %g-sigma clipping' % sigma_cut, size=8)
ax.legend(loc=loc_label, fancybox=True, ncol=1, numpoints=1, prop=prop)
ax.tick_params(axis='both', which='major', labelsize=7)


cmap = matplotlib.cm.seismic
cmap.set_under(color='black')


# fig.suptitle(title_string)
plt.tight_layout()
pp.savefig(fig)


fig = plt.figure()
ax = fig.add_subplot(221)
plt.imshow(c0_mat, cmap=cmap, interpolation='nearest', origin='low',
           vmin=mean_c0 - sigma_cut*mean_c0, vmax=mean_c0 + sigma_cut*mean_c0)
plt.colorbar()
ax.set_title(r"$c_0$", size=13)

ax = fig.add_subplot(222)
plt.imshow(c1_mat, cmap=cmap, interpolation='nearest', origin='low',
           vmin=mean_c1 - sigma_cut*mean_c1, vmax=mean_c1 + sigma_cut*mean_c1)
plt.colorbar()
ax.set_title(r"$c_1$", size=13)

ax = fig.add_subplot(223)
#plt.imshow(c2_mat, cmap='seismic', interpolation='nearest', origin='low', vmin=mean_c2 - sigma_cut*mean_c2, vmax=mean_c2 + sigma_cut*mean_c2)
plt.imshow(c2_mat, cmap=cmap, interpolation='nearest',
           origin='low', vmin=-1.4e-6, vmax=1.4e-6)
plt.colorbar()
ax.set_title(r"$c_2$", size=13)

ax = fig.add_subplot(224)
plt.imshow(chi2_mat, cmap=cmap, interpolation='nearest', origin='low',
           vmin=mean_chi2 - sigma_cut*mean_chi2, vmax=mean_chi2 + sigma_cut*mean_chi2)
plt.colorbar()
ax.set_title(r"$\chi^2/\nu$", size=13)
# fig.suptitle(fit_string)
plt.tight_layout()
pp.savefig(fig)


# Do the plots again, with the corrected vectors and matrices

fig = plt.figure()
ax = fig.add_subplot(221)
n, bins, patches_out = ax.hist(c0_corr_vec, 50, normed=True, facecolor='green',
                               alpha=0.75, label='Mean: %g \n Scatter:%g' % (mean_c0_corr, scatter_c0_corr))
ax.set_title('Histogram of c0 after %g-sigma clipping' % sigma_cut, size=8)
ax.legend(loc=loc_label, fancybox=True, ncol=1, numpoints=1, prop=prop)
ax.tick_params(axis='both', which='major', labelsize=7)


ax = fig.add_subplot(222)
n, bins, patches_out = ax.hist(c1_corr_vec, 50, normed=True, facecolor='yellow',
                               alpha=0.75, label='Mean: %g \n Scatter:%g' % (mean_c1_corr, scatter_c1_corr))
ax.set_title('Histogram of c1 after %g-sigma clipping' % sigma_cut, size=8)
ax.legend(loc=loc_label, fancybox=True, ncol=1, numpoints=1, prop=prop)
ax.tick_params(axis='both', which='major', labelsize=7)

ax = fig.add_subplot(223)
n, bins, patches_out = ax.hist(c2_corr_vec, 50, normed=True, facecolor='red',
                               alpha=0.75, label='Mean: %g \n Scatter:%g' % (mean_c2_corr, scatter_c2_corr))
ax.set_title('Histogram of c2 after %g-sigma clipping' % sigma_cut, size=8)
ax.legend(loc=loc_label, fancybox=True, ncol=1, numpoints=1, prop=prop)
ax.tick_params(axis='both', which='major', labelsize=7)

# with open("all_c2.txt", "a") as myfile:
#    myfile.write("%g %g \n"%(mean_c2, scatter_c2))


ax = fig.add_subplot(224)
n, bins, patches_out = ax.hist(chi2_corr_vec, 50, normed=True, facecolor='blue',
                               alpha=0.75, label='Mean: %g \n Scatter:%g' % (mean_chi2_corr, scatter_chi2_corr))
ax.set_title('Histogram of chi2 after %g-sigma clipping' % sigma_cut, size=8)
ax.legend(loc=loc_label, fancybox=True, ncol=1, numpoints=1, prop=prop)
ax.tick_params(axis='both', which='major', labelsize=7)

fig.suptitle("CORRECTED")
plt.tight_layout()
pp.savefig(fig)


fig = plt.figure()
ax = fig.add_subplot(221)
plt.imshow(c0_corr_mat, cmap=cmap, interpolation='nearest', origin='low',
           vmin=mean_c0_corr - sigma_cut*mean_c0_corr, vmax=mean_c0_corr + sigma_cut*mean_c0_corr)
plt.colorbar()
ax.set_title(r"$c_0$", size=13)

ax = fig.add_subplot(222)
plt.imshow(c1_corr_mat, cmap=cmap, interpolation='nearest', origin='low',
           vmin=mean_c1_corr - sigma_cut*mean_c1_corr, vmax=mean_c1_corr + sigma_cut*mean_c1_corr)
plt.colorbar()
ax.set_title(r"$c_1$", size=13)

ax = fig.add_subplot(223)
#plt.imshow(c2_mat, cmap='seismic', interpolation='nearest', origin='low', vmin=mean_c2 - sigma_cut*mean_c2, vmax=mean_c2 + sigma_cut*mean_c2)
plt.imshow(c2_corr_mat, cmap=cmap, interpolation='nearest',
           origin='low', vmin=-1.5e-6, vmax=1.5e-6)
plt.colorbar()
ax.set_title(r"$c_2$", size=13)

ax = fig.add_subplot(224)
plt.imshow(chi2_corr_mat, cmap=cmap, interpolation='nearest', origin='low',
           vmin=mean_chi2_corr - sigma_cut*mean_chi2_corr, vmax=mean_chi2_corr + sigma_cut*mean_chi2_corr)
plt.colorbar()
ax.set_title(r"$\chi^2/\nu$", size=13)
fig.suptitle("CORRECTED")
plt.tight_layout()
pp.savefig(fig)


"""
##### Plot for PACCD16 paper

fig=plt.figure()
ax=fig.add_subplot(211)
ax.errorbar(t_vec, s_vec, yerr=s_e_vec, fmt='b-o', alpha=0.8, label=" $C_0$: %g $\pm$ %g \n \n $C_1$: %g $\pm$ %g \n \n $C_2$: %g $\pm$ %g" %(p[0], pe[0], p[1], pe[1], p[2], pe[2]))
plt.legend(loc='lower right', fancybox=True, ncol=1, numpoints=1, prop = prop)
plt.xlabel ('Frame number')
plt.ylabel ('Mean signal (e$^{-}$)')

ax=fig.add_subplot(223)
plt.imshow(c2_mat, cmap='seismic', interpolation='nearest', origin='low', vmin=mean_c2 - sigma_cut*mean_c2, vmax=mean_c2 + sigma_cut*mean_c2)
plt.colorbar()
ax.set_title (r"$C_2$", size=11)

ax = fig.add_subplot(224)
n, bins, patches_out = ax.hist(c2_vec, 50, normed=True, facecolor='red', alpha=0.75, label=' Mean: %g \n Scatter:%g'%(mean_c2, scatter_c2))
#ax.set_title('Histogram of c2 after %g-sigma clipping' %sigma_cut, size=10)
ax.legend(loc=loc_label , fancybox=True, ncol=1, numpoints=1, prop = prop)
ax.tick_params(axis='both', which='major', labelsize=11.5)
plt.tight_layout()
pp.savefig(fig)
"""

pp.close()
