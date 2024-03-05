import uproot
import numpy as np
import math
import glob

mass_all = np.array([])
lgdEdXmx_all = np.array([])
zenith_all = np.array([])
integral_all = np.array([])
X_all = []
dEdX_all = []

noise = False # Whether to add noise to dEdX values

num_files = 0
# Open files in each folder
for i in range(0, 21):
    foldername = f'/Data/Simulations/Conex_Flat_lnA/EPOS/Conex_170-205_Prod{i}/showers/*.root'
    print(foldername)
    for filename in glob.glob(foldername):
        num_files += 1
        filecx=uproot.open(f'{filename}')
        try:
            tshowercx=filecx["Shower"]
            theadercx=filecx["Header"]
        except:
            print("Issue getting shower/header information")
            continue

        # for k in list(tshowercx.keys()):
        #     print(k)
        #     try:
        #         print(np.shape(tshowercx[k].array()))
        #     except:
        #         print("not a standard shape")
        
        # lgEcx=tshowercx["lgE"].array()
        # azimuthcx=tshowercx["azimuth"].array()
        # Seed2cx=tshowercx["Seed2"].array()
        # Seed3cx=tshowercx["Seed3"].array()
        # Xfirstcx=tshowercx["Xfirst"].array()
        # Hfirstcx=tshowercx["Hfirst"].array()
        # XfirstIncx=tshowercx["XfirstIn"].array()
        # altitudecx=tshowercx["altitude"].array()
        # X0cx=tshowercx["X0"].array()
        # Xmaxcx=tshowercx["Xmax"].array()
        # Nmaxcx=tshowercx["Nmax"].array()
        # p1cx=tshowercx["p1"].array()
        # p2cx=tshowercx["p2"].array()
        # p3cx=tshowercx["p3"].array()
        # chi2cx=tshowercx["chi2"].array()
        # Nmxcx=tshowercx["Nmx"].array()
        # XmxdEdXcx=tshowercx["XmxdEdX"].array()
        # dEdXmxcx=tshowercx["dEdXmx"].array()
        # lgdEdXmxcx = [math.log(dE) for dE in dEdXmxcx]
        # lgdEdXmx_all = np.append(lgdEdXmx_all, lgdEdXmxcx)
        # cpuTimecx=tshowercx["cpuTime"].array()
        # nXcx=tshowercx["nX"].array()
        

        zenithcx=tshowercx["zenith"].array()
        zenith_all = np.append(zenith_all, zenithcx)
        Xcx=tshowercx["X"].array()
        dEdX=tshowercx["dEdX"].array()

        mass = math.log(theadercx['Particle'].array()[0]/100)
        masses = mass * np.ones(len(zenithcx))
        mass_all = np.append(mass_all, masses)

        for x_arr in Xcx:
            x_arr = [0 if np.isnan(x) else x for x in x_arr]
            X_all.append(x_arr)

        for dE_arr in dEdX:
            if noise:
                # Add Gaussian noise to each dEdX value
                vals_with_noise = []
                for dE_index in range(len(dE_arr)):
                    noise_mean = 500  # Mean of the Gaussian noise
                    noise_std = 250  # Standard deviation of the Gaussian noise
                    noise = np.random.normal(noise_mean, noise_std)
                    vals_with_noise.append(dE_arr[dE_index] + noise)
                dE_arr = vals_with_noise
            
            # Rescale logarithmically
            dE_arr = [0 if x<=1 else math.log(x) for x in dE_arr]
            dEdX_all.append(dE_arr)

print("Number of files:", num_files)

dEdX_all = np.array(dEdX_all, dtype=object)
X_all = np.array(X_all, dtype=object)

# # Compute the integral
# for i in range(len(X_all)):
#     integral = np.trapz(dEdX_all[i], X_all[i])
#     integral_all = np.append(integral_all, integral)
# # Take the log of it
# integral_all = [0 if x<=1 else math.log(x) for x in integral_all]
# # Normalize the integral
# min_val_int = np.min(integral_all, axis=0)
# max_val_int = np.max(integral_all, axis=0)
# integral_all = (integral_all - min_val_int) / (max_val_int - min_val_int)

# # Normalize log(dEdXmx) between 0 and 1(should this be normalized?)
# min_val_Emx = np.min(lgdEdXmx_all, axis=0)
# max_val_Emx = np.max(lgdEdXmx_all, axis=0)
# lgdEdXmx_all = (lgdEdXmx_all - min_val_Emx) / (max_val_Emx - min_val_Emx)

# Normalize zenith angle between 0 and 1
min_val_zen = np.min(zenith_all, axis=0)
max_val_zen = np.max(zenith_all, axis=0)
zenith_all = (zenith_all - min_val_zen) / (max_val_zen - min_val_zen)

print(f"Min Zenith Angle: {min_val_zen}")
print(f"Max Zenith Angle: {max_val_zen}")

# Normalize X between 0 and 1
flattened_X_all = np.concatenate(X_all)
min_val_X = np.min(flattened_X_all)
max_val_X = np.max(flattened_X_all)
X_all = [[(X - min_val_X) / (max_val_X - min_val_X) for X in entry] for entry in X_all]

print(f"Min X: {min_val_X}")
print(f"Max X: {max_val_X}")

# Pad X and dEdX
max_len = np.max([len(arr) for arr in dEdX_all])
X_all = [np.pad(arr, (0, max_len-len(arr))) for arr in X_all]
dEdX_all = [np.pad(arr, (0, max_len-len(arr))) for arr in dEdX_all]

print(f'Mass shape: {np.shape(mass_all)}')
print(f'Zenith shape: {np.shape(zenith_all)}')
print(f'X shape: {np.shape(X_all)}')
print(f'dEdX shape: {np.shape(dEdX_all)}')


# Save the arrays to an npz file
if noise:
    np.savez('/DataFast/zwang/data_with_noise.npz', mass=mass_all, zenith=zenith_all, x=X_all, dEdX=dEdX_all)    
else:
    np.savez('/DataFast/zwang/data_prod_0_to_20.npz', mass=mass_all, zenith=zenith_all, x=X_all, dEdX=dEdX_all)