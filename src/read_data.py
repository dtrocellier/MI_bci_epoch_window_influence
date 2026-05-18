import os
import os.path as osp
import numpy as np
import mne
import torch
import multiprocessing as mp
import ctypes
from sklearn.model_selection import train_test_split
import fnmatch


def read_korea_data(data_path,filtering_setting):
    """
    Load each subject's data as a shared array
    Args:
        Common_config: common config
        TargetC_config: the config of the target client

    Returns:
        Target_data:
        [TarC_Sub1_data_shared_array, TarC_Sub2_shared_array, ..., TarC_SubN_shared_array]
        In TarC_Sub1_data_shared_array:
        [shared_data, shared_label]
        shared_data: (#trials, 1, #channels, #timesteps)
        shared_label: (#trials, )
    """

    # data_path = "..\\..\\..\\Dataset\\Korea\\Filtered_MI_data"
    # filtering_setting = "Filterfirst\\0.3Hz_40Hz_cheby2_sos"
    name = "KoreaU_MI"
    num_folds = 54

    Target_data = []
    client_data_path = os.path.join(
        data_path,
        name,
        filtering_setting,
    )

    files_name_list_temp = data_load(client_data_path)
    if len(files_name_list_temp) > num_folds:
        # only use non-session-wise partitioned file name list
        files_name_list = [f for f in files_name_list_temp if '_' not in f]
    else:
        files_name_list = files_name_list_temp

    for file in files_name_list:
        file_name = os.path.join(client_data_path, file)
        onesub_data = np.load(file_name)
        x_data = onesub_data["x_data"]
        y_data = onesub_data["y_data"]
        # covert the data size into (#trials, 1, #channel, #timesteps) to fit the conv2d
        x_data = np.expand_dims(x_data, axis=1)
        num_trials = x_data.shape[0]
        onesub_data.close()

        shared_array_base = mp.Array(
            ctypes.c_float, x_data.shape[0] * 1 * x_data.shape[2] * x_data.shape[3]
        )
        shared_data = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_data = shared_data.reshape(
            x_data.shape[0], 1, x_data.shape[2], x_data.shape[3]
        )
        shared_data = torch.from_numpy(shared_data)

        # label size is (#trials, )
        shared_array_label_base = mp.Array(ctypes.c_long, num_trials)
        shared_label = np.ctypeslib.as_array(shared_array_label_base.get_obj())
        shared_label = shared_label.reshape(num_trials, 1)
        shared_label = torch.from_numpy(shared_label)

        # load the data into the shared array
        for index in range(num_trials):
            shared_data[index] = torch.from_numpy(x_data[index, :, :, :]).float()
        shared_label[:, 0] = torch.from_numpy(y_data).long()
        shared_label = shared_label.reshape(num_trials)

        del x_data, y_data
        Target_data.append([shared_data, shared_label])

    return Target_data

def data_load(data_path):
    """
    Load the data file name into a list
    Params:
        data_path: str, data path contains all filtered EEG signal
    Return:
        a list,  a list of file names stored in the given data path
    """

    files = os.listdir(data_path)
    return sorted(files)


def read_bigdataset_data(init_path):
    """ Take the path of the data and return the list of the data directory for each participant
    """

    files_dir=os.listdir(init_path)[:3]
    files_dir.sort()

    participant_dir= [os.listdir(osp.join(init_path,files_dir[i]))for i in range(len(files_dir))]
    for list_participant in participant_dir : list_participant.sort() # Manage the fact that on plafrim the files are not in the same order

    print("you have succesfuly acces to the directory : ",init_path)
    # remove the bad participant
    print(files_dir)
    print(participant_dir)


    try : # remove the bad participant
        participant_dir[0].remove("A40")
        participant_dir[0].remove("A59")
    except ValueError : # except if they are already removed (on the big dataset cleaned)
        pass


    return  participant_dir , files_dir


def collect_data(files_dir, participant_dir, init_path):
    """ From the given  path, read the GDF files and return a dictionnary with the data as mne raw object
    """
    dic_data={}

    for i in range(len(files_dir)):
        for j in range(len(participant_dir[i])):
            #Train dataset
            dic_data[participant_dir[i][j]+"_1"]= mne.io.read_raw_gdf(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_R1_acquisition.gdf"), verbose="CRITICAL")
            dic_data[participant_dir[i][j]+"_2"]= mne.io.read_raw_gdf(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_R2_acquisition.gdf"), verbose="CRITICAL")

            #Test dataset
            dic_data[participant_dir[i][j]+"_3"]= mne.io.read_raw_gdf(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_R3_onlineT.gdf"), verbose="CRITICAL")
            dic_data[participant_dir[i][j]+"_4"]= mne.io.read_raw_gdf(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_R4_onlineT.gdf"), verbose="CRITICAL")
            try : # allow to manage the one where there is no _5 and _6 files
                dic_data[participant_dir[i][j]+"_5"]= mne.io.read_raw_gdf(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_R5_onlineT.gdf"), verbose="CRITICAL")
            except FileNotFoundError:
                pass

            try :
                dic_data[participant_dir[i][j]+"_6"]= mne.io.read_raw_gdf(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_R6_onlineT.gdf"), verbose="CRITICAL")
            except FileNotFoundError:
                pass
    return dic_data


def collect_preprocess_data(files_dir, participant_dir, init_path):
    """ From the given  path, read the GDF files and return a dictionnary with the data as mne raw object
    """
    dic_data={}

    for i in range(len(files_dir)):
        for j in range(len(participant_dir[i])):
            #Train dataset
            dic_data[participant_dir[i][j]+"_1"]= mne.io.read_raw_fif(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_1_eeg.fif"), verbose="CRITICAL")
            dic_data[participant_dir[i][j]+"_2"]= mne.io.read_raw_fif(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_2_eeg.fif"), verbose="CRITICAL")

            #Test dataset
            dic_data[participant_dir[i][j]+"_3"]= mne.io.read_raw_fif(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_3_eeg.fif"), verbose="CRITICAL")
            dic_data[participant_dir[i][j]+"_4"]= mne.io.read_raw_fif(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_4_eeg.fif"), verbose="CRITICAL")
            try : # allow to manage the one where there is no _5 and _6 files
                dic_data[participant_dir[i][j]+"_5"]= mne.io.read_raw_fif(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_5_eeg.fif"), verbose="CRITICAL")
            except FileNotFoundError:
                pass

            try :
                dic_data[participant_dir[i][j]+"_6"]= mne.io.read_raw_fif(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_6_eeg.fif"), verbose="CRITICAL")
            except FileNotFoundError:
                pass
    return dic_data


def collect_baseline(files_dir, participant_dir, init_path):
    """ From the given  path, read the GDF files and return a dictionnary with the data as mne raw object
    """
    dic_baseline = {}

    for i in range(len(participant_dir)):
        for j in range(len(participant_dir[i])):
            for file in os.listdir(os.path.join(init_path,files_dir[i],participant_dir[i][j])):
                if fnmatch.fnmatch(file, '*OE*.gdf'):
                    dic_baseline[participant_dir[i][j] + "_baseline"] =  mne.io.read_raw_gdf( osp.join(init_path, files_dir[i], participant_dir[i][j], file))

    return dic_baseline


def extrac_train_test_within(all_subject, test_subject):
    """Extract the train and test key from the list off 'all bubject' and 'test subject' """

    train_key = []
    test_key = []
    list_subjects_train = all_subject.copy()

    for subject in test_subject : # Each subject in the list_subject is remove from the train dataset
        list_subjects_train.remove(subject)
        test_key += [subject  +"_"+ str(i) for i in range(3,7)] # and add it to the train key dataset

    for subject in test_subject : # for the ra
        train_key += [subject  +"_"+ str(i) for i in range(1,3)] # and add it to the train key dataset






    return train_key, test_key

def extrac_train_test(all_subject, test_subject):
    """Extract the train and test key from the list off 'all bubject' and 'test subject' """

    train_key = []
    test_key = []
    list_subjects_train = all_subject.copy()

    for subject in test_subject : # Each subject in the list_subject is remove from the train dataset
        list_subjects_train.remove(subject)
        test_key += [subject  +"_"+ str(i) for i in range(3,7)] # and add it to the train key dataset

    for subj in list_subjects_train : # for the ra
        if subj == "A59":
            train_key += [subj +"_"+ str(i) for i in range(1,5)]
        else :
            train_key += [subj +"_"+ str(i) for i in range(1,7)]




    return train_key, test_key


def collect_npz_data(files_dir, participant_dir, init_path):
    """
    Load each subject's data as a shared array
    Args:
        Common_config: common config
        TargetC_config: the config of the target client

    Returns:
        Target_data:
        [TarC_Sub1_data_shared_array, TarC_Sub2_shared_array, ..., TarC_SubN_shared_array]
        In TarC_Sub1_data_shared_array:
        [shared_data, shared_label]
        shared_data: (#trials, 1, #channels, #timesteps)
        shared_label: (#trials, )
    """

    # data_path = "..\\..\\..\\Dataset\\Korea\\Filtered_MI_data"
    # filtering_setting = "Filterfirst\\0.3Hz_40Hz_cheby2_sos"
    dic_data={}

    for i in range(len(files_dir)):
        for j in range(len(participant_dir[i])):
            #Train dataset
            dic_data[participant_dir[i][j]+"_1"]= np.load(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_1_eeg.npz"))
            dic_data[participant_dir[i][j]+"_2"]= np.load(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_2_eeg.npz"))

            #Test dataset
            dic_data[participant_dir[i][j]+"_3"]= np.load(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_3_eeg.npz"))
            dic_data[participant_dir[i][j]+"_4"]= np.load(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_4_eeg.npz"))
            try : # allow to manage the one where there is no _5 and _6 files
                dic_data[participant_dir[i][j]+"_5"]= np.load(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_5_eeg.npz"))
            except FileNotFoundError:
                pass

            try :
                dic_data[participant_dir[i][j]+"_6"]= np.load(osp.join(init_path,files_dir[i],participant_dir[i][j],participant_dir[i][j]+"_6_eeg.npz"))
            except FileNotFoundError:
                pass
    return dic_data

def collect_korea_data(data_path,filtering_setting):

    dic_data={}

    print(os.listdir("Dataset"))

    Target_data =read_korea_data(data_path,filtering_setting)

    for i in range(len(Target_data)):
        dic_data["subject_"+str(i+1)]= Target_data[i]

    return dic_data



def load_data(path='Dataset', window='0_4'):
    """Returns X and Y for the given epoch window.

    Parameters:
    path (str): base dataset path
    window (str): epoch window suffix, e.g. '0_4' or '0.5_4.5'

    Returns:
    X, Y (lists of torch tensors)
    """
    dataset = 'Large'
    X = torch.load(path + '/' + dataset + '/X_s_' + window + '.pt')
    Y = torch.load(path + '/' + dataset + '/Y_s_' + window + '.pt')
    return X, Y

def load_data_yassine_FB(dict_config, path='Dataset'):
    """Returns X and Y according to the configuration dict

    Parameters:
    dict_config (dict): configuration dict

    Returns:
    X (list of torch tensors): EEG time series with FB
    Y (list of torch tensors): Associated labels

   """
    dict_classes = {'BNCI':4,'BNCI2':2,'Large':2}
    dict_freq = {'BNCI':250,'BNCI2':250,'Large':512}
    dict_config['n_classes'] = 2
    dict_config['sfreq']= 512

    dataset='Large'


    X_EA = torch.load(path + '/' +dataset+'/X_FB.pt')

    assert dict_config['cross'] != dict_config['ft']
    if dict_config['cross']==True:
        X_online = torch.load(path + '/' +dataset+'/X_FB.pt')
    if dict_config['ft']==True:
        X_online = torch.load(path + '/' +dataset+'/X_EA_online_ft.pt')

    Y = torch.load(path + '/' +dataset+'/Y_s.pt')

    if dict_config['model']=='EEGSimpleConv':
        n_chan = X_EA[0][0].shape[1]
        dict_config['params'].append(n_chan)
        dict_config['params'].append(dict_config['n_classes'])
        dict_config['params'].append(dict_config['sfreq'])
        if dict_config['reg_subject']:
            n_subjects = len(X_EA)
            dict_config['params'].append(n_subjects)
            # Check
    if dict_config['reg_subject']:
        assert len(dict_config['params'])==9
    else  :
        assert len(dict_config['params'])==7
    return X_EA,X_online,Y



def loaders_cross_yassine_FB(idx,X_EA,X_online,Y,dataset,reg_subject):
    # Test on the second test set if BNCI and in the 3,4,5,6 for Large database
    sep = 1 if dataset in ['BNCI','BNCI2'] else 2
    n_chan = X_online[0][0].shape[1]
    batch_size = 288 if dataset == 'BNCI' else 256
    n_filters = X_online[0][0].shape[-1]


    train_key= list(range(len(X_EA)))
    train_key.remove(idx)
    train_key , valid_key = train_test_split(train_key, test_size=0.2, random_state=42)

    #Training set
    X_ =  [X_EA[i] for i in train_key]
    train_X = torch.cat([item for sublist in X_ for item in sublist])
    Y_ =  [Y[i] for i in train_key]
    train_Y = torch.cat([item for sublist in Y_ for item in sublist])

    #validation set
    X_ = [X_EA[i] for i in valid_key]
    val_X = torch.cat([item for sublist in X_ for item in sublist])
    Y_ = [Y[i] for i in valid_key]
    val_Y = torch.cat([item for sublist in Y_ for item in sublist])

    #Offline test set
    test_X_ = [X_EA[idx][sep:]]
    test_X_off = torch.cat([item for sublist in test_X_ for item in sublist])
    test_Y_ = [Y[idx][sep:]]
    test_Y_off = torch.cat([item for sublist in test_Y_ for item in sublist])
    # test_X_off = torch.cat(X_EA[idx][sep:])
    # test_Y_off = torch.cat(Y[idx][sep:])

    #Online test set
    test_X_ = [X_online[idx][sep:]]
    test_X_on = torch.cat([item for sublist in test_X_ for item in sublist])
    test_Y_ = [Y[idx][sep:]]
    test_Y_on = torch.cat([item for sublist in test_Y_ for item in sublist])
    # test_X_on = torch.cat(X_online[idx][sep:])
    # test_Y_on = torch.cat(Y[idx][sep:])

    if reg_subject:
        print("La division entre train et test n'est pas encore faite pour la regressionn subject")
        raise NotImplementedError
        #Subject-regularization target
        Y_subject = [[torch.tensor([i]*XXX.shape[0]) for XXX in XX] for i,XX in enumerate(X_EA)]
        Y_subject_ = Y_subject[:idx] + Y_subject[idx+1:]
        train_Y_subject = torch.unbind(torch.cat([item for sublist in Y_subject_ for item in sublist]))
        test_Y_subject = Y_subject[idx]


    # Standardization with training set stats
    mean = train_X.transpose(1,2).reshape(-1, n_chan, n_filters).mean(dim = 0)
    std = train_X.transpose(1,2).reshape(-1, n_chan, n_filters).std(dim = 0)

    # print("mean",mean.size(),  mean)
    # print("std", std.size(), std)

    train_X = (train_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    val_X = (val_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    test_X_off = (test_X_off - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    test_X_on = (test_X_on - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)

    # Add a supplementary dimension so that the shape is  (batch x 1 x chan x time x filterBand)
    train_X = train_X.unsqueeze(1)
    val_X = val_X.unsqueeze(1)
    test_X_off = test_X_off.unsqueeze(1)
    test_X_on = test_X_on.unsqueeze(1)



    test_X_off = torch.unbind(test_X_off)
    test_Y_off = torch.unbind(test_Y_off)
    test_data_off = list(zip(test_X_off, test_Y_off)) if reg_subject else list(zip(test_X_off, test_Y_off))
    test_loader_off = torch.utils.data.DataLoader(test_data_off, batch_size = batch_size, shuffle = True, drop_last = False, num_workers = 0)

    test_X_on = torch.unbind(test_X_on)
    test_Y_on = torch.unbind(test_Y_on)
    test_data_on = list(zip(test_X_on, test_Y_on,test_Y_subject)) if reg_subject else list(zip(test_X_on, test_Y_on))
    test_loader_on = torch.utils.data.DataLoader(test_data_on, batch_size = batch_size, shuffle = True, drop_last = False, num_workers = 0)


    train_X = torch.unbind(train_X)
    train_Y = torch.unbind(train_Y)
    train_data = list(zip(train_X, train_Y,train_Y_subject)) if reg_subject else list(zip(train_X, train_Y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 0)

    val_X = torch.unbind(val_X)
    val_Y = torch.unbind(val_Y)
    val_data = list(zip(val_X, val_Y,val_Y_subject)) if reg_subject else list(zip(val_X, val_Y))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 0)

    return train_loader, val_loader, test_loader_off, test_loader_on



def loaders_cross(idx, X, Y, dataset, reg_subject):
    sep = 1 if dataset in ['BNCI', 'BNCI2'] else 2
    n_chan = X[0][0].shape[1]
    batch_size = 288 if dataset == 'BNCI' else 256

    if isinstance(idx, int):
        idx = [idx]

    train_key = list(range(len(X)))
    for i_test_subject in idx:
        train_key.remove(i_test_subject)
    train_key, valid_key = train_test_split(train_key, test_size=0.2, random_state=42)

    train_X = torch.cat([item for i in train_key for item in X[i]])
    train_Y = torch.cat([item for i in train_key for item in Y[i]])

    val_X = torch.cat([item for i in valid_key for item in X[i]])
    val_Y = torch.cat([item for i in valid_key for item in Y[i]])

    test_X = torch.cat([item for i in idx for item in X[i][sep:]])
    test_Y = torch.cat([item for i in idx for item in Y[i][sep:]])

    if reg_subject:
        raise NotImplementedError("reg_subject is not supported")

    mean = train_X.transpose(1, 2).reshape(-1, n_chan).mean(dim=0)
    std = train_X.transpose(1, 2).reshape(-1, n_chan).std(dim=0)

    train_X = (train_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    val_X = (val_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    test_X = (test_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)

    train_loader = torch.utils.data.DataLoader(
        list(zip(torch.unbind(train_X), torch.unbind(train_Y))),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        list(zip(torch.unbind(val_X), torch.unbind(val_Y))),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        list(zip(torch.unbind(test_X), torch.unbind(test_Y))),
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )

    return train_loader, val_loader, test_loader

def loaders_within(idx,X_EA,X_online,Y,dataset,reg_subject):
    # Test on the second test set if BNCI and in the 3,4,5,6 for Large database
    sep = 1 if dataset in ['BNCI','BNCI2'] else 2
    n_chan = X_EA[0][0].shape[1]
    batch_size = 288 if dataset == 'BNCI' else 16

    # Verifie si idx est un entier si oui on le met dans une liste
    if isinstance(idx, list):
        raise ValueError("idx should be an integer")


    # Collect only the idx subject because we are in the within subject setting
    X_EA= X_EA[idx]
    X_online = X_online[idx]
    Y = Y[idx]

    n_trials = X_EA[0].shape[0]

    start_idx = 0
    stop_idx_train = len(X_EA[0])+len(X_EA[1])
    stop_idx_test = len(X_EA[2])+len(X_EA[3])+len(X_EA[4])+len(X_EA[5])

    train_key = list(range(start_idx, stop_idx_train)) # take the number of trials of the first and second run
    test_key = list(range(stop_idx_train, stop_idx_test)) # take the number of trials of the third, fourth, fifth and sixth run

    train_key , valid_key = train_test_split(train_key, test_size=0.2, random_state=42)

    #Training set
    X_ =  [X_EA[i//n_trials][i%n_trials] for i in train_key]
    train_X = torch.stack(X_)
    Y_ =  [Y[i//n_trials][i%n_trials] for i in train_key]
    train_Y = torch.stack(Y_)

    #validation set
    X_ = [X_EA[i//n_trials][i%n_trials] for i in valid_key]
    val_X = torch.stack(X_)
    Y_ = [Y[i//n_trials][i%n_trials]for i in valid_key]
    val_Y = torch.stack(Y_)

    # #Offline test set
    test_X_ = [X_EA[i//n_trials][i%n_trials] for i in test_key]
    test_X_off = torch.stack(test_X_)
    test_Y_ = [Y[i//n_trials][i%n_trials] for i in test_key]
    test_Y_off = torch.stack(test_Y_)
    # test_X_off = torch.cat(X_EA[idx][sep:])
    # test_Y_off = torch.cat(Y[idx][sep:])

    #Online test set
    test_X_ = [X_online[i//n_trials][i%n_trials] for i in test_key]
    test_X_on = torch.stack(test_X_)
    test_Y_ = [Y[i//n_trials][i%n_trials] for i in test_key]
    test_Y_on = torch.stack(test_Y_)
    # test_X_on = torch.cat(X_online[idx][sep:])
    # test_Y_on = torch.cat(Y[idx][sep:])

    if reg_subject:
        print("La division entre train et test n'est pas encore faite pour la regressionn subject")
        raise NotImplementedError
        #Subject-regularization target
        Y_subject = [[torch.tensor([i]*XXX.shape[0]) for XXX in XX] for i,XX in enumerate(X_EA)]
        Y_subject_ = Y_subject[:idx] + Y_subject[idx+1:]
        train_Y_subject = torch.unbind(torch.cat([item for sublist in Y_subject_ for item in sublist]))
        test_Y_subject = Y_subject[idx]


    # Standardization with training set stats
    mean = train_X.transpose(1,2).reshape(-1, n_chan).mean(dim = 0)
    std = train_X.transpose(1,2).reshape(-1, n_chan).std(dim = 0)

    # print("mean",mean.size(),  mean)
    # print("std", std.size(), std)

    train_X = (train_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    val_X = (val_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    test_X_off = (test_X_off - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    test_X_on = (test_X_on - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)


    test_X_off = torch.unbind(test_X_off)
    test_Y_off = torch.unbind(test_Y_off)
    test_data_off = list(zip(test_X_off, test_Y_off)) if reg_subject else list(zip(test_X_off, test_Y_off))
    test_loader_off = torch.utils.data.DataLoader(test_data_off, batch_size = batch_size, shuffle = True, drop_last = False, num_workers = 0)

    test_X_on = torch.unbind(test_X_on)
    test_Y_on = torch.unbind(test_Y_on)
    test_data_on = list(zip(test_X_on, test_Y_on,test_Y_subject)) if reg_subject else list(zip(test_X_on, test_Y_on))
    test_loader_on = torch.utils.data.DataLoader(test_data_on, batch_size = batch_size, shuffle = True, drop_last = False, num_workers = 0)


    train_X = torch.unbind(train_X)
    train_Y = torch.unbind(train_Y)
    train_data = list(zip(train_X, train_Y,train_Y_subject)) if reg_subject else list(zip(train_X, train_Y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 0)

    val_X = torch.unbind(val_X)
    val_Y = torch.unbind(val_Y)
    val_data = list(zip(val_X, val_Y,val_Y_subject)) if reg_subject else list(zip(val_X, val_Y))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 0)

    return train_loader, val_loader, test_loader_off, test_loader_on




def loaders_cross(idx,X_online,Y,dataset,reg_subject):
    # Test on the second test set if BNCI and in the 3,4,5,6 for Large database
    sep = 1 if dataset in ['BNCI','BNCI2'] else 2
    n_chan = X_online[0][0].shape[1]
    batch_size = 288 if dataset == 'BNCI' else 256


    train_key= list(range(len(X_online)))
    train_key.remove(idx)
    train_key , valid_key = train_test_split(train_key, test_size=0.2, random_state=42)

    #Training set
    X_ =  [X_online[i] for i in train_key]
    train_X = torch.cat([item for sublist in X_ for item in sublist])
    Y_ =  [Y[i] for i in train_key]
    train_Y = torch.cat([item for sublist in Y_ for item in sublist])

    #validation set
    X_ = [X_online[i] for i in valid_key]
    val_X = torch.cat([item for sublist in X_ for item in sublist])
    Y_ = [Y[i] for i in valid_key]
    val_Y = torch.cat([item for sublist in Y_ for item in sublist])

    #Online test set
    test_X_ = [X_online[idx][sep:]]
    test_X_on = torch.cat([item for sublist in test_X_ for item in sublist])
    test_Y_ = [Y[idx][sep:]]
    test_Y_on = torch.cat([item for sublist in test_Y_ for item in sublist])


    if reg_subject:
        print("La division entre train et test n'est pas encore faite pour la regressionn subject")
        raise NotImplementedError
        #Subject-regularization target
        Y_subject = [[torch.tensor([i]*XXX.shape[0]) for XXX in XX] for i,XX in enumerate(X_online)]
        Y_subject_ = Y_subject[:idx] + Y_subject[idx+1:]
        train_Y_subject = torch.unbind(torch.cat([item for sublist in Y_subject_ for item in sublist]))
        test_Y_subject = Y_subject[idx]


    # Standardization with training set stats
    mean = train_X.transpose(1,2).reshape(-1, n_chan).mean(dim = 0)
    std = train_X.transpose(1,2).reshape(-1, n_chan).std(dim = 0)
    #
    # print("mean",mean.size(),  mean)
    # print("std", std.size(), std)

    train_X = (train_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    val_X = (val_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)
    test_X_on = (test_X_on - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)

    test_X_on = torch.unbind(test_X_on)
    test_Y_on = torch.unbind(test_Y_on)
    test_data_on = list(zip(test_X_on, test_Y_on,test_Y_subject)) if reg_subject else list(zip(test_X_on, test_Y_on))
    test_loader_on = torch.utils.data.DataLoader(test_data_on, batch_size = batch_size, shuffle = True, drop_last = False, num_workers = 0)

    train_X = torch.unbind(train_X)
    train_Y = torch.unbind(train_Y)
    train_data = list(zip(train_X, train_Y,train_Y_subject)) if reg_subject else list(zip(train_X, train_Y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 0)

    val_X = torch.unbind(val_X)
    val_Y = torch.unbind(val_Y)
    val_data = list(zip(val_X, val_Y,val_Y_subject)) if reg_subject else list(zip(val_X, val_Y))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 0)

    return train_loader, val_loader, test_loader_on









#%%
