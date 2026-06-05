
import mne
from mne.preprocessing import EOGRegression
import numpy as np
from tqdm import tqdm
#from tensorflow import one_hot


def preprocess(raw, steps = {}):
    """ preprocess the data"""
    assert isinstance(steps, dict), "les steps doivent être un dictionnaire d'étapes"
    raw.load_data()
    if "remove_eog_artifact" in steps.keys():

        mne.set_eeg_reference(raw, ref_channels=['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2', 'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'], copy=False, ch_type="eeg")
        raw.set_channel_types({"EOG1": "eog" , "EOG2": "eog" , "EOG3": "eog" })
        weights = EOGRegression().fit(raw)
        weights.apply(raw, copy=False)

    if "drop_channels" in steps.keys():
        #remove the wanted channels
        for channel in steps["drop_channels"] : #Pour chaque channel  a supprimer
            if channel in raw.ch_names: raw.drop_channels(channel) # Vérifie qu'il est present et le supprime

    if "filter" in steps.keys():
        assert isinstance(steps["filter"], list), "les paramètres de 'filter' doivent une liste suivant cette forme [l_freq,h_freq]"
        raw.filter(steps["filter"][0], steps["filter"][1])

    if "downsample" in steps.keys():
        assert isinstance(steps["downsample"], int), "downsample doit être un int"

        raw.resample(steps["downsample"])

    return raw


def preprocess_data(dic_data, steps_preprocess = {"filter" : [8,30],"drop_channels" : ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd',]}):


    for key in dic_data.keys():
        _= preprocess(dic_data[key],steps_preprocess)

    return dic_data


class Epoching:

    def __init__(self, dic_raw_data,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        self.tmin= steps_preprocess["tmin"]
        self.tmax = steps_preprocess["tmax"]
        self.length_epoch = steps_preprocess["lenght"]
        self.overlap = steps_preprocess["overlap"]
        self.n_events_per_trial = steps_preprocess["n_events_per_trial"]
        # self.bool_one_hot = steps_preprocess["one_hot"]
        self.steps_preprocess = steps_preprocess

        self.key_session = key_session
        self.key_events = key_events
        self.dic_raw_data = dic_raw_data

        if "n_chans" in steps_preprocess.keys() : # Use for extract covariate
            self.n_chans = steps_preprocess["n_chans"]
        else : # Use for extract epoch data
            self.n_chans =  len(self.dic_raw_data[key_session[0]].ch_names) # Get the number of channels
            if "drop_channels" in steps_preprocess.keys() : # if we drop channels we need to update the number of channels
                for channel in dict[key_session[0]].ch_names:
                    if channel in steps_preprocess["drop_channels"]:
                        self.n_chans -= 1

        self.list_start = np.arange(self.tmin, (self.tmax +self.overlap)- self.length_epoch, self.overlap)
        self.list_stop = np.arange(self.tmin+self.length_epoch, (self.tmax+self.overlap), self.overlap)
        self.time_step = int(self.length_epoch  * self.dic_raw_data[key_session[0]].info['sfreq'])
        self.n_events = len(self.list_start)* self.n_events_per_trial * len(key_session) # 40 represent the number of events in each raw data

        self.X= np.zeros((self.n_events, self.n_chans, self.time_step))
        self.Y= np.zeros((self.n_events))

    def run(self):

        i = 0
        for key in tqdm(self.key_session , desc="epoching"):


            if self.steps_preprocess is not None :
                _ =  preprocess(self.dic_raw_data[key],self.steps_preprocess)

            epoch= mne.Epochs(self.dic_raw_data[key], mne.events_from_annotations(self.dic_raw_data[key],self.key_events)[0], tmin= -1 , tmax= 5 , baseline=(None, 0))

            assert len(epoch.events[:,2]) ==   self.n_events_per_trial, ( f"{key} don't have {  self.n_events_per_trial}  events it actually have { len(epoch.events[:,2])} "  )


            for start, stop in zip(  self.list_start,   self.list_stop):

                self.X[i : i +  self.n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
                self.Y[i : i +  self.n_events_per_trial ] = epoch.events[:,2]
                i += self.n_events_per_trial


        # if self.bool_one_hot :
        #     self.Y = one_hot(self.Y , depth= len(self.key_events))

        return self.X, self.Y


class Epoching_with_covariate_as_metadata_run(Epoching) :
    def __init__(self, dic_raw_data , dic_covariate,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        super().__init__(dic_raw_data, steps_preprocess, key_session, key_events)

        self.n_covariate = 1
        self.dic_covariate = dic_covariate
        self.Covariate = np.zeros((self.n_events, self.n_covariate))

    def run(self):

        i = 0
        for key in self.key_session :
            if self.steps_preprocess is not None :
                _ =  preprocess(self.dic_raw_data[key],self.steps_preprocess)

            epoch= mne.Epochs(self.dic_raw_data[key], mne.events_from_annotations(self.dic_raw_data[key],self.key_events)[0], tmin= -1 , tmax= 5 , baseline=(None, 0))

            assert len(epoch.events[:,2]) ==   self.n_events_per_trial, ( f"{key} don't have {  self.n_events_per_trial}  events it actually have { len(epoch.events[:,2])} "  )

            for start, stop in zip(  self.list_start,   self.list_stop):

                    self.X[i : i +  self.n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
                    self.Y[i : i +  self.n_events_per_trial ] = epoch.events[:,2]
                    self.Covariate[i : i+self.n_events_per_trial]  = self.dic_covariate[key].reshape(-1,1)
                    i += self.n_events_per_trial

        return self.X, self.Y, self.Covariate

class Epoching_with_covariate_as_metadata_epoch(Epoching) :
    def __init__(self, dic_raw_data , dic_covariate,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        super().__init__(dic_raw_data, steps_preprocess, key_session, key_events)

        self.n_covariate = len(dic_covariate[key_session[0]])
        self.dic_covariate = dic_covariate
        self.Covariate = np.zeros((self.n_events, self.n_covariate))

    def run(self):

        i = 0
        for key in self.key_session :
            if self.steps_preprocess is not None :
                _ =  preprocess(self.dic_raw_data[key],self.steps_preprocess)

            epoch= mne.Epochs(self.dic_raw_data[key], mne.events_from_annotations(self.dic_raw_data[key],self.key_events)[0], tmin= -1 , tmax= 5 , baseline=(None, 0))

            assert len(epoch.events[:,2]) ==   self.n_events_per_trial, ( f"{key} don't have {  self.n_events_per_trial}  events it actually have { len(epoch.events[:,2])} "  )

            for start, stop in zip(  self.list_start,   self.list_stop):

                self.X[i : i +  self.n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
                self.Y[i : i +  self.n_events_per_trial ] = epoch.events[:,2]
                self.Covariate[i : i+self.n_events_per_trial]  =  [self.dic_covariate[key] for i in range(self.n_events_per_trial)]
                i += self.n_events_per_trial

        return self.X, self.Y, self.Covariate


# def epoching_with_covariate_metadata(dict,dict_covariate, key_session =[], steps_epoching = None , key_events={"769":0 ,"770":1}) :
#     """From the dictionary of mne.rawGDF extract all the epochs selected with Key_session
#      Return the epochs list as X and tje label as Y"""
#
#     #---------------------------------------------
#     tmin= steps_epoching["tmin"]
#     tmax = steps_epoching["tmax"]
#     length_epoch = steps_epoching["lenght"]
#     overlap = steps_epoching["overlap"]
#     n_chans = steps_epoching["n_chans"]  # must be changed if we drop more channels
#     n_events_per_trial = steps_epoching["n_events_per_trial"]
#
#     list_start = np.arange(tmin, (tmax +overlap)- length_epoch, overlap)
#     list_stop = np.arange(tmin+length_epoch, (tmax+overlap), overlap)
#
#     time_step = int(length_epoch  * dict[key_session[0]].info['sfreq'])
#     n_events = len(list_start)* n_events_per_trial * len(key_session) # 40 represent the number of events in each raw data
#     n_covariate = len(dict_covariate[key_session[0]])
#     #---------------------------------------------
#     X= np.zeros((n_events, n_chans, time_step))
#     Y= np.zeros((n_events))
#     Covariate = np.zeros((n_events, n_covariate))
#     #---------------------------------------------
#
#     i = 0
#
#     for key in key_session:
#
#         epoch= mne.Epochs(dict[key], mne.events_from_annotations(dict[key],key_events)[0], tmin= -1 , tmax= 5 , baseline=(-1, 0))
#
#         assert len(epoch.events[:,2]) == n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (key ,n_events_per_trial ,len(epoch.events[:,2])) )
#
#
#         for start, stop in zip(list_start, list_stop):
#
#             X[i : i +n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
#             Y[i : i +n_events_per_trial ] = epoch.events[:,2]
#             Covariate[i : i+n_events_per_trial]  =  [dict_covariate[key] for i in range(n_events_per_trial)]
#             i += n_events_per_trial
#
#     return X,Y,Covariate

#
# def epoching_return_epoch(dict, key_session =[], steps_preprocess = None , key_events={"769":0 ,"770":1}):
#     """From the dictionary of mne.rawGDF extract all the epochs selected with Key_session
#      Return the epochs list as epoch (usefull for braindecode create_from_mne_epochs)
#
#      One_hot : if True return the label as one hot vector with shape (n_events, n_classes) ,
#      if False return the label as a vector with shape (n_events )
#
#      # TODO : work in progress"""
#
#     # #---------------------------------------------
#     # tmin= steps_preprocess["tmin"]
#     # tmax = steps_preprocess["tmax"]
#     # length_epoch = steps_preprocess["lenght"]
#     # overlap = steps_preprocess["overlap"]
#     # n_events_per_trial = steps_preprocess["n_events_per_trial"]
#     # bool_one_hot = steps_preprocess["one_hot"]
#     #
#     # #---------------------------------------------
#     #
#     # list_start = np.arange(tmin, (tmax +overlap)- length_epoch, overlap)
#     # list_stop = np.arange(tmin+length_epoch, (tmax+overlap), overlap)
#     #
#     # #n_chans = len(dict[key_session[0]].ch_names) - len(steps_preprocess["drop_channels"])
#     # n_chans = 27 # must be changed if we drop more channels
#     # time_step = int(length_epoch  * dict[key_session[0]].info['sfreq'])
#     # n_events = len(list_start)* n_events_per_trial * len(key_session) # 40 represent the number of events in each raw data
#     #
#     # X= np.zeros((n_events, n_chans, time_step))
#     # Y= np.zeros((n_events))
#     #
#     # i = 0
#     #
#     # for key in tqdm(key_session , desc="epoching"):
#     #
#     #
#     #     if steps_preprocess is not None :
#     #         _ =  preprocess(dict[key],steps_preprocess)
#     #
#     #     epoch= mne.Epochs(dict[key], mne.events_from_annotations(dict[key],key_events)[0], tmin= -1 , tmax= 5 , baseline=(None, 0))
#     #
#     #     assert len(epoch.events[:,2]) == n_events_per_trial, ( f"{key} don't have {n_events_per_trial}  events it actually have { len(epoch.events[:,2])} "  )
#     #
#     #
#     #     for start, stop in zip(list_start, list_stop):
#     #
#     #         X[i : i +n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
#     #         Y[i : i +n_events_per_trial ] = epoch.events[:,2]
#     #         i += n_events_per_trial
#     #
#     # if bool_one_hot :
#     #     Y = one_hot(Y , depth= len(key_events))
#
#     return X,Y



#
# def epoching_filter_with_covariate_metadata(dict,dict_covariate, filters = [], key_session =[], steps_epoching = None , key_events={"769":0 ,"770":1}) :
#     """From the dictionary of mne.rawGDF extract all the epochs selected with Key_session
#      Return the epochs list as X and tje label as Y"""
#
#     #---------------------------------------------
#
#     assert (isinstance(filters, list) and isinstance(filters[0], list) and len(filters[0]) == 2), "filter must be a list of list of two elements"
#
#     #---------------------------------------------
#     tmin= steps_epoching["tmin"]
#     tmax = steps_epoching["tmax"]
#     length_epoch = steps_epoching["lenght"]
#     overlap = steps_epoching["overlap"]
#     n_chans = steps_epoching["n_chans"]  # must be changed if we drop more channels
#     n_events_per_trial = steps_epoching["n_events_per_trial"]
#
#     list_start = np.arange(tmin, (tmax +overlap)- length_epoch, overlap)
#     list_stop = np.arange(tmin+length_epoch, (tmax+overlap), overlap)
#
#     time_step = int(length_epoch  * dict[key_session[0]].info['sfreq'])
#     n_events = len(list_start)* n_events_per_trial * len(key_session) # 40 represent the number of events in each raw data
#     n_covariate = len(dict_covariate[key_session[0]])
#     n_filter = len(filters)
#     #---------------------------------------------
#     X= np.zeros((n_events , n_chans, time_step, n_filter))
#     Y= np.zeros((n_events))
#     Covariate = np.zeros((n_events, n_covariate))
#     #---------------------------------------------
#
#     i = 0
#
#     for key in key_session:
#         for i_filter, filter in zip(range(len(filters)), filters) :
#
#             epoch= mne.Epochs(dict[key], mne.events_from_annotations(dict[key],key_events)[0], tmin= -1 , tmax= 5 , baseline=(-1, 0)).copy()
#
#             epoch.load_data()
#             epoch.filter(filter[0], filter[1], fir_design='firwin')
#             assert len(epoch.events[:,2]) == n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (key ,n_events_per_trial ,len(epoch.events[:,2])) )
#
#
#             for start, stop in zip(list_start, list_stop):
#                 X[i : i +n_events_per_trial , : , : , i_filter]= epoch.get_data(tmin=start , tmax=stop)
#
#
#             Y[i : i +n_events_per_trial ] = epoch.events[:,2]
#             Covariate[i : i+n_events_per_trial]  =  [dict_covariate[key] for i in range(n_events_per_trial)]
#         i += n_events_per_trial
#
#     return X,Y,Covariate


class Epoching_covariate(Epoching) :
    def __init__(self, dic_raw_data , dic_covariate,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        super().__init__(dic_raw_data, steps_preprocess, key_session, key_events)

        self.n_covariate = len(dic_covariate[key_session[0]])
        self.Covariate = np.zeros((self.n_events, self.n_covariate))

    def run(self):

        i = 0
        for key in tqdm(self.key_session , desc="epoching"):


            if self.steps_preprocess is not None :
                _ =  preprocess(self.dic_raw_data[key],self.steps_preprocess)

            epoch= mne.Epochs(self.dic_raw_data[key], mne.events_from_annotations(self.dic_raw_data[key],self.key_events)[0], tmin= -1 , tmax= 5 , baseline=(None, 0))

            assert len(epoch.events[:,2]) ==   self.n_events_per_trial, ( f"{key} don't have {  self.n_events_per_trial}  events it actually have { len(epoch.events[:,2])} "  )


            for start, stop in zip(  self.list_start,   self.list_stop):

                self.X[i : i +  self.n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
                self.Y[i : i +  self.n_events_per_trial ] = epoch.events[:,2]
                self.Covariate[i : i+self.n_events_per_trial]  =  [self.dict_covariate[key] for i in range(self.n_events_per_trial)]

            i += self.n_events_per_trial
        return self.X, self.Y, self.Covariate


class Epoching_covariate_filter(Epoching) :

    def __init__(self, dic_raw_data , filters,  dic_covariate,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        assert (isinstance(filters, list) and isinstance(filters[0], list) and len(filters[0]) == 2), "filter must be a list of list of two elements"

        super().__init__(dic_raw_data, steps_preprocess, key_session, key_events)

        self.filters = filters
        self.n_filter = len(filters)

        self.n_covariates = len(dic_covariate[key_session[0]])
        self.dic_covariate = dic_covariate

        self.X= np.zeros((self.n_events , self.n_chans, self.time_step, self.n_filter))
        self.Y= np.zeros((self.n_events))
        self.Covariate = np.zeros((self.n_events, self.n_covariate))


    def run(self):

        i = 0
        for key in tqdm(self.key_session , desc="epoching"):

            for i_filter, filter in zip(range(len(self.filters)),  self.filters) :

                epoch= mne.Epochs(dict[key], mne.events_from_annotations(dict[key],self.key_events)[0], tmin= -1 , tmax= 5 , baseline=(-1, 0)).copy()

                epoch.load_data()
                epoch.filter(filter[0], filter[1], fir_design='firwin')
                assert len(epoch.events[:,2]) == self.n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (key ,self.n_events_per_trial ,len(epoch.events[:,2])) )


                for start, stop in zip(self.list_start, self.list_stop):
                    self.X[i : i +self.n_events_per_trial , : , : , i_filter]= epoch.get_data(tmin=start , tmax=stop)


                self.Y[i : i +self.n_events_per_trial ] = epoch.events[:,2]
                self.Covariate[i : i+self.n_events_per_trial]  =  [self.dic_covariate[key] for i in range(self.n_events_per_trial)]
            i += self.n_events_per_trial

            return self.X,self.Y,self.Covariate





class Epoching_Alpha_signal(Epoching):

    def __init__(self, dic_raw_data ,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        super().__init__(dic_raw_data, steps_preprocess, key_session, key_events)
        self.dic_raw_data = {}
        # self.test_dic_raw_data2 = {}

        for key in self.key_session:
            test = dic_raw_data[key].copy().load_data() # Dont know why it work like that and not the way below
            self.dic_raw_data[key] = test # But it works so dont touch it :)
            #self.dic_raw_data[key] = dic_raw_data[key].copy().load_data()

    def run(self):

        i = 0
        for key in self.key_session:

            # Based on Blankertz et al. 2008 (Invariant C
            self.dic_raw_data[key].pick_channels(["O1", "Oz", "O2"])
            self.dic_raw_data[key].filter(8,12)

            events = mne.events_from_annotations(self.dic_raw_data[key],self.key_events)[0]

            epoch= mne.Epochs(self.dic_raw_data[key], events, tmin= -1 , tmax= 5 , baseline=(-1, 0))

            assert len(epoch.events[:,2]) == self.n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (self.key , self.n_events_per_trial,len(epoch.events[:,2])) )


            for start, stop in zip(self.list_start, self.list_stop):

                self.X[i : i +self.n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)

                i += self.n_events_per_trial

        self.X = np.square(self.X)  # Power of the signal
        self.X = self.X.mean(axis=2) # mean of the signal by epoch
        self.X = np.log(self.X) # Log of the power signal
        self.X = self.X.mean(axis=1) # mean along the channels

        return self.X





class Epoching_Gamma_signal(Epoching):

    def __init__(self, dic_raw_data ,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        super().__init__(dic_raw_data, steps_preprocess, key_session, key_events)
        self.dic_raw_data = {}
        # self.test_dic_raw_data2 = {}

        for key in self.key_session:
            test = dic_raw_data[key].copy().load_data() # Dont know why it work like that and not the way below
            self.dic_raw_data[key] = test # But it works so dont touch it :)
            #self.dic_raw_data[key] = dic_raw_data[key].copy().load_data()

    def run(self):

        i = 0
        for key in self.key_session:

            # We choose those parameters based on M. grosse Wentrup et al. 2011
            # They identify a correlation between gamma band and motor imagery in occipital and in frontal to centro parietal area
            self.dic_raw_data[key].pick_channels(['FCz', 'FC1', 'C1', 'C3', 'CP1', 'CP3', 'CP5',  'CPz',  'O1', 'Oz', 'O2', 'FC2', 'Cz', 'C2', 'C4', 'CP2', 'CP4', 'CP6', 'F4', 'FC4', 'FC6', 'FC8', 'C6', 'CP8', 'CP10',  'Fz', 'F3', 'FC3', 'FC5', 'FC7', 'C5', 'TP7', 'TP9'], ordered=False)
            self.dic_raw_data[key].filter(55,85)

            events = mne.events_from_annotations(self.dic_raw_data[key],self.key_events)[0]

            epoch= mne.Epochs(self.dic_raw_data[key], events, tmin= -1 , tmax= 5 , baseline=(-1, 0))

            assert len(epoch.events[:,2]) == self.n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (self.key , self.n_events_per_trial,len(epoch.events[:,2])) )


            for start, stop in zip(self.list_start, self.list_stop):

                self.X[i : i +self.n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)

                i += self.n_events_per_trial

        self.X = np.square(self.X)  # Power of the signal
        self.X = self.X.mean(axis=2) # mean of the signal by epoch
        self.X = np.log(self.X) # Log of the power signal
        self.X = self.X.mean(axis=1) # mean along the channels

        return self.X







class Epoching_Relative_Beta_signal(Epoching):

    def __init__(self, dic_raw_data ,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        super().__init__(dic_raw_data, steps_preprocess, key_session, key_events)
        self.dic_raw_data = {}
        self.dic_raw_data_beta = {}
        self.X_beta= np.zeros((self.n_events, self.n_chans, self.time_step))

        # self.test_dic_raw_data2 = {}

        for key in self.key_session:
            test = dic_raw_data[key].copy().load_data() # Dont know why it work like that and not the way below
            self.dic_raw_data[key] = test # But it works so dont touch it :)
            #self.dic_raw_data[key] = dic_raw_data[key].copy().load_data()

            test = dic_raw_data[key].copy().load_data() # Dont know why it work like that and not the way below
            self.dic_raw_data_beta[key] = test # But it works so dont touch it :)
            #self.dic_raw_data[key] = dic_raw_data[key].copy().load_data()

    def run(self):

        i = 0
        for key in self.key_session:

            # We choose those parameters based on Foong et al. 2020 ( EEG Correlates of fatigue)
            # They identify a correlation between gamma band and motor imagery in occipital and in frontal to centro parietal area

            pick_channels = ["F3", "Fz", "F4"]
            shift = 3.5 # 3 secondes before the event
            self.dic_raw_data[key].pick_channels(pick_channels)
            self.dic_raw_data[key].filter(4,50)

            self.dic_raw_data_beta[key].pick_channels(pick_channels)
            self.dic_raw_data_beta[key].filter(12,30)

            events = mne.events_from_annotations(self.dic_raw_data[key],self.key_events)[0]

            epoch= mne.Epochs(self.dic_raw_data[key], events, tmin= -4 , tmax= 5 , baseline=(-4, 0))
            epoch_beta= mne.Epochs(self.dic_raw_data_beta[key], events, tmin= -4 , tmax= 5 , baseline=(-4, 0))

            assert len(epoch.events[:,2]) == self.n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (self.key , self.n_events_per_trial,len(epoch.events[:,2])) )


            for start, stop in zip(self.list_start, self.list_stop):

                self.X[i : i +self.n_events_per_trial ] = epoch.get_data(tmin=start - shift , tmax=stop - shift)
                self.X_beta[i : i +self.n_events_per_trial ] = epoch_beta.get_data(tmin=start - shift, tmax=stop - shift)

                i += self.n_events_per_trial

        self.X = np.square(self.X)  # Power of the signal
        self.X = self.X.mean(axis=2) # mean of the signal by epoch
        # self.X = np.log(self.X) # Log of the power signal

        self.X_beta = np.square(self.X_beta)  # Power of the signal
        self.X_beta = self.X_beta.mean(axis=2) # mean of the signal by epoch
        # self.X_beta = np.log(self.X_beta) # Log of the power signal

        self.X = self.X_beta /self.X # Relative power of the signal (beta / all band)
        self.X = 10* np.log10(self.X) # Log of the power signal

        self.X = self.X.mean(axis=1) # mean along the channels

        return self.X


class Epoching_EOG_artefact(Epoching):

    def __init__(self, dic_raw_data ,  steps_preprocess = None , key_session =[] , key_events={"769":0 ,"770":1} ):

        super().__init__(dic_raw_data, steps_preprocess, key_session, key_events)

        for key in self.key_session:
            self.dic_raw_data[key] = dic_raw_data[key].copy()
            self.dic_raw_data[key].load_data()



    def run(self):

        for key in tqdm(self.key_session , desc="epoching"):

            self.dic_raw_data[key].pick_channels(["EOG1", "EOG2", "EOG3"])
            self.dic_raw_data[key].filter(0.5,3.5)

            events = mne.events_from_annotations(self.dic_raw_data[key],self.key_events)[0]

            epoch= mne.Epochs(self.dic_raw_data[key], events, tmin= -1 , tmax= 5 , baseline=(-1, 0))

            assert len(epoch.events[:,2]) == self.n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (self.key , self.n_events_per_trial,len(epoch.events[:,2])) )

            i = 0
            for start, stop in zip(self.list_start, self.list_stop):

                self.X[i : i +self.n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)

                i += self.n_events_per_trial

            self.X = np.square(self.X)  # Power of the signal
            self.X = self.X.mean(axis=2) # mean of the signal by epoch
            self.X = np.log(self.X) # Log of the power signal
            self.X = self.X.mean(axis=1) # mean along the channels

            return self.X


#
#
#
# def extract_eog_artefact(raw, steps_epoching={}):
#     """Take in input the raw signal before preprocessing, return the raw signal of artefact
#     the artefact can be :
#     -prefrontal muscular artefact : PFMA
#     """
#
#     #---------------------------------------------
#     tmin= steps_epoching["tmin"]
#     tmax = steps_epoching["tmax"]
#     length_epoch = steps_epoching["lenght"]
#     overlap = steps_epoching["overlap"]
#     key_events = steps_epoching["key_events"]
#     list_start = np.arange(tmin, (tmax +overlap)- length_epoch, overlap)
#     list_stop = np.arange(tmin+length_epoch, (tmax+overlap), overlap)
#     n_chans = 3  # must be changed if we drop more channels
#     n_events_per_trial = 16
#     time_step = int(length_epoch  * raw.info['sfreq'])
#     n_events = len(list_start)* n_events_per_trial # 40 represent the number of events in each raw data
#
#
#     X= np.zeros((n_events, n_chans, time_step))
#
#
#     i = 0
#     #---------------------------------------------
#
#     raw_artefact.pick_channels(["EOG1", "EOG2", "EOG3"])
#     raw_artefact.filter(0.5,3.5)
#
#     events = mne.events_from_annotations(raw_artefact,key_events)[0]
#
#     epoch= mne.Epochs(raw_artefact, events, tmin= -1 , tmax= 5 , baseline=(-1, 0))
#
#     assert len(epoch.events[:,2]) == n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (key , n_events_per_trial,len(epoch.events[:,2])) )
#
#
#     for start, stop in zip(list_start, list_stop):
#
#         X[i : i +n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
#
#         i += n_events_per_trial
#
#     X = np.square(X)  # Power of the signal
#     X = X.mean(axis=2) # mean of the signal by epoch
#     X = np.log(X) # Log of the power signal
#     X = X.mean(axis=1) # mean along the channels
#
#     return X
#
# def extract_frontal_artefact(raw, steps_epoching={}):
#     """Take in input the raw signal before preprocessing, return the raw signal of artefact
#     the artefact can be :
#     -prefrontal muscular artefact : PFMA
#     """
#     raw_artefact =raw.copy()
#     raw_artefact.load_data()
#
#     #---------------------------------------------
#     tmin= steps_epoching["tmin"]
#     tmax = steps_epoching["tmax"]
#     length_epoch = steps_epoching["lenght"]
#     overlap = steps_epoching["overlap"]
#     key_events = steps_epoching["key_events"]
#     list_start = np.arange(tmin, (tmax +overlap)- length_epoch, overlap)
#     list_stop = np.arange(tmin+length_epoch, (tmax+overlap), overlap)
#     n_chans = 3  # must be changed if we drop more channels
#     n_events_per_trial= 16
#
#     time_step = int(length_epoch  * raw.info['sfreq'])
#     n_events = len(list_start)* n_events_per_trial # 40 represent the number of events in each raw data
#
#
#     X= np.zeros((n_events, n_chans, time_step))
#
#
#     i = 0
#     #---------------------------------------------
#
#     raw_artefact.pick_channels(["Fz", "F3","F4"])
#     raw_artefact.filter(40,70)
#
#     events = mne.events_from_annotations(raw_artefact,key_events)[0]
#
#     epoch= mne.Epochs(raw_artefact, events, tmin= -1 , tmax= 5 , baseline=(-1, 0))
#
#     assert len(epoch.events[:,2]) == n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (key ,n_events_per_trial, len(epoch.events[:,2])) )
#
#
#     for start, stop in zip(list_start, list_stop):
#
#         X[i : i +n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
#
#         i += n_events_per_trial
#
#     X = np.square(X)  # Power of the signal
#     X = X.mean(axis=2) # mean of the signal by epoch
#     X = np.log(X) # Log of the power signal
#     X = X.mean(axis=1) # mean along the channels
#
#     return X
# #
# def extract_prefrontal_alpha(raw, steps_epoching={}):
#     """Take in input the raw signal before preprocessing, return the raw signal of artefact
#     the artefact can be :
#     -prefrontal muscular artefact : PFMA
#     """
#     raw_artefact =raw.copy()
#     raw_artefact.load_data()
#
#     # #---------------------------------------------
#     # tmin= steps_epoching["tmin"]
#     # tmax = steps_epoching["tmax"]
#     # length_epoch = steps_epoching["lenght"]
#     # overlap = steps_epoching["overlap"]
#     # key_events = steps_epoching["key_events"]
#     # list_start = np.arange(tmin, (tmax +overlap)- length_epoch, overlap)
#     # list_stop = np.arange(tmin+length_epoch, (tmax+overlap), overlap)
#     # n_chans = 3  # must be changed if we drop more channels
#     # n_events_per_trial= 16
#     #
#     # time_step = int(length_epoch  * raw.info['sfreq'])
#     # n_events = len(list_start)* n_events_per_trial # 40 represent the number of events in each raw data
#     #
#     #
#     # X= np.zeros((n_events, n_chans, time_step))
#
#
#     i = 0
#     #---------------------------------------------
#
#     raw_artefact.pick_channels(["Fz", "F3","F4"])
#     raw_artefact.filter(40,70)
#
#     events = mne.events_from_annotations(raw_artefact,key_events)[0]
#
#     epoch= mne.Epochs(raw_artefact, events, tmin= -1 , tmax= 5 , baseline=(-1, 0))
#
#     assert len(epoch.events[:,2]) == n_events_per_trial, ( "'%s' don't have %s events it actually have %s " % (key ,n_events_per_trial, len(epoch.events[:,2])) )
#
#
#     for start, stop in zip(list_start, list_stop):
#
#         X[i : i +n_events_per_trial ] = epoch.get_data(tmin=start , tmax=stop)
#
#         i += n_events_per_trial
#
#     X = np.square(X)  # Power of the signal
#     X = X.mean(axis=2) # mean of the signal by epoch
#     X = np.log(X) # Log of the power signal
#     X = X.mean(axis=1) # mean along the channels
#
#     return X