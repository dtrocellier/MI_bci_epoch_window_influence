import numpy as np
import pandas as pd
# from preprocess import epoching_with_covariate_metadata, epoching_filter_with_covariate_metadata ( #TODO : import the class version of the function)
from preprocess import *
from tqdm import tqdm


class Classifier:
    def __init__(self, pipelines, session, steps_preprocess = None , dic_data = None, dic_covariate =    None , filters = None, metadata_run = True , dataset= "Big_Dataset"):

        """

        filters : list of filters to apply to the data in case of filter bank classifier; if None, CSP is used """

        self.pipelines = pipelines
        self.session = session
        self.steps_preprocess = steps_preprocess
        self.accuracy = pd.DataFrame(np.zeros((len(self.session),len(self.pipelines))), index= self.session ,columns= self.pipelines.keys())
        self.dic_data = dic_data
        self.dic_covariate = dic_covariate
        self.filters = filters
        self.metadata_run = metadata_run
        self.dataset =dataset

        self._index = None

    def epoching(self , train_key, test_key) :

        if self.dic_covariate is None :
            X_train, Y_train = Epoching(self.dic_data, self.steps_preprocess, train_key).run()
            X_test, Y_test = Epoching(self.dic_data, self.steps_preprocess, test_key).run()
            Covariate_train, Covariate_test = None, None


        else :
            if self.filters is None :
            # If CSP LDA classifier is used
                if self.metadata_run is False : #Usefull for metadata that variate for each epoch

                    if train_key == [] : #manage the case where there is no train set (for extract_epoch_run
                        X_train, Y_train, Covariate_train = None, None, None
                    else :
                        X_train, Y_train, Covariate_train= Epoching_with_covariate_as_metadata_epoch(self.dic_data,self.dic_covariate, self.steps_preprocess, train_key).run()


                    if test_key == [] :
                        X_test, Y_test, Covariate_test = None, None, None
                    else :
                        X_test, Y_test, Covariate_test = Epoching_with_covariate_as_metadata_epoch(self.dic_data,self.dic_covariate, self.steps_preprocess,  test_key).run()


                else: # Usefull for metadata collected for each run
                    if train_key ==[] :
                        X_train, Y_train, Covariate_train = None, None, None
                    else :
                        X_train, Y_train, Covariate_train= Epoching_with_covariate_as_metadata_run(self.dic_data,self.dic_covariate, self.steps_preprocess, train_key).run()

                    if test_key == [] :
                        X_test, Y_test, Covariate_test = None, None, None
                    else :
                        X_test, Y_test, Covariate_test = Epoching_with_covariate_as_metadata_run(self.dic_data,self.dic_covariate, self.steps_preprocess,  test_key).run()

                    #     X_train, Y_train, Covariate_train= Epoching_with_covariate_as_metadata_run(self.dic_data,self.dic_covariate, self.steps_preprocess, train_key).run()
                    # X_test, Y_test, Covariate_test = Epoching_with_covariate_as_metadata_run(self.dic_data,self.dic_covariate, self.steps_preprocess,  test_key).run()
                    #



            # If Filter bank LDA classifier is used
            else :
                X_train, Y_train, Covariate_train= Epoching_filter_with_covariate_as_metadata(self.dic_data,self.dic_covariate, self. filters, train_key, self.steps_preprocess)
                X_test, Y_test, Covariate_test = Epoching_filter_with_covariate_as_metadata(self.dic_data,self.dic_covariate, self.filters,test_key, self.steps_preprocess)

        return X_train, Y_train, Covariate_train, X_test, Y_test, Covariate_test


    def extract_epoch_run(self , subject , session:int, run:int):
        """ Return the epochs of the test set ready to be used by the classifier
        It returns only the run asked, usefull for comparing with the within session classifier

        :return: X, Y, Covariate"""

        # assert isinstance(subject, int) and isinstance(session, int) and isinstance(run,int), "subject session and run must be int"


        keys = []

        if self.dataset == "Workload" :

            assert isinstance(subject, int) and isinstance(session, int) and isinstance(run,int), "subject session and run must be int"
            for run in [run] :

                keys += [f"S{subject:02}" +"_Sess"+ str(session) +"_" +str( run)]
        elif self.dataset == "Big_Dataset" :
            for run in [run] :
                assert session == 1, "session must be 1 for the big dataset"
                assert run in [1,2,3,4,5,6], "run must be in [1,2,3,4,5,6] for the big dataset"

                keys += [subject +"_" +str( run)]
                print(keys)



        # keys += [f"S{subject:02}" +"_Sess"+ str(session) +"_" +str( run)]


        _, _, _, X, Y, Covariate = self.epoching([], keys)
        return X, Y, Covariate



    def train_test(self , X_train, Y_train, X_test, Y_test, Covariate_train, Covariate_test):

            for classifier in self.pipelines.keys() :
                #-----------------------------------------------------------------------------------------------------------
                # Train phase
                try :

                    if classifier in ["6 csp+Indedpendant_lda" , "8 csp+Indedpendant_lda", 'fb_bpow+Ilda'] :
                        self.pipelines[classifier].fit(X_train,Y_train, independant_lda__covariate = Covariate_train)
                    elif classifier in [ "6 csp+Indedpendant_lda_lasso"] :
                        self.pipelines[classifier].fit(X_train,Y_train, independant_lda_lasso__covariate = Covariate_train)

                    elif classifier in [ "6 csp+lda_cov" , "8 csp+lda_cov" , "fb_bpow+lda_cov" ]:
                        X_transform = self.pipelines[classifier].steps[0][1].fit_transform(X_train, Y_train)
                        X_cov = np.c_[X_transform ,  Covariate_train]
                        self.pipelines[classifier].steps[1][1].fit(X_cov, Y_train)

                    else :
                        self.pipelines[classifier].fit(X_train,Y_train)

                except Exception as e:
                    if isinstance(e, np.linalg.LinAlgError):
                        print(f"singular matrix on train phase for {self._subject} in {classifier}")
                        self.accuracy[classifier] [self._index]= np.nan
                        continue
                    else:
                        raise e


                #-----------------------------------------------------------------------------------------------------------
                # Test phase
                try :

                    if classifier in ["6 csp+Indedpendant_lda" , "8 csp+Indedpendant_lda", 'fb_bpow+Ilda', "6 csp+Indedpendant_lda_lasso"] :
                        X_transform = self.pipelines[classifier].steps[0][1].transform(X_test)
                        self.accuracy[classifier] [self._index]= self.pipelines[classifier].steps[1][1].score(X_transform,Y_test, Covariate_test)
                    elif classifier in [ "6 csp+lda_cov" , "8 csp+lda_cov" , "fb_bpow+lda_cov" ]:

                            X_transform = self.pipelines[classifier].steps[0][1].transform(X_test)
                            X_cov = np.c_[X_transform , Covariate_test]
                            self.accuracy[classifier] [self._index]= self.pipelines[classifier].steps[1][1].score(X_cov, Y_test)
                    else :
                            self.accuracy[classifier] [self._index]=  self.pipelines[classifier].score(X_test,Y_test)

                except Exception as e:

                    if isinstance(e, np.linalg.LinAlgError):
                        print(f"singular matrix on test phase for {self._subject} in {classifier}")
                        self.accuracy[classifier] [self._index]= np.nan
                        continue
                    else:
                        raise e
    def weight(self):
        """ Return a dictionnary of  weight of the classifier """

        dict_a_estimator = {}
        for key_pipeline in self.pipelines.keys():
            if key_pipeline in ["6 csp+Indedpendant_lda" , "8 csp+Indedpendant_lda", 'fb_bpow+Ilda'] :
                dict_a_estimator[key_pipeline] = self.pipelines[key_pipeline]["independant_lda"].a_estimator
            elif key_pipeline in ["6 csp+Indedpendant_lda_lasso"] :
                dict_a_estimator[key_pipeline] = self.pipelines[key_pipeline]["independant_lda_lasso"].a_estimator
            # print(pipelines.a_estimator)
        return dict_a_estimator






class Within_Session(Classifier):


    def extract_train_test(self) :
        """ Depending the dataset given give in outpout the epoched signal

            dataset : "Workload" or "Big_Dataset """

        if self.dataset == "Workload" :

            train_key = [self._subject+"_"+ str(i) for i in range(1,5)]
            test_key = [self._subject+"_"+ str(i) for i in range(5,13)]

        elif self.dataset == "Big_Dataset" :
            train_key = [self._subject+"_"+ str(i) for i in range(1,3)] # train on run 1 and 2
            test_key = [self._subject+"_"+ str(i) for i in range(3,7)] # test on run 3, 4, 5 and 6

        elif self.dataset == "Big_Dataset_train_3_4" :
            train_key = [self._subject+"_"+ str(i) for i in range(3,5) ] # train on run 3 and 4
            test_key = [self._subject+"_"+ str(i) for i in range(5,7)] # test on run 5 and 6

        elif self.dataset == "Big_Dataset_train_5_6" :
            train_key = [self._subject+"_"+ str(i) for i in range(5,7)]  # train on run 5 and 6
            test_key = [self._subject+"_"+ str(i) for i in range(5,7)] # test on run 5 and 6
        else:
            raise ValueError("dataset must be Workload or BigDataset")


        return self.epoching(train_key, test_key)

    def run(self):

        for self._subject in tqdm(self.session):
            self._index = self._subject
            X_train, Y_train, Covariate_train, X_test, Y_test, Covariate_test = self.extract_train_test()
            self.train_test(X_train, Y_train, X_test, Y_test, Covariate_train, Covariate_test)

        return self.accuracy

class Cross_Session(Classifier):



    def __init__(self, pipelines, session, subjects, steps_preprocess = None , dic_data = None, dic_covariate = None, filters = None, metadata_run = True, dataset= "Big_Dataset"):
        super().__init__(pipelines, session, steps_preprocess, dic_data, dic_covariate, filters, metadata_run, dataset)
        self.list_subjects = subjects


    def extract_train_test(self , session_train, session_test) :
        """ Take in input the """


        train_key = []
        test_key = []


        for session in session_train :
            train_key += [self._subject+ session +"_"+ str(i) for i in range(1,13)]
        for session in [session_test] :
            train_key += [self._subject+ session +"_"+ str(i) for i in range(1,5)]
            test_key += [self._subject+ session +"_"+ str(i) for i in range(5,13)]


        return self.epoching(train_key, test_key)


    def run(self):


        list_session =["_Sess1", "_Sess2", "_Sess3"]
        for self._subject in tqdm(self.list_subjects) :

            for i in range(len(list_session)):
                session_train = list_session.copy()
                session_test = session_train.pop(i)
                self._index = self._subject + session_test

                X_train, Y_train, Covariate_train, X_test, Y_test, Covariate_test = self.extract_train_test(session_train, session_test)
                self.train_test(X_train, Y_train, X_test, Y_test, Covariate_train, Covariate_test)

        return self.accuracy





class Cross_Subject(Classifier):

    def __init__(self, pipelines, session, subjects, steps_preprocess = None , dic_data = None, dic_covariate = None ,  filters = None, metadata_run = True, dataset= "Big_Dataset"):
        super().__init__(pipelines, session, steps_preprocess, dic_data, dic_covariate , filters, metadata_run, dataset)
        self.list_subjects = subjects
        self.accuracy = pd.DataFrame(np.zeros((len(self.list_subjects),len(self.pipelines))), index= self.list_subjects ,columns= self.pipelines.keys())


    def extract_train_test(self) :
        """ Return the epochs of the train and test set ready to be used by the classifier

         It  divided by train and test depending on the self._subject attribute
         if self._subject is a string, it will be the test subject
         if self._subject is a list of string, they will be the test subjects"""

        if self.dataset == "Workload" :


            list_subjects_train = self.list_subjects.copy()
            if isinstance(self._subject, str) : # If there is one subject
                list_subjects_train.remove(self._subject)
            elif isinstance(self._subject, list) : # If there is multiple subjects
                for subj in self._subject :
                    list_subjects_train.remove(subj)
            else:
                raise ValueError("self._subject must be a string or a list of string")


            list_session =["_Sess1", "_Sess2", "_Sess3"]
            train_key = []
            test_key = []

            for session in list_session :
                for subj in list_subjects_train :
                    train_key += [subj+ session +"_"+ str(i) for i in range(1,13)]
                if isinstance(self._subject, str) : # If there is one subject
                    test_key += [self._subject+ session +"_"+ str(i) for i in range(5,13)]
                elif isinstance(self._subject, list) : # If there is multiple subjects
                    for subj in self._subject :
                        test_key += [subj+ session +"_"+ str(i) for i in range(5,13)]

        elif self.dataset == "Big_Dataset" :

            list_subjects_train = self.list_subjects.copy()
            if isinstance(self._subject, str) : # If there is one subject
                list_subjects_train.remove(self._subject)
            elif isinstance(self._subject, list) : # If there is multiple subjects
                for subj in self._subject :
                    list_subjects_train.remove(subj)
            else:
                raise ValueError("self._subject must be a string or a list of string")


            train_key = []
            test_key = []


            for subj in list_subjects_train :
                train_key += [subj +"_"+ str(i) for i in range(1,7)]
            if isinstance(self._subject, str) : # If there is one subject
                test_key += [self._subject+ "_"+ str(i) for i in range(3,7)]
            elif isinstance(self._subject, list) : # If there is multiple subjects
                for subj in self._subject :
                    test_key += [subj +"_"+ str(i) for i in range(3,7)]





        return self.epoching(train_key, test_key)





    def run(self):



        for self._subject in tqdm(self.list_subjects) :

            self._index = self._subject

            X_train, Y_train, Covariate_train, X_test, Y_test, Covariate_test = self.extract_train_test()
            self.train_test(X_train, Y_train, X_test, Y_test, Covariate_train, Covariate_test)

        return self.accuracy
